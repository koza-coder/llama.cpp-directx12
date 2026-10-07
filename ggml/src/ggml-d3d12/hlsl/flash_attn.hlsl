#include "common.hlsli"

// FLASH_ATTN_EXT with the online softmax of the CPU reference, in two passes so no per-thread loop runs over the
// whole KV length and no n_q x n_kv score matrix is stored:
//   pass 1 (default): one thread per (output row, KV block of BLK entries) writes the block's log-sum-exp of the
//     scores and its softmax-weighted value sum to tmp[(t * n_blocks + b) * (DV + 1)]
//   pass 2 (COMBINE): one thread per output row merges its blocks (and the sink) into dst
// Output row = i3 * n_q * n_head + i1 * n_head + i2; a dispatch covers rows row0 .. row0 + n_rows - 1 and t is
// the row index inside that range.
// defines: DK, DV (head sizes, multiples of 4), K_F16, K_F32 or K_Q8_0, V_F16, V_F32 or V_Q8_0, K_ALIGNED, V_ALIGNED,
//          HAS_MASK (f16 mask), HAS_SINKS, SOFTCAP, COMBINE, DECODE (see the second main)

RWByteAddressBuffer q_buf : register(u0);
RWByteAddressBuffer k_buf : register(u1);
RWByteAddressBuffer v_buf : register(u2);
RWByteAddressBuffer mask  : register(u3);
RWByteAddressBuffer sinks : register(u4);
RWByteAddressBuffer dst   : register(u5);
RWByteAddressBuffer tmp   : register(u6);

cbuffer Params : register(b0) {
    uint offset_q;
    uint offset_k;
    uint offset_v;
    uint offset_mask;
    uint offset_sinks;
    uint offset_dst;

    uint stride_q1;
    uint stride_q2;
    uint stride_q3;
    uint stride_k1;
    uint stride_k2;
    uint stride_k3;
    uint stride_v1;
    uint stride_v2;
    uint stride_v3;
    uint stride_m1;
    uint stride_m2;
    uint stride_m3;

    uint mask_ne2;
    uint mask_ne3;
    uint n_q;       // query rows (q->ne[1])
    uint n_head;    // q->ne[2]
    uint n_kv;      // k->ne[1]
    uint rk2;       // q heads per k head
    uint rk3;
    uint rv2;
    uint rv3;

    float scale;    // already divided by logit_softcap when SOFTCAP is set
    float max_bias;
    float logit_softcap;
    float n_head_log2;
    float m0;
    float m1;

    uint blk_size;
    uint n_blocks;
    uint row0;
    uint n_rows;
    uint nwg_x;
};

// K and V rows are read 4 elements at a time (DK and DV are multiples of 4). f16 rows whose byte address is
// a multiple of 4 (K_ALIGNED / V_ALIGNED) take one Load per two elements.
#define F16_LOAD4(buf, i, out) { \
    uint _b0, _b1, _b2, _b3; \
    LOAD_U16_UNALIGNED(buf, (i) * 2, _b0); \
    LOAD_U16_UNALIGNED(buf, (i) * 2 + 2, _b1); \
    LOAD_U16_UNALIGNED(buf, (i) * 2 + 4, _b2); \
    LOAD_U16_UNALIGNED(buf, (i) * 2 + 6, _b3); \
    out = f16tof32(uint4(_b0, _b1, _b2, _b3)); \
}
#define F16_LOAD4_ALIGNED(buf, i, out) { \
    const uint _w0 = (buf).Load((i) * 2); \
    const uint _w1 = (buf).Load((i) * 2 + 4); \
    out = f16tof32(uint4(_w0 & 0xFFFFu, _w0 >> 16, _w1 & 0xFFFFu, _w1 >> 16)); \
}

#if defined(K_Q8_0) || defined(K_Q4_0) || defined(K_Q4_1) || defined(K_Q5_0) || defined(K_Q5_1) || defined(K_IQ4_NL)
#define K_BLOCK
#endif
#if defined(V_Q8_0) || defined(V_Q4_0) || defined(V_Q4_1) || defined(V_Q5_0) || defined(V_Q5_1) || defined(V_IQ4_NL)
#define V_BLOCK
#endif

// q8_0 K/V: offsets and strides are in blocks, element i = block * 32 + index in block; the 4 elements always
// share one block (head sizes are multiples of 32)
float4 load_q8_0_4(RWByteAddressBuffer buf, uint i) {
    const uint byte = (i / 32u) * 34u;
    uint dbits;
    LOAD_U16_UNALIGNED(buf, byte, dbits);
    // the 4 quants start at an even byte, so they are either 4-byte aligned or 2 bytes past it. Not
    // LOAD_U32_UNALIGNED: with its masked form the R9700 gave wrong results and page faults for head
    // sizes 256 and 576 (2026-09-19); why is not known.
    const uint a = byte + 2u + i % 32u;
    uint w = buf.Load(a & ~3u);
    if ((a & 2u) != 0u) {
        w = (w >> 16) | (buf.Load((a & ~3u) + 4u) << 16);
    }
    const int4 q = (int4) (uint4(w << 24, w << 16, w << 8, w)) >> 24;
    return f16tof32(dbits) * (float4) q;
}

// q4_0 / q4_1 / q5_0 / q5_1 / iq4_nl K/V: 32-element blocks, element i = block * 32 + index in block. The 4 elements
// share one block and one half (low or high nibbles); block sizes are even, so 4-byte reads start on an even byte.
uint load_u32_even(RWByteAddressBuffer buf, uint a) {
    uint w = buf.Load(a & ~3u);
    if ((a & 2u) != 0u) {
        w = (w >> 16) | (buf.Load((a & ~3u) + 4u) << 16);
    }
    return w;
}

uint4 nib4(uint w, uint p) {
    const uint n = p >= 16u ? (w >> 4) : w;
    return uint4(n & 15u, (n >> 8) & 15u, (n >> 16) & 15u, (n >> 24) & 15u);
}

float4 load_q4_0_4(RWByteAddressBuffer buf, uint i) {
    const uint byte = (i / 32u) * 18u;
    const uint p = i % 32u;
    uint dbits;
    LOAD_U16_UNALIGNED(buf, byte, dbits);
    const uint4 n = nib4(load_u32_even(buf, byte + 2u + (p & 15u)), p);
    return f16tof32(dbits) * ((float4) n - 8.0f);
}

float4 load_q4_1_4(RWByteAddressBuffer buf, uint i) {
    const uint byte = (i / 32u) * 20u;
    const uint p = i % 32u;
    uint dbits, mbits;
    LOAD_U16_UNALIGNED(buf, byte, dbits);
    LOAD_U16_UNALIGNED(buf, byte + 2u, mbits);
    const uint4 n = nib4(load_u32_even(buf, byte + 4u + (p & 15u)), p);
    return f16tof32(dbits) * (float4) n + f16tof32(mbits);
}

float4 load_q5_0_4(RWByteAddressBuffer buf, uint i) {
    const uint byte = (i / 32u) * 22u;
    const uint p = i % 32u;
    uint dbits;
    LOAD_U16_UNALIGNED(buf, byte, dbits);
    const uint qh = load_u32_even(buf, byte + 2u) >> p;
    const uint4 hi = uint4(qh << 4, qh << 3, qh << 2, qh << 1) & 16u;
    const uint4 n = nib4(load_u32_even(buf, byte + 6u + (p & 15u)), p) | hi;
    return f16tof32(dbits) * ((float4) n - 16.0f);
}

float4 load_q5_1_4(RWByteAddressBuffer buf, uint i) {
    const uint byte = (i / 32u) * 24u;
    const uint p = i % 32u;
    uint dbits, mbits;
    LOAD_U16_UNALIGNED(buf, byte, dbits);
    LOAD_U16_UNALIGNED(buf, byte + 2u, mbits);
    const uint qh = load_u32_even(buf, byte + 4u) >> p;
    const uint4 hi = uint4(qh << 4, qh << 3, qh << 2, qh << 1) & 16u;
    const uint4 n = nib4(load_u32_even(buf, byte + 8u + (p & 15u)), p) | hi;
    return f16tof32(dbits) * (float4) n + f16tof32(mbits);
}

static const int iq4_nl_vals[16] = { -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113 };

float4 load_iq4_nl_4(RWByteAddressBuffer buf, uint i) {
    const uint byte = (i / 32u) * 18u;
    const uint p = i % 32u;
    uint dbits;
    LOAD_U16_UNALIGNED(buf, byte, dbits);
    const uint4 n = nib4(load_u32_even(buf, byte + 2u + (p & 15u)), p);
    return f16tof32(dbits) * float4(iq4_nl_vals[n.x], iq4_nl_vals[n.y], iq4_nl_vals[n.z], iq4_nl_vals[n.w]);
}

float4 load_bf16_4(RWByteAddressBuffer buf, uint i) {
    uint b0, b1, b2, b3;
    LOAD_U16_UNALIGNED(buf, i * 2u, b0);
    LOAD_U16_UNALIGNED(buf, i * 2u + 2u, b1);
    LOAD_U16_UNALIGNED(buf, i * 2u + 4u, b2);
    LOAD_U16_UNALIGNED(buf, i * 2u + 6u, b3);
    return asfloat(uint4(b0, b1, b2, b3) << 16);
}

float4 load_k4(uint i) {
    float4 r;
#if defined(K_Q8_0)
    r = load_q8_0_4(k_buf, i);
#elif defined(K_Q4_0)
    r = load_q4_0_4(k_buf, i);
#elif defined(K_Q4_1)
    r = load_q4_1_4(k_buf, i);
#elif defined(K_Q5_0)
    r = load_q5_0_4(k_buf, i);
#elif defined(K_Q5_1)
    r = load_q5_1_4(k_buf, i);
#elif defined(K_IQ4_NL)
    r = load_iq4_nl_4(k_buf, i);
#elif defined(K_BF16)
    r = load_bf16_4(k_buf, i);
#elif defined(K_F16) && defined(K_ALIGNED)
    F16_LOAD4_ALIGNED(k_buf, i, r);
#elif defined(K_F16)
    F16_LOAD4(k_buf, i, r);
#else
    r = asfloat(k_buf.Load4(i * 4));
#endif
    return r;
}

float4 load_v4(uint i) {
    float4 r;
#if defined(V_Q8_0)
    r = load_q8_0_4(v_buf, i);
#elif defined(V_Q4_0)
    r = load_q4_0_4(v_buf, i);
#elif defined(V_Q4_1)
    r = load_q4_1_4(v_buf, i);
#elif defined(V_Q5_0)
    r = load_q5_0_4(v_buf, i);
#elif defined(V_Q5_1)
    r = load_q5_1_4(v_buf, i);
#elif defined(V_IQ4_NL)
    r = load_iq4_nl_4(v_buf, i);
#elif defined(V_BF16)
    r = load_bf16_4(v_buf, i);
#elif defined(V_F16) && defined(V_ALIGNED)
    F16_LOAD4_ALIGNED(v_buf, i, r);
#elif defined(V_F16)
    F16_LOAD4(v_buf, i, r);
#else
    r = asfloat(v_buf.Load4(i * 4));
#endif
    return r;
}

#define NEG_INF_SCORE -3.4028235e38f

#if !defined(DECODE)
[numthreads(WG_SIZE, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    const uint gi = flat_index(id, nwg_x);
#if defined(COMBINE)
    const uint t = gi;
    if (t >= n_rows) {
        return;
    }
    const uint row = row0 + t;
    const uint i2  = row % n_head;

    // same online softmax as pass 1, over block log-sum-exps whose values are already normalized
    float4 acc[DV / 4];
    for (uint d0 = 0; d0 < DV / 4; d0++) {
        acc[d0] = 0.0f;
    }
    float M = NEG_INF_SCORE;
    float S = 0.0f;
    for (uint b = 0; b < n_blocks; b++) {
        const uint base = (t * n_blocks + b) * (DV + 1);
        const float s = LOAD_F32(tmp, base);
        if (s <= NEG_INF_SCORE) {   // every entry of the block masked
            continue;
        }
        if (s > M) {
            const float ms = exp(M - s);
            M = s;
            for (uint d1 = 0; d1 < DV / 4; d1++) {
                acc[d1] = acc[d1] * ms + asfloat(tmp.Load4((base + 1 + 4 * d1) * 4));
            }
            S = S * ms + 1.0f;
        } else {
            const float vs = exp(s - M);
            for (uint d2 = 0; d2 < DV / 4; d2++) {
                acc[d2] += vs * asfloat(tmp.Load4((base + 1 + 4 * d2) * 4));
            }
            S += vs;
        }
    }

#if defined(HAS_SINKS)
    // the sink is one more logit in the denominator with no value vector
    const float sink = LOAD_F32(sinks, offset_sinks + i2);
    if (sink > M) {
        const float ms = exp(M - sink);
        for (uint d3 = 0; d3 < DV / 4; d3++) {
            acc[d3] *= ms;
        }
        S = S * ms + 1.0f;
    } else {
        S += exp(sink - M);
    }
#endif

    const float inv = S == 0.0f ? 0.0f : 1.0f / S;
    const uint dst_base = offset_dst + row * DV;
    for (uint d4 = 0; d4 < DV; d4++) {
        STORE_F32(dst, dst_base + d4, acc[d4 / 4][d4 % 4] * inv);
    }
#else
    if (gi >= n_rows * n_blocks) {
        return;
    }
    const uint t   = gi / n_blocks;
    const uint b   = gi % n_blocks;
    const uint row = row0 + t;
    const uint i3  = row / (n_q * n_head);
    const uint i1  = (row % (n_q * n_head)) / n_head;
    const uint i2  = row % n_head;

    const uint q_base = offset_q + i3 * stride_q3 + i2 * stride_q2 + i1 * stride_q1;
#if defined(GGML_D3D11) || defined(Q_FROM_MEM)
    // D3D11 and Q_FROM_MEM: q is read again for each KV entry, so fewer registers are in use (exp162: on the Intel
    // Iris Xe, D3D12 pass 1 holding q took 27.6 ms per call, the same pass on D3D11 6.6 ms). With q in registers the
    // R9700 gave NaN in acc[16] now and then (hsv=128); probably a register spill problem in the driver.
#define Q4(a) asfloat(q_buf.Load4((q_base + 4 * (a)) * 4))
#else
    float4 q[DK / 4];
    for (uint a = 0; a < DK / 4; a++) {
        q[a] = asfloat(q_buf.Load4((q_base + 4 * a) * 4));
    }
#define Q4(a) q[a]
#endif
    const uint k_base = offset_k + (i3 / rk3) * stride_k3 + (i2 / rk2) * stride_k2;
    const uint v_base = offset_v + (i3 / rv3) * stride_v3 + (i2 / rv2) * stride_v2;

#if defined(HAS_MASK)
    const uint m_base = offset_mask + (i3 % mask_ne3) * stride_m3 + (i2 % mask_ne2) * stride_m2 + i1 * stride_m1;
    float slope = 1.0f;
    if (max_bias > 0.0f) {
        const float h = (float) i2;
        slope = h < n_head_log2 ? pow(m0, h + 1.0f) : pow(m1, 2.0f * (h - n_head_log2) + 1.0f);
    }
#endif

    float4 acc[DV / 4];
    for (uint d0 = 0; d0 < DV / 4; d0++) {
        acc[d0] = 0.0f;
    }
    float M = NEG_INF_SCORE;   // running maximum; exp(M - s) is 0 for the first entry
    float S = 0.0f;            // softmax denominator scaled by exp(-M)

    const uint j1 = min((b + 1) * blk_size, n_kv);
    for (uint j = b * blk_size; j < j1; j++) {
        float mv = 0.0f;
#if defined(HAS_MASK)
        uint mbits;
        LOAD_U16_UNALIGNED(mask, (m_base + j) * 2, mbits);
        if (mbits == 0xFC00u) {   // -inf: entry not visible
            continue;
        }
        mv = slope * f16tof32(mbits);
#endif
#if defined(K_BLOCK)
        const uint kj = (k_base + j * stride_k1) * 32u;
#else
        const uint kj = k_base + j * stride_k1;
#endif
        float s = 0.0f;
        for (uint a = 0; a < DK / 4; a++) {
            s += dot(Q4(a), load_k4(kj + 4 * a));
        }
        s *= scale;
#if defined(SOFTCAP)
#if defined(GGML_D3D11)
        // tanh of a large input can give NaN (inf / inf); tanh is 1.0f past 9.01 anyway, as in unary.hlsl
        s = logit_softcap * tanh(clamp(s, -9.010913f, 9.010913f));
#else
        s = logit_softcap * tanh(s);
#endif
#endif
        s += mv;

#if defined(V_BLOCK)
        const uint vj = (v_base + j * stride_v1) * 32u;
#else
        const uint vj = v_base + j * stride_v1;
#endif
        if (s > M) {
            // new maximum: rescale what was accumulated, this entry gets weight 1
            const float ms = exp(M - s);
            M = s;
            for (uint d1 = 0; d1 < DV / 4; d1++) {
                acc[d1] = acc[d1] * ms + load_v4(vj + 4 * d1);
            }
            S = S * ms + 1.0f;
        } else {
            const float vs = exp(s - M);
            for (uint d2 = 0; d2 < DV / 4; d2++) {
                acc[d2] += vs * load_v4(vj + 4 * d2);
            }
            S += vs;
        }
    }

    const uint base = (t * n_blocks + b) * (DV + 1);
    if (S == 0.0f) {
        STORE_F32(tmp, base, NEG_INF_SCORE);
        return;
    }
    const float inv = 1.0f / S;
    STORE_F32(tmp, base, M + log(S));
    for (uint d4 = 0; d4 < DV; d4++) {
        STORE_F32(tmp, base + 1 + d4, acc[d4 / 4][d4 % 4] * inv);
    }
#endif
}
#else
// DECODE: pass 1 with one workgroup per (output row, block of WG_SIZE KV entries), for few query rows. Thread
// j scores KV entry j, then thread d sums the softmax-weighted values of output element d. Same tmp layout.
groupshared float q_sh[DK];
groupshared float p_sh[WG_SIZE];
groupshared float red_sh[WG_SIZE];

[numthreads(WG_SIZE, 1, 1)]
void main(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    const uint wg  = gid.x + nwg_x * gid.y;
    const uint tid = gtid.x;
    if (wg >= n_rows * n_blocks) {
        return;
    }
    const uint t   = wg / n_blocks;
    const uint b   = wg % n_blocks;
    const uint row = row0 + t;
    const uint i3  = row / (n_q * n_head);
    const uint i1  = (row % (n_q * n_head)) / n_head;
    const uint i2  = row % n_head;

    const uint q_base = offset_q + i3 * stride_q3 + i2 * stride_q2 + i1 * stride_q1;
    for (uint a = tid; a < DK; a += WG_SIZE) {
        q_sh[a] = LOAD_F32(q_buf, q_base + a);
    }
    GroupMemoryBarrierWithGroupSync();

    const uint k_base = offset_k + (i3 / rk3) * stride_k3 + (i2 / rk2) * stride_k2;
    const uint v_base = offset_v + (i3 / rv3) * stride_v3 + (i2 / rv2) * stride_v2;

    // score of entry j; NEG_INF_SCORE when past the end or masked
    const uint j = b * blk_size + tid;
    float s = NEG_INF_SCORE;
    if (j < n_kv) {
        bool visible = true;
        float mv = 0.0f;
#if defined(HAS_MASK)
        const uint m_base = offset_mask + (i3 % mask_ne3) * stride_m3 + (i2 % mask_ne2) * stride_m2 + i1 * stride_m1;
        uint mbits;
        LOAD_U16_UNALIGNED(mask, (m_base + j) * 2, mbits);
        visible = mbits != 0xFC00u;
        float slope = 1.0f;
        if (max_bias > 0.0f) {
            const float h = (float) i2;
            slope = h < n_head_log2 ? pow(m0, h + 1.0f) : pow(m1, 2.0f * (h - n_head_log2) + 1.0f);
        }
        mv = slope * f16tof32(mbits);
#endif
        if (visible) {
#if defined(K_BLOCK)
            const uint kj = (k_base + j * stride_k1) * 32u;
#else
            const uint kj = k_base + j * stride_k1;
#endif
            float d = 0.0f;
            for (uint a = 0; a < DK / 4; a++) {
                d += dot(float4(q_sh[4 * a], q_sh[4 * a + 1], q_sh[4 * a + 2], q_sh[4 * a + 3]), load_k4(kj + 4 * a));
            }
            d *= scale;
#if defined(SOFTCAP)
            d = logit_softcap * tanh(d);
#endif
            s = d + mv;
        }
    }

    // block maximum
    red_sh[tid] = s;
    GroupMemoryBarrierWithGroupSync();
    for (uint h = WG_SIZE / 2; h > 0; h /= 2) {
        if (tid < h) {
            red_sh[tid] = max(red_sh[tid], red_sh[tid + h]);
        }
        GroupMemoryBarrierWithGroupSync();
    }
    const float M = red_sh[0];
    GroupMemoryBarrierWithGroupSync();

    const float p = s <= NEG_INF_SCORE ? 0.0f : exp(s - M);
    p_sh[tid] = p;
    red_sh[tid] = p;
    GroupMemoryBarrierWithGroupSync();
    for (uint h2 = WG_SIZE / 2; h2 > 0; h2 /= 2) {
        if (tid < h2) {
            red_sh[tid] += red_sh[tid + h2];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    const float S = red_sh[0];

    const uint base = (t * n_blocks + b) * (DV + 1);
    if (S == 0.0f) {
        if (tid == 0) {
            STORE_F32(tmp, base, NEG_INF_SCORE);
        }
        return;
    }
    if (tid == 0) {
        STORE_F32(tmp, base, M + log(S));
    }
    const float inv = 1.0f / S;
    const uint  jn  = min(blk_size, n_kv - b * blk_size);
    // thread tid owns 4 consecutive output elements per step
    for (uint d4 = tid; d4 < DV / 4; d4 += WG_SIZE) {
        float4 acc = 0.0f;
        for (uint jj = 0; jj < jn; jj++) {
            const float pj = p_sh[jj];
            if (pj != 0.0f) {
#if defined(V_BLOCK)
                const uint vj = (v_base + (b * blk_size + jj) * stride_v1) * 32u;
#else
                const uint vj = v_base + (b * blk_size + jj) * stride_v1;
#endif
                acc += pj * load_v4(vj + 4 * d4);
            }
        }
        acc *= inv;
        STORE_F32(tmp, base + 1 + 4 * d4, acc.x);
        STORE_F32(tmp, base + 2 + 4 * d4, acc.y);
        STORE_F32(tmp, base + 3 + 4 * d4, acc.z);
        STORE_F32(tmp, base + 4 + 4 * d4, acc.w);
    }
}
#endif
