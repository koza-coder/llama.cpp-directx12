#include "common.hlsli"

// LIGHTNING_INDEXER (f32 q/w, f32, f16, bf16 or quantised k, f16 mask, f32 dst):
//   dst[s, t, ik] = sum over heads of max(dot(q[s, t, h], k[s, ik]), 0) * w[s, t, h] + mask[s, t, ik]
//
// The weights are prescaled by the caller, so there is nothing to normalise afterwards and every
// destination element is independent: one thread each. The mask is always f16, so this kernel
// always needs USE_16BIT. defines: K_F16, K_BF16, K_Q8_0, K_Q4_0, K_Q4_1, K_Q5_0, K_Q5_1, K_IQ4_NL (quantised: stride_k2 etc.
// count blocks, the element is dequantised on the fly).

RWByteAddressBuffer q_buf : register(u0);   // {n_embd, n_head, n_tokens, n_stream}
RWByteAddressBuffer k_buf : register(u1);   // {n_embd, *, n_kv, n_stream}
RWByteAddressBuffer w_buf : register(u2);   // {n_head, n_tokens, *, n_stream}
RWByteAddressBuffer m_buf : register(u3);   // {n_kv, n_tokens, *, n_stream or 1}, f16
RWByteAddressBuffer dst   : register(u4);   // {n_kv, n_tokens, *, n_stream}

cbuffer Params : register(b0) {
    uint offset_q;
    uint offset_k;
    uint offset_w;
    uint offset_m;
    uint offset_dst;
    uint stride_q1;   // per head
    uint stride_q2;   // per token
    uint stride_q3;   // per stream
    uint stride_k2;   // per kv position, in k elements
    uint stride_k3;
    uint stride_w1;
    uint stride_w3;
    uint stride_m1;   // in f16 elements
    uint stride_m3;
    uint stride_d1;
    uint stride_d3;
    uint n_embd;
    uint n_head;
    uint n_tokens;
    uint n_kv;
    uint m_streams;   // mask ne[3], the stream axis the mask is broadcast over
    uint ne;
    uint nwg_x;
};

#if defined(K_F16)
#define KVAL(row, e) LOAD_F16(k_buf, (row) + (e))
#elif defined(K_BF16)
float k_bf16(uint i) {
    uint bits;
    LOAD_U16_UNALIGNED(k_buf, i * 2, bits);
    return asfloat(bits << 16);
}
#define KVAL(row, e) k_bf16((row) + (e))
#elif defined(K_Q8_0) || defined(K_Q4_0) || defined(K_Q4_1) || defined(K_Q5_0) || defined(K_Q5_1) || defined(K_IQ4_NL)
static const float KVALUES_IQ4NL[16] = { -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113 };

uint k_byte(uint a) {
    return (k_buf.Load(a & ~3u) >> ((a & 3u) * 8u)) & 0xFFu;
}

// element e of the row whose first block is number `row` (blocks of 32 values)
float k_quant(uint row, uint e) {
    const uint j = e & 31u;
#if defined(K_Q8_0)
    const uint base = (row + (e >> 5)) * 34u;
    uint dbits;
    LOAD_U16_UNALIGNED(k_buf, base, dbits);
    int q = (int) k_byte(base + 2u + j);
    q = q > 127 ? q - 256 : q;
    return (float) q * f16tof32(dbits);
#elif defined(K_Q4_0) || defined(K_IQ4_NL)
    const uint base = (row + (e >> 5)) * 18u;
    uint dbits;
    LOAD_U16_UNALIGNED(k_buf, base, dbits);
    const uint b   = k_byte(base + 2u + (j & 15u));
    const uint nib = j < 16u ? (b & 0xFu) : (b >> 4);
#if defined(K_IQ4_NL)
    return KVALUES_IQ4NL[nib] * f16tof32(dbits);
#else
    return ((float) nib - 8.0f) * f16tof32(dbits);
#endif
#elif defined(K_Q4_1)
    const uint base = (row + (e >> 5)) * 20u;
    uint dbits, mbits;
    LOAD_U16_UNALIGNED(k_buf, base, dbits);
    LOAD_U16_UNALIGNED(k_buf, base + 2u, mbits);
    const uint b   = k_byte(base + 4u + (j & 15u));
    const uint nib = j < 16u ? (b & 0xFu) : (b >> 4);
    return (float) nib * f16tof32(dbits) + f16tof32(mbits);
#elif defined(K_Q5_0)
    const uint base = (row + (e >> 5)) * 22u;
    uint dbits, qh;
    LOAD_U16_UNALIGNED(k_buf, base, dbits);
    LOAD_U32_UNALIGNED(k_buf, base + 2u, qh);
    const uint b = k_byte(base + 6u + (j & 15u));
    const uint x = (j < 16u ? (b & 0xFu) : (b >> 4)) | (((qh >> j) & 1u) << 4);
    return ((float) x - 16.0f) * f16tof32(dbits);
#else   // K_Q5_1
    const uint base = (row + (e >> 5)) * 24u;
    uint dbits, mbits, qh;
    LOAD_U16_UNALIGNED(k_buf, base, dbits);
    LOAD_U16_UNALIGNED(k_buf, base + 2u, mbits);
    LOAD_U32_UNALIGNED(k_buf, base + 4u, qh);
    const uint b = k_byte(base + 8u + (j & 15u));
    const uint x = (j < 16u ? (b & 0xFu) : (b >> 4)) | (((qh >> j) & 1u) << 4);
    return (float) x * f16tof32(dbits) + f16tof32(mbits);
#endif
}
#define KVAL(row, e) k_quant((row), (e))
#else
#define KVAL(row, e) LOAD_F32(k_buf, (row) + (e))
#endif

[numthreads(WG_SIZE, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    const uint i = flat_index(gid, nwg_x);
    if (i >= ne) {
        return;
    }
    const uint s  = i / (n_tokens * n_kv);
    const uint r  = i % (n_tokens * n_kv);
    const uint t  = r / n_kv;
    const uint ik = r % n_kv;

    const uint k_row = offset_k + ik * stride_k2 + s * stride_k3;
    const uint w_row = offset_w + t * stride_w1 + s * stride_w3;
    const uint q_pos = offset_q + t * stride_q2 + s * stride_q3;

    float score = 0.0f;
    for (uint h = 0; h < n_head; h++) {
        const uint q_row = q_pos + h * stride_q1;
        float qk = 0.0f;
        for (uint e = 0; e < n_embd; e++) {
            qk += LOAD_F32(q_buf, q_row + e) * KVAL(k_row, e);
        }
        score += max(qk, 0.0f) * LOAD_F32(w_buf, w_row + h);
    }
    const uint m_row = offset_m + t * stride_m1 + (s % m_streams) * stride_m3;
    STORE_F32(dst, offset_dst + t * stride_d1 + s * stride_d3 + ik, score + LOAD_F16(m_buf, m_row + ik));
}
