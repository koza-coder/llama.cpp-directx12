#include "common.hlsli"

// SET_ROWS f32/f16 -> quantized (and CPY, with IDENT): dst[idx[row]] = quantize(src[row]), one thread per source row.
// Blocks are 18-34 bytes (f16 scale first), so rows do not start on word boundaries; a row writes its bytes word by word and keeps
// the neighbour bytes of its first and last word. Rows whose global dst row numbers have the other parity run in
// a second dispatch, so two threads never touch the same word. dst is contiguous; offsets and strides are in
// blocks. defines: QT_{Q8_0,Q4_0,Q4_1,Q5_0,Q5_1,IQ4_NL,Q1_0,Q2_0}, SRC_F16 (needs USE_16BIT), I64_IDX,
// IDENT (row = i_src1, idx unused: CPY)
// every quantizer follows quantize_row_*_ref of ggml-quants.c

#if defined(QT_Q8_0)
#define QK 32u
#define TS 34u
#elif defined(QT_Q4_0) || defined(QT_IQ4_NL)
#define QK 32u
#define TS 18u
#elif defined(QT_Q4_1)
#define QK 32u
#define TS 20u
#elif defined(QT_Q5_0)
#define QK 32u
#define TS 22u
#elif defined(QT_Q5_1)
#define QK 32u
#define TS 24u
#elif defined(QT_Q1_0)
#define QK 128u
#define TS 18u
#elif defined(QT_Q2_0)
#define QK 64u
#define TS 18u
#endif

#ifdef SRC_F16
#define LOAD_SRC(i) LOAD_F16(src, i)
#else
#define LOAD_SRC(i) LOAD_F32(src, i)
#endif

RWByteAddressBuffer src : register(u0);
RWByteAddressBuffer idx : register(u1);
RWByteAddressBuffer dst : register(u2);

cbuffer Params : register(b0) {
    uint offset_src;
    uint offset_idx;
    uint offset_dst;

    uint stride_src1;
    uint stride_src2;
    uint stride_src3;

    uint stride_idx0;
    uint stride_idx1;
    uint stride_idx2;

    uint stride_dst1;   // blocks per dst row
    uint dst_ne1;
    uint dst_ne2;

    uint ne0;
    uint n_rows;
    uint ne2;
    uint ne3;

    uint idx1;
    uint idx2;
    uint parity;

    uint nwg_x;
};

static uint g_word;
static uint g_blk[TS];   // the bytes of the block being built

float rnd(float x) {
    return sign(x) * floor(abs(x) + 0.5f);   // roundf: halves away from zero
}

void put16(uint o, uint h) {
    g_blk[o]     = h & 0xFFu;
    g_blk[o + 1] = (h >> 8) & 0xFFu;
}

#ifdef QT_IQ4_NL
static const int kvalues[16] = { -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113 };

int best_index_int8(float x) {
    if (x <= (float) kvalues[0]) return 0;
    if (x >= (float) kvalues[15]) return 15;
    int ml = 0, mu = 15;
    while (mu - ml > 1) {
        const int mav = (ml + mu) / 2;
        if (x < (float) kvalues[mav]) mu = mav; else ml = mav;
    }
    return x - (float) kvalues[mu - 1] < (float) kvalues[mu] - x ? mu - 1 : mu;
}
#endif

void quant_block(uint s) {
    float v[QK];
    for (uint j = 0; j < QK; j++) {
        v[j] = LOAD_SRC(s + j);
    }
    for (uint z = 0; z < TS; z++) {
        g_blk[z] = 0;
    }
#if defined(QT_Q8_0)
    float amax = 0.0f;
    for (uint j = 0; j < 32u; j++) {
        amax = max(amax, abs(v[j]));
    }
    const float d  = amax / 127.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    put16(0, f32_to_f16_rne(d));
    for (uint j2 = 0; j2 < 32u; j2++) {
        g_blk[2 + j2] = (uint) (int) rnd(v[j2] * id) & 0xFFu;
    }
#elif defined(QT_Q4_0) || defined(QT_Q5_0)
    float amax = 0.0f;
    float mx   = 0.0f;
    for (uint j = 0; j < 32u; j++) {
        if (amax < abs(v[j])) {
            amax = abs(v[j]);
            mx   = v[j];
        }
    }
#ifdef QT_Q4_0
    const float d = mx / -8.0f;
#else
    const float d = mx / -16.0f;
#endif
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    put16(0, f32_to_f16_rne(d));
#ifdef QT_Q4_0
    for (uint j = 0; j < 16u; j++) {
        precise float t0 = v[j] * id + 8.5f;
        precise float t1 = v[16u + j] * id + 8.5f;
        const uint xi0 = (uint) min(15, (int) t0);
        const uint xi1 = (uint) min(15, (int) t1);
        g_blk[2 + j] = xi0 | (xi1 << 4);
    }
#else
    uint qh = 0;
    for (uint j = 0; j < 16u; j++) {
        precise float t0 = v[j] * id + 16.5f;
        precise float t1 = v[16u + j] * id + 16.5f;
        const uint xi0 = (uint) min(31, (int) t0);
        const uint xi1 = (uint) min(31, (int) t1);
        g_blk[6 + j] = (xi0 & 0xFu) | ((xi1 & 0xFu) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << j;
        qh |= ((xi1 & 0x10u) >> 4) << (j + 16u);
    }
    for (uint k = 0; k < 4u; k++) {
        g_blk[2 + k] = (qh >> (8u * k)) & 0xFFu;
    }
#endif
#elif defined(QT_Q4_1) || defined(QT_Q5_1)
    float mn = 3.402823466e+38f;
    float mx = -3.402823466e+38f;
    for (uint j = 0; j < 32u; j++) {
        if (v[j] < mn) mn = v[j];
        if (v[j] > mx) mx = v[j];
    }
#ifdef QT_Q4_1
    const float d = (mx - mn) / 15.0f;
#else
    const float d = (mx - mn) / 31.0f;
#endif
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    put16(0, f32_to_f16_rne(d));
    put16(2, f32_to_f16_rne(mn));
#ifdef QT_Q4_1
    for (uint j = 0; j < 16u; j++) {
        precise float t0 = (v[j] - mn) * id + 0.5f;
        precise float t1 = (v[16u + j] - mn) * id + 0.5f;
        const uint xi0 = (uint) min(15, (int) t0);
        const uint xi1 = (uint) min(15, (int) t1);
        g_blk[4 + j] = xi0 | (xi1 << 4);
    }
#else
    uint qh = 0;
    for (uint j = 0; j < 16u; j++) {
        precise float t0 = (v[j] - mn) * id + 0.5f;
        precise float t1 = (v[16u + j] - mn) * id + 0.5f;
        const uint xi0 = (uint) t0 & 0xFFu;
        const uint xi1 = (uint) t1 & 0xFFu;
        g_blk[8 + j] = (xi0 & 0xFu) | ((xi1 & 0xFu) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << j;
        qh |= ((xi1 & 0x10u) >> 4) << (j + 16u);
    }
    for (uint k = 0; k < 4u; k++) {
        g_blk[4 + k] = (qh >> (8u * k)) & 0xFFu;
    }
#endif
#elif defined(QT_IQ4_NL)
    float amax = 0.0f;
    float mx   = 0.0f;
    for (uint j = 0; j < 32u; j++) {
        if (abs(v[j]) > amax) {
            amax = abs(v[j]);
            mx   = v[j];
        }
    }
    float d = 0.0f;
    uint  L[32];
    for (uint j2 = 0; j2 < 32u; j2++) {
        L[j2] = 0;
    }
    if (amax >= 1e-15f) {
        const float id0 = 1.0f / (mx / (float) kvalues[0]);
        float sumqx = 0.0f;
        float sumq2 = 0.0f;
        for (uint j3 = 0; j3 < 32u; j3++) {
            const int   l = best_index_int8(id0 * v[j3]);
            const float q = (float) kvalues[l];
            precise float w = v[j3] * v[j3];
            L[j3] = (uint) l;
            precise float aqx = w * q * v[j3];
            precise float aq2 = w * q * q;
            sumqx += aqx;
            sumq2 += aq2;
        }
        d = sumq2 > 0.0f ? sumqx / sumq2 : 0.0f;
    }
    put16(0, f32_to_f16_rne(d));
    for (uint j4 = 0; j4 < 16u; j4++) {
        g_blk[2 + j4] = L[j4] | (L[16u + j4] << 4);
    }
#elif defined(QT_Q1_0)
    float sum_abs = 0.0f;
    for (uint j = 0; j < 128u; j++) {
        sum_abs += abs(v[j]);
    }
    put16(0, f32_to_f16_rne(sum_abs / 128.0f));
    for (uint j2 = 0; j2 < 128u; j2++) {
        if (v[j2] >= 0.0f) {
            g_blk[2 + j2 / 8u] |= 1u << (j2 % 8u);
        }
    }
#elif defined(QT_Q2_0)
    float amax = 0.0f;
    for (uint j = 0; j < 64u; j++) {
        amax = max(amax, abs(v[j]));
    }
    const float id = amax > 0.0f ? 1.0f / amax : 0.0f;
    put16(0, f32_to_f16_rne(amax));
    for (uint j2 = 0; j2 < 64u; j2++) {
        const int q = clamp((int) rnd(v[j2] * id) + 1, 0, 3);
        g_blk[2 + j2 / 4u] |= ((uint) q) << ((j2 % 4u) * 2u);
    }
#endif
}

// write one byte at absolute byte address pos of a row that ends at byte end
void put_byte(uint pos, uint end, uint b) {
    if ((pos & 3u) == 0u) {
        // a word that the row does not fill to the end keeps the neighbour's high bytes
        g_word = pos + 4u > end ? dst.Load(pos) : 0u;
    }
    const uint sh = (pos & 3u) * 8u;
    g_word = (g_word & ~(0xFFu << sh)) | ((b & 0xFFu) << sh);
    if ((pos & 3u) == 3u || pos + 1u == end) {
        dst.Store(pos & ~3u, g_word);
    }
}

[numthreads(WG_SIZE, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint row = flat_index(gid, nwg_x);
    if (row >= ne3 * ne2 * n_rows) {
        return;
    }
    const uint i_src3 = row / (ne2 * n_rows);
    row = row % (ne2 * n_rows);
    const uint i_src2 = row / n_rows;
    const uint i_src1 = row % n_rows;

#ifdef IDENT
    const uint idx_val = i_src1;
#else
    const uint idx_elem = offset_idx + i_src1 * stride_idx0 + (i_src2 % idx1) * stride_idx1 + (i_src3 % idx2) * stride_idx2;
#ifdef I64_IDX
    const uint idx_val = idx.Load(idx_elem * 8);
#else
    const uint idx_val = idx.Load(idx_elem * 4);
#endif
#endif
    const uint grow = (i_src3 * dst_ne2 + i_src2) * dst_ne1 + idx_val;   // dst row number in the contiguous tensor
    if ((grow & 1u) != parity) {
        return;
    }

    const uint i_src_row = offset_src + i_src1 * stride_src1 + i_src2 * stride_src2 + i_src3 * stride_src3;
    const uint base      = (offset_dst + grow * stride_dst1) * TS;
    const uint n_blocks  = ne0 / QK;
    const uint end       = base + n_blocks * TS;

    // the first word may start before the row: keep the neighbour's low bytes
    g_word = dst.Load(base & ~3u);
    uint pos = base;
    for (uint b = 0; b < n_blocks; b++) {
        quant_block(i_src_row + b * QK);
        for (uint z = 0; z < TS; z++) {
            put_byte(pos, end, g_blk[z]);
            pos++;
        }
    }
}
