// Dequantization of one src0 row, shared by the matrix-vector kernel and by get_rows of quantized types.
// The includer defines: SRC0_<TYPE>, TPR, MAX_COLS and the ACC(value, element_index) macro, and provides the
// row length k. Values are produced strided by lane: sub-block or block number lane, lane + TPR, ...
#include "iq_tables.hlsli"

static const float KVALUES_MXFP4[16] = { 0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12 };

// e8m0 scale, halved to match the doubled e2m1 table (ggml_e8m0_to_fp32_half)
float e8m0_to_f32_half(uint e) {
    return asfloat(e < 2u ? (0x00200000u << e) : ((e - 1u) << 23));
}

// ue4m3 scale (nvfp4), halved like ggml_ue4m3_to_fp32 to match the doubled e2m1 table
float ue4m3_to_f32_half(uint x) {
    if (x == 0u || x == 0x7Fu) {
        return 0.0f;
    }
    const uint e = (x >> 3) & 0xFu;
    const uint m = x & 7u;
    const float raw = e == 0u ? (float) m * (1.0f / 512.0f) : asfloat(((e + 120u) << 23) | (m << 20));
    return raw * 0.5f;
}

static const uint POW3_TQ1[5] = { 1, 3, 9, 27, 81 };
// one ternary digit of a tq1_0 byte: ((byte * 3^n) & 255) * 3 >> 8, minus 1
float tq1_digit(uint b, uint n) {
    return (float) ((((b * POW3_TQ1[n]) & 0xFFu) * 3u) >> 8) - 1.0f;
}

static const float KVALUES_IQ4NL[16] = { -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113 };
// partial dot products of one src0 row against the active columns of src1, strided by lane
void dot_row(RWByteAddressBuffer src0, uint src0_base, uint lane, uint ncols, uint src1_base[MAX_COLS],
             inout float acc[MAX_COLS]) {
#if defined(SRC0_F32)
    for (uint i = lane * 4; i < k; i += TPR * 4) {
        const uint4 w = src0.Load4((src0_base + i) * 4);
        ACC(asfloat(w.x), i);
        ACC(asfloat(w.y), i + 1);
        ACC(asfloat(w.z), i + 2);
        ACC(asfloat(w.w), i + 3);
    }
#elif defined(SRC0_F16)
    for (uint i = lane * 4; i < k; i += TPR * 4) {
        uint w0, w1;
        LOAD_U32_UNALIGNED(src0, (src0_base + i) * 2, w0);
        LOAD_U32_UNALIGNED(src0, (src0_base + i) * 2 + 4, w1);
        ACC(f16tof32(w0 & 0xFFFFu), i);
        ACC(f16tof32(w0 >> 16), i + 1);
        ACC(f16tof32(w1 & 0xFFFFu), i + 2);
        ACC(f16tof32(w1 >> 16), i + 3);
    }
#elif defined(SRC0_Q4_0) && defined(ONE_COL) && !defined(SRC1_F16)
    // single column: the block as one 20-byte window (Load4 + Load) and src1 in Load4s. the generic path issues 41 scalar
    // load messages per block, this one 13 (DX11 exp146: FXC kept them apart, tg 3.4x on Intel).
    // Blocks start on 2 bytes: odd = the block starts at byte 2 of the window.
    for (uint blk = lane; blk < k / 32; blk += TPR) {
        const uint  base = (src0_base + blk) * 18;
        const uint  a0   = base & ~3u;
        const bool  odd  = (base & 2u) != 0;
        const uint4 w03  = src0.Load4(a0);
        const uint  W[5] = { w03.x, w03.y, w03.z, w03.w, src0.Load(a0 + 16) };
        const float d    = f16tof32(odd ? (W[0] >> 16) : (W[0] & 0xFFFFu));
        const uint  ys   = (src1_base[0] + blk * 32) * 4;
        float       sum  = 0.0f;
        [unroll] for (uint j = 0; j < 4; j++) {
            const uint   q   = odd ? W[j + 1] : ((W[j] >> 16) | (W[j + 1] << 16));
            const float4 ylo = asfloat(src1.Load4(ys + 16 * j));
            const float4 yhi = asfloat(src1.Load4(ys + 64 + 16 * j));
            sum += ((float) ( q        & 0xFu) - 8.0f) * ylo.x + ((float) ((q >>  8) & 0xFu) - 8.0f) * ylo.y +
                   ((float) ((q >> 16) & 0xFu) - 8.0f) * ylo.z + ((float) ((q >> 24) & 0xFu) - 8.0f) * ylo.w +
                   ((float) ((q >>  4) & 0xFu) - 8.0f) * yhi.x + ((float) ((q >> 12) & 0xFu) - 8.0f) * yhi.y +
                   ((float) ((q >> 20) & 0xFu) - 8.0f) * yhi.z + ((float) ( q >> 28        ) - 8.0f) * yhi.w;
        }
        acc[0] += sum * d;
    }
#elif defined(SRC0_Q8_0) && defined(ONE_COL) && !defined(SRC1_F16)
    // single column: the 34-byte block as one 36-byte window (2x Load4 + Load) and src1 in Load4s
    for (uint blk = lane; blk < k / 32; blk += TPR) {
        const uint  base = (src0_base + blk) * 34;
        const uint  a0   = base & ~3u;
        const bool  odd  = (base & 2u) != 0;
        const uint4 wa   = src0.Load4(a0);
        const uint4 wb   = src0.Load4(a0 + 16);
        const uint  W[9] = { wa.x, wa.y, wa.z, wa.w, wb.x, wb.y, wb.z, wb.w, src0.Load(a0 + 32) };
        const float d    = f16tof32(odd ? (W[0] >> 16) : (W[0] & 0xFFFFu));
        const uint  ys   = (src1_base[0] + blk * 32) * 4;
        float       sum  = 0.0f;
        [unroll] for (uint j = 0; j < 8; j++) {
            const uint   q = odd ? W[j + 1] : ((W[j] >> 16) | (W[j + 1] << 16));
            const float4 y = asfloat(src1.Load4(ys + 16 * j));
            sum += (float) sbyte_of(q, 0) * y.x + (float) sbyte_of(q, 1) * y.y +
                   (float) sbyte_of(q, 2) * y.z + (float) sbyte_of(q, 3) * y.w;
        }
        acc[0] += sum * d;
    }
#elif defined(SRC0_Q4_0)
    // block: f16 d, 16 bytes of nibbles; low nibbles are elements 0..15, high nibbles 16..31
    for (uint blk = lane; blk < k / 32; blk += TPR) {
        const uint base = (src0_base + blk) * 18;
        uint dbits;
        LOAD_U16_UNALIGNED(src0, base, dbits);
        const float d = f16tof32(dbits);
        [unroll] for (uint j = 0; j < 4; j++) {
            uint q;
            LOAD_U32_UNALIGNED(src0, base + 2 + 4 * j, q);
            [unroll] for (uint b = 0; b < 4; b++) {
                const uint byte = byte_of(q, b);
                ACC(((float) (byte & 0xFu) - 8.0f) * d, blk * 32 + j * 4 + b);
                ACC(((float) (byte >> 4) - 8.0f) * d, blk * 32 + 16 + j * 4 + b);
            }
        }
    }
#elif defined(SRC0_Q4_1)
    // block: f16 d, f16 m, 16 bytes of nibbles; value = q * d + m
    for (uint blk = lane; blk < k / 32; blk += TPR) {
        const uint base = (src0_base + blk) * 20;
        uint w;
        LOAD_U32_UNALIGNED(src0, base, w);
        const float d = f16tof32(w & 0xFFFFu);
        const float m = f16tof32(w >> 16);
        [unroll] for (uint j = 0; j < 4; j++) {
            uint q;
            LOAD_U32_UNALIGNED(src0, base + 4 + 4 * j, q);
            [unroll] for (uint b = 0; b < 4; b++) {
                const uint byte = byte_of(q, b);
                ACC((float) (byte & 0xFu) * d + m, blk * 32 + j * 4 + b);
                ACC((float) (byte >> 4) * d + m, blk * 32 + 16 + j * 4 + b);
            }
        }
    }
#elif defined(SRC0_Q5_0) || defined(SRC0_Q5_1)
    // Q5_0 block: f16 d, u32 qh, 16 bytes of nibbles; value = (q | bit << 4) - 16, times d
    // Q5_1 block: f16 d, f16 m, u32 qh, 16 bytes of nibbles; value = (q | bit << 4) * d + m
#if defined(SRC0_Q5_0)
    const uint bsize = 22;
    const uint hoff  = 2;
#else
    const uint bsize = 24;
    const uint hoff  = 4;
#endif
    for (uint blk = lane; blk < k / 32; blk += TPR) {
        const uint base = (src0_base + blk) * bsize;
        uint w, qh;
        LOAD_U32_UNALIGNED(src0, base, w);
        LOAD_U32_UNALIGNED(src0, base + hoff, qh);
        const float d = f16tof32(w & 0xFFFFu);
#if defined(SRC0_Q5_0)
        const float m = -16.0f * d;
#else
        const float m = f16tof32(w >> 16);
#endif
        [unroll] for (uint j = 0; j < 4; j++) {
            uint q;
            LOAD_U32_UNALIGNED(src0, base + hoff + 4 + 4 * j, q);
            [unroll] for (uint b = 0; b < 4; b++) {
                const uint byte = byte_of(q, b);
                const uint e    = j * 4 + b;
                ACC((float) ((byte & 0xFu) | (((qh >> e) & 1u) << 4)) * d + m, blk * 32 + e);
                ACC((float) ((byte >> 4) | (((qh >> (e + 16)) & 1u) << 4)) * d + m, blk * 32 + 16 + e);
            }
        }
    }
#elif defined(SRC0_BF16)
    for (uint i = lane * 4; i < k; i += TPR * 4) {
        uint w0, w1;
        LOAD_U32_UNALIGNED(src0, (src0_base + i) * 2, w0);
        LOAD_U32_UNALIGNED(src0, (src0_base + i) * 2 + 4, w1);
        ACC(asfloat(w0 << 16), i);
        ACC(asfloat(w0 & 0xFFFF0000u), i + 1);
        ACC(asfloat(w1 << 16), i + 2);
        ACC(asfloat(w1 & 0xFFFF0000u), i + 3);
    }
#elif defined(SRC0_MXFP4)
    // block: e8m0 scale byte, 16 bytes of nibbles indexing the e2m1 table
    for (uint blk = lane; blk < k / 32; blk += TPR) {
        const uint base = (src0_base + blk) * 17;
        uint ebits;
        LOAD_U16_UNALIGNED(src0, base & ~1u, ebits);
        const float d = e8m0_to_f32_half((base & 1u) != 0 ? (ebits >> 8) : (ebits & 0xFFu));
        [unroll] for (uint j = 0; j < 4; j++) {
            uint q;
            LOAD_U32_UNALIGNED(src0, base + 1 + 4 * j, q);
            [unroll] for (uint b = 0; b < 4; b++) {
                const uint byte = byte_of(q, b);
                ACC(KVALUES_MXFP4[byte & 0xFu] * d, blk * 32 + j * 4 + b);
                ACC(KVALUES_MXFP4[byte >> 4] * d, blk * 32 + 16 + j * 4 + b);
            }
        }
    }
#elif defined(SRC0_IQ4_NL)
    // block: f16 d, 16 bytes of nibbles mapped through the non-linear table
    for (uint blk = lane; blk < k / 32; blk += TPR) {
        const uint base = (src0_base + blk) * 18;
        uint dbits;
        LOAD_U16_UNALIGNED(src0, base, dbits);
        const float d = f16tof32(dbits);
        [unroll] for (uint j = 0; j < 4; j++) {
            uint q;
            LOAD_U32_UNALIGNED(src0, base + 2 + 4 * j, q);
            [unroll] for (uint b = 0; b < 4; b++) {
                const uint byte = byte_of(q, b);
                ACC(KVALUES_IQ4NL[byte & 0xFu] * d, blk * 32 + j * 4 + b);
                ACC(KVALUES_IQ4NL[byte >> 4] * d, blk * 32 + 16 + j * 4 + b);
            }
        }
    }
#elif defined(SRC0_Q8_0)
    // block: f16 d, 32 int8
    for (uint blk = lane; blk < k / 32; blk += TPR) {
        const uint base = (src0_base + blk) * 34;
        uint dbits;
        LOAD_U16_UNALIGNED(src0, base, dbits);
        const float d = f16tof32(dbits);
        [unroll] for (uint j = 0; j < 8; j++) {
            uint q;
            LOAD_U32_UNALIGNED(src0, base + 2 + 4 * j, q);
            [unroll] for (uint b = 0; b < 4; b++) {
                ACC((float) sbyte_of(q, b) * d, blk * 32 + j * 4 + b);
            }
        }
    }
#elif defined(SRC0_Q4_K) && defined(ONE_COL) && !defined(SRC1_F16)
    // single column: header and nibbles as Load4s (144-byte blocks are 16-byte aligned), src1 as Load4s
    for (uint sb = lane; sb < k / 32; sb += TPR) {
        const uint  blk  = sb / 8;
        const uint  s    = sb % 8;
        const uint  base = (src0_base + blk) * 144;
        const uint4 hd   = src0.Load4(base);
        const float d    = f16tof32(hd.x & 0xFFFFu);
        const float dmin = f16tof32(hd.x >> 16);
        uint sc, mn;
        if (s < 4) {
            sc = byte_of(hd.y, s) & 63u;
            mn = byte_of(hd.z, s) & 63u;
        } else {
            sc = (byte_of(hd.w, s - 4) & 0xFu) | ((byte_of(hd.y, s - 4) >> 6) << 4);
            mn = (byte_of(hd.w, s - 4) >> 4) | ((byte_of(hd.z, s - 4) >> 6) << 4);
        }
        const uint  shift = (s & 1u) * 4u;
        const uint  qbase = base + 16 + 32 * (s / 2);
        const uint4 q0    = src0.Load4(qbase);
        const uint4 q1    = src0.Load4(qbase + 16);
        const uint  Q[8]  = { q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w };
        const uint  ys    = (src1_base[0] + blk * 256 + s * 32) * 4;
        float sq = 0.0f, sy = 0.0f;
        [unroll] for (uint j = 0; j < 8; j++) {
            const float4 y = asfloat(src1.Load4(ys + 16 * j));
            const uint   q = Q[j] >> shift;
            sq += (float) (q & 0xFu) * y.x + (float) ((q >> 8) & 0xFu) * y.y +
                  (float) ((q >> 16) & 0xFu) * y.z + (float) ((q >> 24) & 0xFu) * y.w;
            sy += y.x + y.y + y.z + y.w;
        }
        acc[0] += d * (float) sc * sq - dmin * (float) mn * sy;
    }
#elif defined(SRC0_Q6_K) && defined(ONE_COL) && !defined(SRC1_F16)
    // single column: ql and qh as 36-byte windows (2x Load4 + Load; 210-byte blocks start on 2 bytes), src1 as Load4s
    for (uint sb = lane; sb < k / 32; sb += TPR) {
        const uint blk  = sb / 8;
        const uint s    = sb % 8;
        const uint h    = s / 4;
        const uint t    = s % 4;
        const uint base = (src0_base + blk) * 210;
        uint dbits;
        LOAD_U16_UNALIGNED(src0, base + 208, dbits);
        const float d = f16tof32(dbits);
        const uint  ql_base = base + 64 * h + 32 * (t & 1u);
        const uint  qh_base = base + 128 + 32 * h;
        const uint  sc_base = base + 192 + 8 * h + 2 * t;
        const uint  lshift  = (t >> 1) * 4u;
        const uint  hshift  = t * 2u;
        uint scw;
        LOAD_U32_UNALIGNED(src0, sc_base, scw);
        const bool  odd  = (base & 2u) != 0;   // ql_base and qh_base share base's 4-byte phase
        const uint  la   = ql_base & ~3u;
        const uint  ha   = qh_base & ~3u;
        const uint4 l0   = src0.Load4(la);
        const uint4 l1   = src0.Load4(la + 16);
        const uint4 h0   = src0.Load4(ha);
        const uint4 h1   = src0.Load4(ha + 16);
        const uint  L[9] = { l0.x, l0.y, l0.z, l0.w, l1.x, l1.y, l1.z, l1.w, src0.Load(la + 32) };
        const uint  H[9] = { h0.x, h0.y, h0.z, h0.w, h1.x, h1.y, h1.z, h1.w, src0.Load(ha + 32) };
        const uint  ys   = (src1_base[0] + blk * 256 + s * 32) * 4;
        float sum0 = 0.0f, sum1 = 0.0f;
        [unroll] for (uint j = 0; j < 8; j++) {
            const uint   ql = odd ? ((L[j] >> 16) | (L[j + 1] << 16)) : L[j];
            const uint   qh = odd ? ((H[j] >> 16) | (H[j + 1] << 16)) : H[j];
            const float4 y  = asfloat(src1.Load4(ys + 16 * j));
            float p = 0.0f;
            [unroll] for (uint b = 0; b < 4; b++) {
                const int q = (int) (((byte_of(ql, b) >> lshift) & 0xFu) | (((byte_of(qh, b) >> hshift) & 3u) << 4)) - 32;
                p += (float) q * y[b];
            }
            if (j < 4) { sum0 += p; } else { sum1 += p; }
        }
        acc[0] += d * ((float) sbyte_of(scw, 0) * sum0 + (float) sbyte_of(scw, 1) * sum1);
    }
#elif defined(SRC0_Q5_K) && defined(ONE_COL) && !defined(SRC1_F16)
    // single column: header, qh and ql as Load4s (176-byte blocks are 16-byte aligned), src1 as Load4s
    for (uint sb = lane; sb < k / 32; sb += TPR) {
        const uint  blk  = sb / 8;
        const uint  s    = sb % 8;
        const uint  base = (src0_base + blk) * 176;
        const uint4 hd   = src0.Load4(base);
        const float d    = f16tof32(hd.x & 0xFFFFu);
        const float dmin = f16tof32(hd.x >> 16);
        uint sc, mn;
        if (s < 4) {
            sc = byte_of(hd.y, s) & 63u;
            mn = byte_of(hd.z, s) & 63u;
        } else {
            sc = (byte_of(hd.w, s - 4) & 0xFu) | ((byte_of(hd.y, s - 4) >> 6) << 4);
            mn = (byte_of(hd.w, s - 4) >> 4) | ((byte_of(hd.z, s - 4) >> 6) << 4);
        }
        const uint  shift = (s & 1u) * 4u;
        const uint  qbase = base + 48 + 32 * (s / 2);
        const uint4 q0    = src0.Load4(qbase);
        const uint4 q1    = src0.Load4(qbase + 16);
        const uint4 h0    = src0.Load4(base + 16);
        const uint4 h1    = src0.Load4(base + 32);
        const uint  Q[8]  = { q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w };
        const uint  H[8]  = { h0.x, h0.y, h0.z, h0.w, h1.x, h1.y, h1.z, h1.w };
        const uint  ys    = (src1_base[0] + blk * 256 + s * 32) * 4;
        float sq = 0.0f, sy = 0.0f;
        [unroll] for (uint j = 0; j < 8; j++) {
            const float4 y = asfloat(src1.Load4(ys + 16 * j));
            const uint   q = (Q[j] >> shift) & 0x0F0F0F0Fu;
            const uint   h = ((H[j] >> s) & 0x01010101u) << 4;   // bit s of each qh byte -> +16
            const uint   v = q | h;
            sq += (float) (v & 0xFFu) * y.x + (float) ((v >> 8) & 0xFFu) * y.y +
                  (float) ((v >> 16) & 0xFFu) * y.z + (float) (v >> 24) * y.w;
            sy += y.x + y.y + y.z + y.w;
        }
        acc[0] += d * (float) sc * sq - dmin * (float) mn * sy;
    }
#elif defined(SRC0_Q4_K)
    // super-block of 256: f16 d, f16 dmin, 12 bytes of 6-bit scales/mins, 128 bytes of nibbles.
    // sub-block s (32 values): pair s/2 uses bytes 16 + 32*(s/2), low nibbles for even s, high for odd s.
    for (uint sb = lane; sb < k / 32; sb += TPR) {
        const uint blk  = sb / 8;
        const uint s    = sb % 8;
        const uint base = (src0_base + blk) * 144;
        uint w;
        LOAD_U32_UNALIGNED(src0, base, w);
        const float d    = f16tof32(w & 0xFFFFu);
        const float dmin = f16tof32(w >> 16);
        uint sc0, sc1, sc2;
        LOAD_U32_UNALIGNED(src0, base + 4, sc0);
        LOAD_U32_UNALIGNED(src0, base + 8, sc1);
        LOAD_U32_UNALIGNED(src0, base + 12, sc2);
        uint sc, mn;
        if (s < 4) {
            sc = byte_of(sc0, s) & 63u;
            mn = byte_of(sc1, s) & 63u;
        } else {
            sc = (byte_of(sc2, s - 4) & 0xFu) | ((byte_of(sc0, s - 4) >> 6) << 4);
            mn = (byte_of(sc2, s - 4) >> 4) | ((byte_of(sc1, s - 4) >> 6) << 4);
        }
        const float dl = d * (float) sc;
        const float ml = dmin * (float) mn;
        const uint  shift = (s & 1u) * 4u;
        const uint  qbase = base + 16 + 32 * (s / 2);
        [unroll] for (uint j = 0; j < 8; j++) {
            uint q;
            LOAD_U32_UNALIGNED(src0, qbase + 4 * j, q);
            [unroll] for (uint b = 0; b < 4; b++) {
                const float v = dl * (float) ((byte_of(q, b) >> shift) & 0xFu) - ml;
                ACC(v, blk * 256 + s * 32 + j * 4 + b);
            }
        }
    }
#elif defined(SRC0_Q5_K)
    // super-block of 256 (176 bytes): f16 d, f16 dmin, 12 bytes of 6-bit scales/mins, 32 bytes qh, 128 bytes ql.
    // like Q4_K with the nibbles at 48; bit s of qh[l] adds 16 to value l of sub-block s.
    for (uint sb = lane; sb < k / 32; sb += TPR) {
        const uint blk  = sb / 8;
        const uint s    = sb % 8;
        const uint base = (src0_base + blk) * 176;
        uint w;
        LOAD_U32_UNALIGNED(src0, base, w);
        const float d    = f16tof32(w & 0xFFFFu);
        const float dmin = f16tof32(w >> 16);
        uint sc0, sc1, sc2;
        LOAD_U32_UNALIGNED(src0, base + 4, sc0);
        LOAD_U32_UNALIGNED(src0, base + 8, sc1);
        LOAD_U32_UNALIGNED(src0, base + 12, sc2);
        uint sc, mn;
        if (s < 4) {
            sc = byte_of(sc0, s) & 63u;
            mn = byte_of(sc1, s) & 63u;
        } else {
            sc = (byte_of(sc2, s - 4) & 0xFu) | ((byte_of(sc0, s - 4) >> 6) << 4);
            mn = (byte_of(sc2, s - 4) >> 4) | ((byte_of(sc1, s - 4) >> 6) << 4);
        }
        const float dl = d * (float) sc;
        const float ml = dmin * (float) mn;
        const uint  shift = (s & 1u) * 4u;
        const uint  qbase = base + 48 + 32 * (s / 2);
        [unroll] for (uint j = 0; j < 8; j++) {
            uint q, h;
            LOAD_U32_UNALIGNED(src0, qbase + 4 * j, q);
            LOAD_U32_UNALIGNED(src0, base + 16 + 4 * j, h);
            [unroll] for (uint b = 0; b < 4; b++) {
                const uint v5 = ((byte_of(q, b) >> shift) & 0xFu) | (((byte_of(h, b) >> s) & 1u) << 4);
                ACC(dl * (float) v5 - ml, blk * 256 + s * 32 + j * 4 + b);
            }
        }
    }
#elif defined(SRC0_Q2_K) && defined(ONE_COL) && !defined(SRC1_F16)
    // single column: scales, d/dmin and the 2-bit quants as Loads (84-byte blocks are 4-byte aligned), src1 as Load4s
    for (uint sb = lane; sb < k / 32; sb += TPR) {
        const uint  blk  = sb / 8;
        const uint  s    = sb % 8;
        const uint  base = (src0_base + blk) * 84;
        const uint  w    = src0.Load(base + 80);
        const float d    = f16tof32(w & 0xFFFFu);
        const float dmin = f16tof32(w >> 16);
        const uint  scw  = (src0.Load(base + 4 * (s / 2)) >> (16u * (s % 2u))) & 0xFFFFu;
        const float dl0  = d * (float) (scw & 0xFu);
        const float ml0  = dmin * (float) ((scw >> 4) & 0xFu);
        const float dl1  = d * (float) ((scw >> 8) & 0xFu);
        const float ml1  = dmin * (float) (scw >> 12);
        const uint  shift = 2u * (s % 4u);
        const uint  qbase = base + 16 + 32 * (s / 4);
        const uint4 q0   = src0.Load4(qbase);
        const uint4 q1   = src0.Load4(qbase + 16);
        const uint  Q[8] = { q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w };
        const uint  ys   = (src1_base[0] + blk * 256 + s * 32) * 4;
        float sqa = 0.0f, sya = 0.0f, sqb = 0.0f, syb = 0.0f;
        [unroll] for (uint j = 0; j < 8; j++) {
            const float4 y = asfloat(src1.Load4(ys + 16 * j));
            const uint   v = (Q[j] >> shift) & 0x03030303u;
            const float  sq = (float) (v & 0xFFu) * y.x + (float) ((v >> 8) & 0xFFu) * y.y +
                              (float) ((v >> 16) & 0xFFu) * y.z + (float) (v >> 24) * y.w;
            const float  sy = y.x + y.y + y.z + y.w;
            if (j < 4) { sqa += sq; sya += sy; } else { sqb += sq; syb += sy; }
        }
        acc[0] += dl0 * sqa - ml0 * sya + dl1 * sqb - ml1 * syb;
    }
#elif defined(SRC0_Q2_K)
    // super-block of 256 (84 bytes): 16 bytes of 4-bit scale/min pairs, 64 bytes of 2-bit quants, f16 d, f16 dmin.
    // sub-block s: values e < 16 use scale byte 2s, e >= 16 use 2s + 1; quant byte 16 + 32 * (s / 4) + e, shift 2 * (s % 4).
    for (uint sb = lane; sb < k / 32; sb += TPR) {
        const uint blk  = sb / 8;
        const uint s    = sb % 8;
        const uint base = (src0_base + blk) * 84;
        uint w, scw;
        LOAD_U32_UNALIGNED(src0, base + 80, w);
        const float d    = f16tof32(w & 0xFFFFu);
        const float dmin = f16tof32(w >> 16);
        LOAD_U16_UNALIGNED(src0, base + 2 * s, scw);
        const float dl0 = d * (float) (scw & 0xFu);
        const float ml0 = dmin * (float) ((scw >> 4) & 0xFu);
        const float dl1 = d * (float) ((scw >> 8) & 0xFu);
        const float ml1 = dmin * (float) (scw >> 12);
        const uint  shift = 2u * (s % 4u);
        const uint  qbase = base + 16 + 32 * (s / 4);
        [unroll] for (uint j = 0; j < 8; j++) {
            uint q;
            LOAD_U32_UNALIGNED(src0, qbase + 4 * j, q);
            const float dl = j < 4 ? dl0 : dl1;
            const float ml = j < 4 ? ml0 : ml1;
            [unroll] for (uint b = 0; b < 4; b++) {
                ACC(dl * (float) ((byte_of(q, b) >> shift) & 3u) - ml, blk * 256 + s * 32 + j * 4 + b);
            }
        }
    }
#elif defined(SRC0_Q3_K)
    // super-block of 256 (110 bytes): 32 bytes hmask, 64 bytes of 2-bit quants, 12 bytes of 6-bit scales, f16 d.
    // sub-block s: scale index 2s (e < 16) or 2s + 1; quant byte 32 + 32 * (s / 4) + e, shift 2 * (s % 4);
    // value = scale * (q - (bit s of hmask[e] ? 0 : 4))
    for (uint sb = lane; sb < k / 32; sb += TPR) {
        const uint blk  = sb / 8;
        const uint s    = sb % 8;
        const uint base = (src0_base + blk) * 110;
        uint dbits;
        LOAD_U16_UNALIGNED(src0, base + 108, dbits);
        const float d = f16tof32(dbits);
        float dls[2];
        [unroll] for (uint h = 0; h < 2; h++) {
            const uint is = 2 * s + h;
            // single bytes at odd offsets: LOAD_U16_UNALIGNED only handles even addresses
            uint lo, hi;
            if (is < 8) {
                LOAD_U32_UNALIGNED(src0, base + 96 + is, lo);
                lo = lo & 0xFu;
            } else {
                LOAD_U32_UNALIGNED(src0, base + 96 + is - 8, lo);
                lo = (lo >> 4) & 0xFu;
            }
            LOAD_U32_UNALIGNED(src0, base + 104 + is % 4, hi);
            hi = (hi >> (2 * (is / 4))) & 3u;
            dls[h] = d * ((float) (lo | (hi << 4)) - 32.0f);
        }
        const uint shift = 2u * (s % 4u);
        const uint qbase = base + 32 + 32 * (s / 4);
        [unroll] for (uint j = 0; j < 8; j++) {
            uint q, hm;
            LOAD_U32_UNALIGNED(src0, qbase + 4 * j, q);
            LOAD_U32_UNALIGNED(src0, base + 4 * j, hm);
            const float dl = dls[j < 4 ? 0 : 1];
            [unroll] for (uint b = 0; b < 4; b++) {
                const int qv = (int) ((byte_of(q, b) >> shift) & 3u) - (((byte_of(hm, b) >> s) & 1u) != 0 ? 0 : 4);
                ACC(dl * (float) qv, blk * 256 + s * 32 + j * 4 + b);
            }
        }
    }
#elif defined(SRC0_IQ4_XS) && defined(ONE_COL) && !defined(SRC1_F16)
    // single column: header and the 16 nibble bytes of the sub-block as Loads (136-byte blocks are 4-byte aligned), src1 as Load4s
    for (uint sb = lane; sb < k / 32; sb += TPR) {
        const uint  blk  = sb / 8;
        const uint  s    = sb % 8;
        const uint  base = (src0_base + blk) * 136;
        const uint2 hd   = src0.Load2(base);
        const float d    = f16tof32(hd.x & 0xFFFFu);
        const uint  ls   = ((byte_of(hd.y, s / 2) >> (4 * (s % 2))) & 0xFu) | ((((hd.x >> 16) >> (2 * s)) & 3u) << 4);
        const float dl   = d * ((float) ls - 32.0f);
        const uint4 q    = src0.Load4(base + 8 + 16 * s);
        const uint  Q[4] = { q.x, q.y, q.z, q.w };
        const uint  ys   = (src1_base[0] + blk * 256 + s * 32) * 4;
        float sum = 0.0f;
        [unroll] for (uint j = 0; j < 4; j++) {
            const float4 ylo = asfloat(src1.Load4(ys + 16 * j));
            const float4 yhi = asfloat(src1.Load4(ys + 64 + 16 * j));
            const uint   w   = Q[j];
            sum += KVALUES_IQ4NL[w & 0xFu] * ylo.x + KVALUES_IQ4NL[(w >> 8) & 0xFu] * ylo.y +
                   KVALUES_IQ4NL[(w >> 16) & 0xFu] * ylo.z + KVALUES_IQ4NL[(w >> 24) & 0xFu] * ylo.w +
                   KVALUES_IQ4NL[(w >> 4) & 0xFu] * yhi.x + KVALUES_IQ4NL[(w >> 12) & 0xFu] * yhi.y +
                   KVALUES_IQ4NL[(w >> 20) & 0xFu] * yhi.z + KVALUES_IQ4NL[w >> 28] * yhi.w;
        }
        acc[0] += dl * sum;
    }
#elif defined(SRC0_IQ4_XS)
    // super-block of 256 (136 bytes): f16 d, u16 scales_h, 4 bytes scales_l, 128 bytes of table nibbles.
    // sub-block s: scale = (nibble s of scales_l | 2 bits s of scales_h << 4) - 32; nibble bytes 8 + 16 s.
    for (uint sb = lane; sb < k / 32; sb += TPR) {
        const uint blk  = sb / 8;
        const uint s    = sb % 8;
        const uint base = (src0_base + blk) * 136;
        uint w, sl;
        LOAD_U32_UNALIGNED(src0, base, w);
        LOAD_U32_UNALIGNED(src0, base + 4, sl);
        const float d  = f16tof32(w & 0xFFFFu);
        const uint  ls = ((byte_of(sl, s / 2) >> (4 * (s % 2))) & 0xFu) | ((((w >> 16) >> (2 * s)) & 3u) << 4);
        const float dl = d * ((float) ls - 32.0f);
        [unroll] for (uint j = 0; j < 4; j++) {
            uint q;
            LOAD_U32_UNALIGNED(src0, base + 8 + 16 * s + 4 * j, q);
            [unroll] for (uint b = 0; b < 4; b++) {
                const uint byte = byte_of(q, b);
                ACC(KVALUES_IQ4NL[byte & 0xFu] * dl, blk * 256 + s * 32 + j * 4 + b);
                ACC(KVALUES_IQ4NL[byte >> 4] * dl, blk * 256 + s * 32 + 16 + j * 4 + b);
            }
        }
    }
#elif defined(SRC0_IQ3_S)
    // super-block of 256 (110 bytes): f16 d, 64 bytes qs, 8 bytes qh, 32 bytes signs, 4 bytes scales.
    // sub-block s: scale d * (1 + 2 * nibble s of scales), 4 groups l of 8 values; group l uses grid entries
    // qs[8s + 2l] (values 0..3) and qs[8s + 2l + 1] (values 4..7), 9th index bits 2l and 2l+1 of qh[s];
    // value j of group l is negated when bit j of signs[4s + l] is set
    for (uint sb = lane; sb < k / 32; sb += TPR) {
        const uint blk  = sb / 8;
        const uint s    = sb % 8;
        const uint base = (src0_base + blk) * 110;
        uint dbits, sc, qh;
        LOAD_U16_UNALIGNED(src0, base, dbits);
        LOAD_U32_UNALIGNED(src0, base + 106 + s / 2, sc);
        LOAD_U32_UNALIGNED(src0, base + 66 + s, qh);
        const float db = f16tof32(dbits) * (float) (1u + 2u * ((sc >> (4u * (s & 1u))) & 0xFu));
        [unroll] for (uint l = 0; l < 4; l++) {
            uint q, sg;
            LOAD_U32_UNALIGNED(src0, base + 2 + 8 * s + 2 * l, q);
            LOAD_U32_UNALIGNED(src0, base + 74 + 4 * s + l, sg);
            const uint g1 = IQ3S_GRID[(q & 0xFFu) | (((qh >> (2u * l)) & 1u) << 8)];
            const uint g2 = IQ3S_GRID[((q >> 8) & 0xFFu) | (((qh >> (2u * l + 1u)) & 1u) << 8)];
            [unroll] for (uint j = 0; j < 8; j++) {
                const uint  gv  = j < 4 ? byte_of(g1, j) : byte_of(g2, j - 4);
                const float sgn = ((sg >> j) & 1u) != 0 ? -1.0f : 1.0f;
                ACC(db * (float) gv * sgn, blk * 256 + s * 32 + l * 8 + j);
            }
        }
    }
#elif defined(SRC0_IQ2_S)
    // super-block of 256 (82 bytes): f16 d, 64 bytes qs (32 grid indices, then 32 sign bytes), 8 bytes qh,
    // 8 bytes scales. sub-block s: 4 groups l of 8 values from grid entry qs[4s + l] | bits 2l..2l+1 of qh[s] << 8,
    // scale d * (0.5 + nibble) * 0.25 with the low nibble for l < 2; signs in qs[32 + 4s + l]
    for (uint sb = lane; sb < k / 32; sb += TPR) {
        const uint blk  = sb / 8;
        const uint s    = sb % 8;
        const uint base = (src0_base + blk) * 82;
        uint dbits, sc, qh;
        LOAD_U16_UNALIGNED(src0, base, dbits);
        LOAD_U32_UNALIGNED(src0, base + 74 + s, sc);
        LOAD_U32_UNALIGNED(src0, base + 66 + s, qh);
        const float d = f16tof32(dbits);
        [unroll] for (uint l = 0; l < 4; l++) {
            uint q, sg;
            LOAD_U32_UNALIGNED(src0, base + 2 + 4 * s + l, q);
            LOAD_U32_UNALIGNED(src0, base + 34 + 4 * s + l, sg);
            const uint  gi = (q & 0xFFu) | (((qh >> (2u * l)) & 3u) << 8);
            const float dl = d * (0.5f + (float) ((sc >> (l < 2 ? 0u : 4u)) & 0xFu)) * 0.25f;
            [unroll] for (uint j = 0; j < 8; j++) {
                const uint  gv  = j < 4 ? byte_of(IQ2S_GRID_LO[gi], j) : byte_of(IQ2S_GRID_HI[gi], j - 4);
                const float sgn = ((sg >> j) & 1u) != 0 ? -1.0f : 1.0f;
                ACC(dl * (float) gv * sgn, blk * 256 + s * 32 + l * 8 + j);
            }
        }
    }
#elif defined(SRC0_IQ2_XXS)
    // super-block of 256 (66 bytes): f16 d, then 32 uint16 qs. Each sub-block of 32 reads two
    // uint32: the first is four grid indices, the second carries the four 7-bit sign codes and,
    // in its top nibble, the sub-block scale. Scale d * (0.5 + nibble) * 0.25.
    for (uint sb = lane; sb < k / 32; sb += TPR) {
        const uint blk  = sb / 8;
        const uint s    = sb % 8;
        const uint base = (src0_base + blk) * 66;
        uint dbits, a0, a1;
        LOAD_U16_UNALIGNED(src0, base, dbits);
        LOAD_U32_UNALIGNED(src0, base + 2 + 8 * s, a0);
        LOAD_U32_UNALIGNED(src0, base + 6 + 8 * s, a1);
        const float db = f16tof32(dbits) * (0.5f + (float) (a1 >> 28)) * 0.25f;
        [unroll] for (uint l = 0; l < 4; l++) {
            const uint gi = byte_of(a0, l);
            const uint sg = KSIGNS[(a1 >> (7u * l)) & 127u];
            [unroll] for (uint j = 0; j < 8; j++) {
                const uint  gv  = j < 4 ? byte_of(IQ2XXS_GRID_LO[gi], j) : byte_of(IQ2XXS_GRID_HI[gi], j - 4);
                const float sgn = ((sg >> j) & 1u) != 0 ? -1.0f : 1.0f;
                ACC(db * (float) gv * sgn, blk * 256 + s * 32 + l * 8 + j);
            }
        }
    }
#elif defined(SRC0_IQ2_XS)
    // super-block of 256 (74 bytes): f16 d, 32 uint16 qs, 8 scale bytes. Each qs entry is a
    // 9-bit grid index plus a 7-bit sign code. The two nibbles of scales[s] scale the first and
    // second half of the sub-block, as d * (0.5 + nibble) * 0.25.
    for (uint sb = lane; sb < k / 32; sb += TPR) {
        const uint blk  = sb / 8;
        const uint s    = sb % 8;
        const uint base = (src0_base + blk) * 74;
        uint dbits, sc;
        LOAD_U16_UNALIGNED(src0, base, dbits);
        LOAD_U32_UNALIGNED(src0, base + 66 + s, sc);
        const float d   = f16tof32(dbits);
        const float db0 = d * (0.5f + (float) (sc & 0xFu)) * 0.25f;
        const float db1 = d * (0.5f + (float) ((sc >> 4) & 0xFu)) * 0.25f;
        [unroll] for (uint l = 0; l < 4; l++) {
            uint q;
            LOAD_U16_UNALIGNED(src0, base + 2 + 2 * (4 * s + l), q);
            const uint  gi = q & 511u;
            const uint  sg = KSIGNS[(q >> 9) & 127u];
            const float dl = l < 2 ? db0 : db1;
            [unroll] for (uint j = 0; j < 8; j++) {
                const uint  gv  = j < 4 ? byte_of(IQ2XS_GRID_LO[gi], j) : byte_of(IQ2XS_GRID_HI[gi], j - 4);
                const float sgn = ((sg >> j) & 1u) != 0 ? -1.0f : 1.0f;
                ACC(dl * (float) gv * sgn, blk * 256 + s * 32 + l * 8 + j);
            }
        }
    }
#elif defined(SRC0_IQ3_XXS)
    // super-block of 256 (98 bytes): f16 d, 64 grid-index bytes, then 32 bytes of packed scales
    // and signs, one uint32 per sub-block. Each group l of 8 values takes two grid entries of
    // 4 bytes each; the sign code is bits 7l..7l+6 of that uint32, the scale its top nibble,
    // applied as d * (0.5 + nibble) * 0.5.
    for (uint sb = lane; sb < k / 32; sb += TPR) {
        const uint blk  = sb / 8;
        const uint s    = sb % 8;
        const uint base = (src0_base + blk) * 98;
        uint dbits, aux;
        LOAD_U16_UNALIGNED(src0, base, dbits);
        LOAD_U32_UNALIGNED(src0, base + 66 + 4 * s, aux);
        const float db = f16tof32(dbits) * (0.5f + (float) (aux >> 28)) * 0.5f;
        [unroll] for (uint l = 0; l < 4; l++) {
            uint q;
            LOAD_U32_UNALIGNED(src0, base + 2 + 8 * s + 2 * l, q);
            const uint sg = KSIGNS[(aux >> (7u * l)) & 127u];
            const uint g1 = IQ3XXS_GRID[q & 0xFFu];
            const uint g2 = IQ3XXS_GRID[(q >> 8) & 0xFFu];
            // one ACC per step, like the IQ3_S branch. Two ACCs in a half-width loop double what
            // the unroller expands per step, and ACC is itself a loop over the columns, so this is
            // the deepest unroll of any branch here.
            [unroll] for (uint j = 0; j < 8; j++) {
                const uint  gv  = j < 4 ? byte_of(g1, j) : byte_of(g2, j - 4);
                const float sgn = ((sg >> j) & 1u) != 0 ? -1.0f : 1.0f;
                ACC(db * (float) gv * sgn, blk * 256 + s * 32 + l * 8 + j);
            }
        }
    }
#elif defined(SRC0_IQ1_S)
    // super-block of 256 (50 bytes): f16 d, 32 grid-index low bytes, 8 uint16 qh. Per sub-block
    // qh holds the 3 high bits of each of the four grid indices, a 3-bit scale in bits 12..14 and
    // the sign of the delta in bit 15. The codebook is signed bytes, and every value is shifted
    // by +-IQ1S_DELTA (0.125) before scaling - there are no per-value sign bits here.
    for (uint sb = lane; sb < k / 32; sb += TPR) {
        const uint blk  = sb / 8;
        const uint s    = sb % 8;
        const uint base = (src0_base + blk) * 50;
        uint dbits, qh;
        LOAD_U16_UNALIGNED(src0, base, dbits);
        LOAD_U16_UNALIGNED(src0, base + 34 + 2 * s, qh);
        const float dl    = f16tof32(dbits) * (float) (2u * ((qh >> 12) & 7u) + 1u);
        const float delta = (qh & 0x8000u) != 0 ? -0.125f : 0.125f;
        [unroll] for (uint l = 0; l < 4; l++) {
            uint q;
            LOAD_U32_UNALIGNED(src0, base + 2 + 4 * s + l, q);
            const uint gi = (q & 0xFFu) | (((qh >> (3u * l)) & 7u) << 8);
            [unroll] for (uint j = 0; j < 8; j++) {
                const int gv = j < 4 ? sbyte_of(IQ1S_GRID_LO[gi], j) : sbyte_of(IQ1S_GRID_HI[gi], j - 4);
                ACC(dl * ((float) gv + delta), blk * 256 + s * 32 + l * 8 + j);
            }
        }
    }
#elif defined(SRC0_TQ2_0)
    // super-block of 256 (66 bytes): 64 bytes qs, f16 d. Value h * 128 + l * 32 + m is bits 2l..2l+1 of
    // qs[h * 32 + m], minus 1, times d. Sub-block s (of 8) = half h = s / 4, shift l = s % 4. (port of the OpenGL path)
    for (uint sb = lane; sb < k / 32; sb += TPR) {
        const uint blk  = sb / 8;
        const uint s    = sb % 8;
        const uint h    = s / 4;
        const uint l    = s % 4;
        const uint base = (src0_base + blk) * 66;
        uint dbits;
        LOAD_U16_UNALIGNED(src0, base + 64, dbits);
        const float d = f16tof32(dbits);
        for (uint w = 0; w < 8; w++) {
            uint q;
            LOAD_U32_UNALIGNED(src0, base + h * 32 + 4 * w, q);
            [unroll] for (uint b = 0; b < 4; b++) {
                ACC(((float) ((byte_of(q, b) >> (2 * l)) & 3u) - 1.0f) * d, blk * 256 + h * 128 + l * 32 + w * 4 + b);
            }
        }
    }
#elif defined(SRC0_NVFP4)
    // block of 64 (36 bytes): 4 ue4m3 scale bytes (one per 16 values), 32 bytes of nibbles; sub-block s holds
    // values s*16 + j (low nibble) and s*16 + 8 + j (high nibble) of bytes 8s..8s+7
    for (uint sb = lane; sb < k / 16; sb += TPR) {
        const uint blk  = sb / 4;
        const uint s    = sb % 4;
        const uint base = (src0_base + blk) * 36;
        uint sc;
        LOAD_U32_UNALIGNED(src0, base, sc);
        const float d = ue4m3_to_f32_half(byte_of(sc, s));
        [unroll] for (uint w = 0; w < 2; w++) {
            uint q;
            LOAD_U32_UNALIGNED(src0, base + 4 + 8 * s + 4 * w, q);
            [unroll] for (uint b = 0; b < 4; b++) {
                const uint byte = byte_of(q, b);
                ACC(KVALUES_MXFP4[byte & 0xFu] * d, blk * 64 + s * 16 + w * 4 + b);
                ACC(KVALUES_MXFP4[byte >> 4] * d, blk * 64 + s * 16 + 8 + w * 4 + b);
            }
        }
    }
#elif defined(SRC0_TQ1_0)
    // block of 256 (54 bytes): qs[48] (5 digits per byte), qh[4] (4 digits per byte), f16 d.
    // 14 units per block: t 0-4 = digit n of qs bytes 0-31 (values n*32 + m); t 5-9 = digit n of qs bytes 32-47
    // (values 160 + n*16 + m); t 10-13 = digit n of qh (values 240 + n*4 + j)
    for (uint sb = lane; sb < (k / 256) * 14; sb += TPR) {
        const uint blk  = sb / 14;
        const uint t    = sb % 14;
        const uint base = (src0_base + blk) * 54;
        uint dbits;
        LOAD_U16_UNALIGNED(src0, base + 52, dbits);
        const float d = f16tof32(dbits);
        if (t < 10) {
            const uint n    = t % 5;
            const uint boff = t < 5 ? 0u : 32u;
            const uint cnt  = t < 5 ? 8u : 4u;
            const uint vbase = t < 5 ? n * 32 : 160 + n * 16;
            for (uint w = 0; w < cnt; w++) {
                uint q;
                LOAD_U32_UNALIGNED(src0, base + boff + 4 * w, q);
                [unroll] for (uint b = 0; b < 4; b++) {
                    ACC(tq1_digit(byte_of(q, b), n) * d, blk * 256 + vbase + w * 4 + b);
                }
            }
        } else {
            const uint n = t - 10;
            uint q;
            LOAD_U32_UNALIGNED(src0, base + 48, q);
            [unroll] for (uint b = 0; b < 4; b++) {
                ACC(tq1_digit(byte_of(q, b), n) * d, blk * 256 + 240 + n * 4 + b);
            }
        }
    }
#elif defined(SRC0_Q1_0)
    // block of 128 (18 bytes): f16 d, 16 bytes of sign bits; bit j of the block is +d when set, -d when clear
    for (uint sb = lane; sb < k / 32; sb += TPR) {
        const uint blk  = sb / 4;
        const uint s    = sb % 4;
        const uint base = (src0_base + blk) * 18;
        uint dbits, q;
        LOAD_U16_UNALIGNED(src0, base, dbits);
        LOAD_U32_UNALIGNED(src0, base + 2 + 4 * s, q);
        const float d = f16tof32(dbits);
        for (uint j = 0; j < 32; j++) {
            ACC(((q >> j) & 1u) != 0u ? d : -d, blk * 128 + s * 32 + j);
        }
    }
#elif defined(SRC0_Q2_0)
    // block of 64 (18 bytes): f16 d, 16 bytes of 2-bit codes q (4 per byte, low bits first); value = (q - 1) * d
    for (uint sb = lane; sb < k / 32; sb += TPR) {
        const uint blk  = sb / 2;
        const uint s    = sb % 2;
        const uint base = (src0_base + blk) * 18;
        uint dbits, q0, q1;
        LOAD_U16_UNALIGNED(src0, base, dbits);
        LOAD_U32_UNALIGNED(src0, base + 2 + 8 * s, q0);
        LOAD_U32_UNALIGNED(src0, base + 6 + 8 * s, q1);
        const float d = f16tof32(dbits);
        for (uint j = 0; j < 16; j++) {
            ACC(((float) ((q0 >> (2 * j)) & 3u) - 1.0f) * d, blk * 64 + s * 32 + j);
            ACC(((float) ((q1 >> (2 * j)) & 3u) - 1.0f) * d, blk * 64 + s * 32 + 16 + j);
        }
    }
#elif defined(SRC0_IQ1_M)
    // super-block of 256 (56 bytes): 32 grid-index low bytes, 16 qh bytes, 8 scale bytes, and no
    // separate d - the f16 scale is assembled from one nibble of each of the four scale uint16s.
    // Each qh byte carries the 3 high index bits and the delta sign for two groups of 8, and each
    // scale uint16 holds two 3-bit sub-block scales per half.
    for (uint sb = lane; sb < k / 32; sb += TPR) {
        const uint blk  = sb / 8;
        const uint s    = sb % 8;
        const uint base = (src0_base + blk) * 56;
        uint sc0, sc1, sc2, sc3;
        LOAD_U16_UNALIGNED(src0, base + 48, sc0);
        LOAD_U16_UNALIGNED(src0, base + 50, sc1);
        LOAD_U16_UNALIGNED(src0, base + 52, sc2);
        LOAD_U16_UNALIGNED(src0, base + 54, sc3);
        const uint  sbits = (sc0 >> 12) | ((sc1 >> 8) & 0x00F0u) | ((sc2 >> 4) & 0x0F00u) | (sc3 & 0xF000u);
        const float d     = f16tof32(sbits);

        const uint sc  = s < 2 ? sc0 : (s < 4 ? sc1 : (s < 6 ? sc2 : sc3));
        const uint sh  = 6u * (s & 1u);
        const float dl1 = d * (float) (2u * ((sc >> sh) & 7u) + 1u);
        const float dl2 = d * (float) (2u * ((sc >> (sh + 3u)) & 7u) + 1u);

        uint qh0, qh1;
        LOAD_U32_UNALIGNED(src0, base + 32 + 2 * s, qh0);
        qh1 = (qh0 >> 8) & 0xFFu;
        qh0 = qh0 & 0xFFu;
        [unroll] for (uint l = 0; l < 4; l++) {
            uint q;
            LOAD_U32_UNALIGNED(src0, base + 4 * s + l, q);
            const uint h  = l < 2 ? qh0 : qh1;
            const uint gi = (q & 0xFFu) | (((l & 1u) == 0 ? (h << 8) : (h << 4)) & 0x700u);
            const float delta = (h & ((l & 1u) == 0 ? 0x08u : 0x80u)) != 0 ? -0.125f : 0.125f;
            const float dl    = l < 2 ? dl1 : dl2;
            [unroll] for (uint j = 0; j < 8; j++) {
                const int gv = j < 4 ? sbyte_of(IQ1S_GRID_LO[gi], j) : sbyte_of(IQ1S_GRID_HI[gi], j - 4);
                ACC(dl * ((float) gv + delta), blk * 256 + s * 32 + l * 8 + j);
            }
        }
    }
#elif defined(SRC0_Q6_K)
    // super-block of 256: 128 bytes ql, 64 bytes qh, 16 int8 scales, f16 d.
    // sub-block s: half h = s/4 (128 values), t = s%4 selects the quarter inside the half.
    for (uint sb = lane; sb < k / 32; sb += TPR) {
        const uint blk  = sb / 8;
        const uint s    = sb % 8;
        const uint h    = s / 4;
        const uint t    = s % 4;
        const uint base = (src0_base + blk) * 210;
        uint dbits;
        LOAD_U16_UNALIGNED(src0, base + 208, dbits);
        const float d = f16tof32(dbits);
        const uint  ql_base = base + 64 * h + 32 * (t & 1u);
        const uint  qh_base = base + 128 + 32 * h;
        const uint  sc_base = base + 192 + 8 * h + 2 * t;
        const uint  lshift  = (t >> 1) * 4u;
        const uint  hshift  = t * 2u;
        uint scw;
        LOAD_U32_UNALIGNED(src0, sc_base, scw);
        const float d0 = d * (float) sbyte_of(scw, 0);
        const float d1 = d * (float) sbyte_of(scw, 1);
        [unroll] for (uint j = 0; j < 8; j++) {
            uint ql, qh;
            LOAD_U32_UNALIGNED(src0, ql_base + 4 * j, ql);
            LOAD_U32_UNALIGNED(src0, qh_base + 4 * j, qh);
            const float dsc = (j < 4) ? d0 : d1;
            [unroll] for (uint b = 0; b < 4; b++) {
                const int q = (int) (((byte_of(ql, b) >> lshift) & 0xFu) | (((byte_of(qh, b) >> hshift) & 3u) << 4)) - 32;
                ACC(dsc * (float) q, blk * 256 + s * 32 + j * 4 + b);
            }
        }
    }
#endif
}

// copy the parameters of matrix M into the working set
#define SELECT_MAT(M) { \
    mrows = m_##M; o_src0 = offset_src0_##M; o_dst = offset_dst_##M; \
    s01 = stride_01_##M; s02 = stride_02_##M; s03 = stride_03_##M; \
    add_flag = add_flag_##M; o_add = offset_add_##M; \
    add_ne1 = add_ne1_##M; add_ne2 = add_ne2_##M; add_ne3 = add_ne3_##M; \
    add_s1 = add_s1_##M; add_s2 = add_s2_##M; add_s3 = add_s3_##M; \
}

