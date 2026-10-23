#include "common.hlsli"

// Tiled matrix product for the prompt path: dst[row, col] = sum_k src0[row, k] * src1[col, k].
//
// The matrix-vector kernel re-reads src0 once per group of 4 columns, which is the right shape for
// single token decoding and the wrong one for a long prompt. Here a workgroup owns a TILE_M x TILE_N
// tile of dst and walks k in steps of TILE_K: each step dequantizes TILE_M x TILE_K weights into
// groupshared once and every one of them is then used TILE_N times from there. Accumulators live in
// registers, so there is no reduction tree and no groupshared traffic per output.
//
// TILE_K is 32: one Q4_0/Q8_0 block, and one sub-block of a K-quant super-block, so a tile never
// straddles a scale boundary.
//
// The dequantization mirrors dequant_row.hlsli, which the matrix-vector kernel uses and which the op
// suite covers; only the loop shape differs (a contiguous run of 32 instead of a lane-strided walk).
//
// defines: SRC0_<TYPE>, SRC1_F16 (f16 columns instead of f32). F32, F16, Q4_0, Q8_0, Q4_K and Q6_K have
// their own loader below; every other type with 32-value blocks uses dot_row from dequant_row.hlsli.
// MMID: mul_mat_id. Each workgroup takes one tile of the table that mul_mat_id_prep.hlsl makes: one
// expert and up to TILE_N of its (token, slot) pairs as the columns.
// src0 offsets/strides are in elements for float types and in blocks for quant types.

#if defined(BIG_TILE)
#if defined(BT_DP4A)
// exp617/exp622: integer-dot variant of the BIG_TILE kernel. Weights stay quantized (Q4_0 nibbles - 8, or Q8_0 bytes) and are packed 4 per word;
// the f32 columns are quantized to int8 per 32-block (Q8_0 style, scale = amax / 127) while they are staged; the 32-element block product is
// 8 dot4add_i8packed per output and one scale multiply-add. Needs SM 6.4. Tile 128 x 64 (8 x 4 per thread) or 64 x 32 (4 x 2).
#if defined(SRC0_Q4_K)
#define A_MODE 1        // value = dl * q - ml (q 0..15): sum = dl * isum - ml * (sum of the int8 column values)
#elif defined(SRC0_Q6_K)
#define A_MODE 2        // value = d * sc * (q - 32), one scale per 16 values: sum = sc0 * isum(0..15) + sc1 * isum(16..31)
#else
#define A_MODE 0        // Q4_0 / Q8_0: value = d * q
#endif
#define TILE_M BT_TM
#define TILE_N BT_TN
#define TILE_K 32
#define NTHREADS 256
#define REG_M BT_RM
#define REG_N BT_RN
#define TH_Y (TILE_M / REG_M)
#define TH_X (TILE_N / REG_N)
#define A_SPLIT (NTHREADS / TILE_M)     // threads per src0 row in the staging
#define A_ELEMS (32 / A_SPLIT)          // block elements per staging thread
#define A_WORDS (A_ELEMS / 4)
#define B_PARTS (NTHREADS / TILE_N)     // threads per src1 column in the staging
#define B_ELEMS (32 / B_PARTS)
#define B_WORDS (B_ELEMS / 4)
#if A_MODE != 0 && A_SPLIT != 2
#error "K-quant DP4a staging needs two threads per row (TILE_M 128)"
#endif

RWByteAddressBuffer src1 : register(u0);
RWByteAddressBuffer src0 : register(u1);
RWByteAddressBuffer dst  : register(u2);

cbuffer Params : register(b0) {
    uint offset_src0;
    uint offset_src1;
    uint offset_dst;
    uint m;
    uint n;
    uint k;
    uint stride_01;
    uint stride_02;
    uint stride_03;
    uint stride_11;
    uint stride_12;
    uint stride_13;
    uint ne2;
    uint broadcast2;
    uint broadcast3;
    uint n_batches;
    uint nwg_x;
};

groupshared uint  Aq[8][TILE_M];
groupshared float Ad[TILE_M];
groupshared float Am[TILE_M];   // Q4_K: ml, Q6_K: second scale
groupshared uint  Bq[8][TILE_N];
groupshared float Bd[TILE_N];
groupshared float Bmx[TILE_N][B_PARTS];
groupshared float Bsm[TILE_N][B_PARTS];   // per part: sum of the quantized column values
groupshared float Bs[TILE_N];

[numthreads(NTHREADS, 1, 1)]
void main(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    const uint tid       = gtid.x;
    const uint ty        = tid % TH_Y;   // row group: rows ty + TH_Y * i
    const uint tx        = tid / TH_Y;   // column group: columns tx + TH_X * j
#define BT_ROW(i) (ty + TH_Y * (i))
#define BT_COL(j) (tx + TH_X * (j))
    const uint wg_linear = gid.y * nwg_x + gid.x;

    const uint tiles_m = (m + TILE_M - 1) / TILE_M;
    const uint tiles_n = (n + TILE_N - 1) / TILE_N;
    const uint tile_id = wg_linear % (tiles_m * tiles_n);
    const uint batch   = wg_linear / (tiles_m * tiles_n);
    if (batch >= n_batches) {
        return;
    }
    const uint dst2_idx  = batch % ne2;
    const uint dst3_idx  = batch / ne2;
    const uint src02_idx = dst2_idx / broadcast2;
    const uint src03_idx = dst3_idx / broadcast3;

    const uint row0 = (tile_id % tiles_m) * TILE_M;
    const uint col0 = (tile_id / tiles_m) * TILE_N;

    const uint a_batch = offset_src0 + src03_idx * stride_03 + src02_idx * stride_02;
    const uint b_batch = offset_src1 + dst3_idx * stride_13 + dst2_idx * stride_12;

    const uint a_row  = tid % TILE_M;
    const uint a_part = tid / TILE_M;          // elements a_part * A_ELEMS .. + A_ELEMS - 1 of the block
    const uint a_r    = row0 + a_row;
    const uint b_col  = tid / B_PARTS;
    const uint b_part = tid % B_PARTS;         // k range b_part * B_ELEMS .. + B_ELEMS - 1
    const bool b_ok   = col0 + b_col < n;
    const uint b_base = b_batch + (col0 + b_col) * stride_11;

    float acc[REG_M][REG_N];
    [unroll] for (uint im = 0; im < REG_M; im++) {
        [unroll] for (uint jn = 0; jn < REG_N; jn++) {
            acc[im][jn] = 0.0f;
        }
    }

    for (uint kt = 0; kt < k; kt += TILE_K) {
        if (a_part < A_SPLIT) {
            uint  aw[A_WORDS];
            float ad = 0.0f;
            float am = 0.0f;
            [unroll] for (uint z = 0; z < A_WORDS; z++) {
                aw[z] = 0u;
            }
            if (a_r < m) {
#if defined(SRC0_Q4_K)
                // super-block of 256: f16 d, f16 dmin, 12 bytes of 6-bit scales/mins, 128 bytes of nibbles (see the fp32 loader)
                const uint blk  = kt / 256;
                const uint s    = (kt % 256) / 32;
                const uint base = (a_batch + a_r * stride_01 + blk) * 144;
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
                ad = d * (float) sc;
                am = dmin * (float) mn;
                const uint shift = (s & 1u) * 4u;
                const uint qbase = base + 16 + 32 * (s / 2) + 16 * a_part;
                [unroll] for (uint j = 0; j < A_WORDS; j++) {
                    uint q;
                    LOAD_U32_UNALIGNED(src0, qbase + 4 * j, q);
                    aw[j] = (q >> shift) & 0x0F0F0F0Fu;
                }
#elif defined(SRC0_Q6_K)
                // super-block of 256: 128 bytes ql, 64 bytes qh, 16 int8 scales, f16 d; the two threads of a row take one scale each
                const uint blk  = kt / 256;
                const uint s    = (kt % 256) / 32;
                const uint h    = s / 4;
                const uint t    = s % 4;
                const uint base = (a_batch + a_r * stride_01 + blk) * 210;
                uint dbits;
                LOAD_U16_UNALIGNED(src0, base + 208, dbits);
                const uint ql_base = base + 64 * h + 32 * (t & 1u) + 16 * a_part;
                const uint qh_base = base + 128 + 32 * h + 16 * a_part;
                const uint sc_base = base + 192 + 8 * h + 2 * t;
                const uint lshift  = (t >> 1) * 4u;
                const uint hshift  = t * 2u;
                uint scw;
                LOAD_U32_UNALIGNED(src0, sc_base, scw);
                ad = f16tof32(dbits) * (float) sbyte_of(scw, a_part);   // this thread's half: scale a_part
                [unroll] for (uint j = 0; j < A_WORDS; j++) {
                    uint ql, qh;
                    LOAD_U32_UNALIGNED(src0, ql_base + 4 * j, ql);
                    LOAD_U32_UNALIGNED(src0, qh_base + 4 * j, qh);
                    const uint v = ((ql >> lshift) & 0x0F0F0F0Fu) | (((qh >> hshift) & 0x03030303u) << 4);
                    aw[j] = ((v | 0x80808080u) - 0x20202020u) ^ 0x80808080u;   // v - 32 per byte, no borrows
                }
#elif defined(SRC0_Q4_0)
                const uint base = (a_batch + a_r * stride_01 + kt / 32) * 18;
                uint dbits;
                LOAD_U16_UNALIGNED(src0, base, dbits);
                ad = f16tof32(dbits);
                const uint e0    = a_part * A_ELEMS;
                const bool high  = e0 >= 16;
                const uint byte0 = e0 % 16;
#if defined(BT_AL) && (A_WORDS == 4)
                const uint p_al  = base + 2 + byte0;
                const uint a0_al = p_al & ~3u;
                const uint4 w4_al = src0.Load4(a0_al);
                const uint  w5_al = src0.Load(a0_al + 16);
                const bool sh_al = (p_al & 2u) != 0u;
                uint qv_al[4];
                qv_al[0] = sh_al ? ((w4_al.x >> 16) | (w4_al.y << 16)) : w4_al.x;
                qv_al[1] = sh_al ? ((w4_al.y >> 16) | (w4_al.z << 16)) : w4_al.y;
                qv_al[2] = sh_al ? ((w4_al.z >> 16) | (w4_al.w << 16)) : w4_al.z;
                qv_al[3] = sh_al ? ((w4_al.w >> 16) | (w5_al << 16)) : w4_al.w;
                [unroll] for (uint j = 0; j < A_WORDS; j++) {
                    const uint q = qv_al[j];
                    const uint nib = (high ? (q >> 4) : q) & 0x0F0F0F0Fu;
#else
                [unroll] for (uint j = 0; j < A_WORDS; j++) {
                    uint q;
                    LOAD_U32_UNALIGNED(src0, base + 2 + byte0 + 4 * j, q);
                    const uint nib = (high ? (q >> 4) : q) & 0x0F0F0F0Fu;
#endif
                    // nibble - 8 per byte without borrows: set bit 7, subtract 8, flip bit 7
                    aw[j] = ((nib | 0x80808080u) - 0x08080808u) ^ 0x80808080u;
                }
#else
                const uint base = (a_batch + a_r * stride_01 + kt / 32) * 34;
                uint dbits;
                LOAD_U16_UNALIGNED(src0, base, dbits);
                ad = f16tof32(dbits);
#if defined(BT_AL) && (A_WORDS == 4)
                const uint p_al  = base + 2 + a_part * A_ELEMS;
                const uint a0_al = p_al & ~3u;
                const uint4 w4_al = src0.Load4(a0_al);
                const uint  w5_al = src0.Load(a0_al + 16);
                const bool sh_al = (p_al & 2u) != 0u;
                uint qv_al[4];
                qv_al[0] = sh_al ? ((w4_al.x >> 16) | (w4_al.y << 16)) : w4_al.x;
                qv_al[1] = sh_al ? ((w4_al.y >> 16) | (w4_al.z << 16)) : w4_al.y;
                qv_al[2] = sh_al ? ((w4_al.z >> 16) | (w4_al.w << 16)) : w4_al.z;
                qv_al[3] = sh_al ? ((w4_al.w >> 16) | (w5_al << 16)) : w4_al.w;
                aw[0] = qv_al[0];
                aw[1] = qv_al[1];
                aw[2] = qv_al[2];
                aw[3] = qv_al[3];
#else
                [unroll] for (uint j = 0; j < A_WORDS; j++) {
                    LOAD_U32_UNALIGNED(src0, base + 2 + a_part * A_ELEMS + 4 * j, aw[j]);
                }
#endif
#endif
            }
            [unroll] for (uint z2 = 0; z2 < A_WORDS; z2++) {
                Aq[a_part * A_WORDS + z2][a_row] = aw[z2];
            }
#if A_MODE == 2
            if (a_part == 0) {
                Ad[a_row] = ad;
            } else {
                Am[a_row] = ad;
            }
#else
            if (a_part == 0) {
                Ad[a_row] = ad;
                Am[a_row] = am;
            }
#endif
        }
        float bv[B_ELEMS];
        float amax = 0.0f;
        [unroll] for (uint i = 0; i < B_ELEMS; i++) {
            const uint ki = b_part * B_ELEMS + i;
#if defined(SRC1_F16)
            uint bits;
            LOAD_U16_UNALIGNED(src1, (b_base + kt + ki) * 2, bits);
            bv[i] = b_ok ? f16tof32(bits) : 0.0f;
#else
            bv[i] = b_ok ? LOAD_F32(src1, b_base + kt + ki) : 0.0f;
#endif
            amax = max(amax, abs(bv[i]));
        }
        Bmx[b_col][b_part] = amax;
        GroupMemoryBarrierWithGroupSync();
        float mx = 0.0f;
        [unroll] for (uint p = 0; p < B_PARTS; p++) {
            mx = max(mx, Bmx[b_col][p]);
        }
        const float inv = mx > 0.0f ? 127.0f / mx : 0.0f;
        int qsum = 0;
        [unroll] for (uint jw = 0; jw < B_WORDS; jw++) {
            uint w = 0u;
            [unroll] for (uint bb = 0; bb < 4; bb++) {
                const int qv = (int) round(bv[jw * 4 + bb] * inv);
                qsum += qv;
                w |= ((uint) (qv & 0xFF)) << (8u * bb);
            }
            Bq[b_part * B_WORDS + jw][b_col] = w;
        }
#if A_MODE == 1
        Bsm[b_col][b_part] = (float) qsum;
#endif
        if (b_part == 0) {
            Bd[b_col] = mx / 127.0f;
        }
#if A_MODE == 1
        GroupMemoryBarrierWithGroupSync();   // the part sums are read below; costs one more barrier per k step on Q4_K only
        if (b_part == 0) {
            float bs = 0.0f;
            [unroll] for (uint p2 = 0; p2 < B_PARTS; p2++) {
                bs += Bsm[b_col][p2];
            }
            Bs[b_col] = bs;
        }
#endif
        GroupMemoryBarrierWithGroupSync();

        int isum[REG_M][REG_N];
#if A_MODE == 2
        int ihi[REG_M][REG_N];
#endif
        [unroll] for (uint im1 = 0; im1 < REG_M; im1++) {
            [unroll] for (uint jn1 = 0; jn1 < REG_N; jn1++) {
                isum[im1][jn1] = 0;
#if A_MODE == 2
                ihi[im1][jn1] = 0;
#endif
            }
        }
        [unroll] for (uint wd = 0; wd < 8; wd++) {
            uint a[REG_M], b[REG_N];
            [unroll] for (uint im2 = 0; im2 < REG_M; im2++) {
                a[im2] = Aq[wd][BT_ROW(im2)];
            }
            [unroll] for (uint in2 = 0; in2 < REG_N; in2++) {
                b[in2] = Bq[wd][BT_COL(in2)];
            }
            [unroll] for (uint im3 = 0; im3 < REG_M; im3++) {
                [unroll] for (uint in3 = 0; in3 < REG_N; in3++) {
#if A_MODE == 2
                    if (wd < 4) {
                        isum[im3][in3] = dot4add_i8packed(a[im3], b[in3], isum[im3][in3]);
                    } else {
                        ihi[im3][in3] = dot4add_i8packed(a[im3], b[in3], ihi[im3][in3]);
                    }
#else
                    isum[im3][in3] = dot4add_i8packed(a[im3], b[in3], isum[im3][in3]);
#endif
                }
            }
        }
        float sa[REG_M], sm[REG_M], sb[REG_N], sbs[REG_N];
        [unroll] for (uint im4 = 0; im4 < REG_M; im4++) {
            sa[im4] = Ad[BT_ROW(im4)];
            sm[im4] = Am[BT_ROW(im4)];
        }
        [unroll] for (uint in4 = 0; in4 < REG_N; in4++) {
            sb[in4] = Bd[BT_COL(in4)];
#if A_MODE == 1
            sbs[in4] = Bs[BT_COL(in4)];
#else
            sbs[in4] = 0.0f;
#endif
        }
        [unroll] for (uint im5 = 0; im5 < REG_M; im5++) {
            [unroll] for (uint in5 = 0; in5 < REG_N; in5++) {
#if A_MODE == 1
                acc[im5][in5] += sb[in5] * (sa[im5] * (float) isum[im5][in5] - sm[im5] * sbs[in5]);
#elif A_MODE == 2
                acc[im5][in5] += sb[in5] * (sa[im5] * (float) isum[im5][in5] + sm[im5] * (float) ihi[im5][in5]);
#else
                acc[im5][in5] += (float) isum[im5][in5] * (sa[im5] * sb[in5]);
#endif
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }

    const uint dst_base = offset_dst + dst3_idx * (m * n * ne2) + dst2_idx * (m * n);
    [unroll] for (uint jn4 = 0; jn4 < REG_N; jn4++) {
        const uint c = col0 + BT_COL(jn4);
        if (c >= n) {
            continue;
        }
        [unroll] for (uint im6 = 0; im6 < REG_M; im6++) {
            const uint r = row0 + BT_ROW(im6);
            if (r < m) {
                STORE_F32(dst, dst_base + c * m + r, acc[im6][jn4]);
            }
        }
    }
}
#else   // BT_DP4A
#ifndef BT_AMODE
#define BT_AMODE 0
#endif
#ifndef BT_BVEC
#define BT_BVEC 0
#endif
#ifndef BT_HALF
#define BT_HALF 0
#endif
// exp526: BT_AMODE 2 = one thread loads the whole 18-byte Q4_0 block as 5 aligned words (needs BT_ASPLIT=1); BT_BVEC 1 = float4 loads of the f32 columns
// exp504: bigger register tile for the dense prompt path (Q4_0 / Q8_0, f32 or f16 columns, no MMID).
// 128 x 64 tile of dst per workgroup, k in steps of 32, 256 threads; each thread owns 8 rows x 4 columns
// (32 accumulators, 12 groupshared reads per 32 FMA instead of 6 per 8). All 256 threads dequantize
// (two threads per src0 row, one 16-element half of the 32-element block each), the old kernel used 64.
// Rows of a thread are interleaved (ty + 16 * i) so a wave reads 16 consecutive groupshared words and
// writes 16 consecutive dst rows.
// BT_TM x BT_TN tile, BT_RM x BT_RN values per thread, (BT_TM / BT_RM) * (BT_TN / BT_RN) = 256 threads
#define TILE_M BT_TM
#define TILE_N BT_TN
#define TILE_K 32
#define NTHREADS 256
#define REG_M BT_RM
#define REG_N BT_RN
#define TH_Y (TILE_M / REG_M)
#define TH_X (TILE_N / REG_N)
#if defined(BT_ASPLIT)
#define A_SPLIT BT_ASPLIT               // threads per src0 row in the staging (threads beyond TILE_M * A_SPLIT stay idle there)
#else
#define A_SPLIT (NTHREADS / TILE_M)     // threads per src0 row in the staging
#endif
#define A_ELEMS (32 / A_SPLIT)          // block elements per staging thread
#define B_PARTS (NTHREADS / TILE_N)     // threads per src1 column in the staging
#define B_ELEMS (32 / B_PARTS)

RWByteAddressBuffer src1 : register(u0);
RWByteAddressBuffer src0 : register(u1);
RWByteAddressBuffer dst  : register(u2);

cbuffer Params : register(b0) {
    uint offset_src0;
    uint offset_src1;
    uint offset_dst;
    uint m;
    uint n;
    uint k;
    uint stride_01;
    uint stride_02;
    uint stride_03;
    uint stride_11;
    uint stride_12;
    uint stride_13;
    uint ne2;
    uint broadcast2;
    uint broadcast3;
    uint n_batches;
    uint nwg_x;
};

#if BT_HALF && BT_BVEC && BT_LAYOUT == 1 && !defined(SRC1_F16) && (A_ELEMS == 32 || A_ELEMS == 16 || A_ELEMS == 8)
#define HALF_ON 1
groupshared uint Atile[TILE_K / 2][TILE_M];
groupshared uint Btile[TILE_K / 2][TILE_N];
#else
#define HALF_ON 0
groupshared float Atile[TILE_K][TILE_M];
groupshared float Btile[TILE_K][TILE_N];
#endif

[numthreads(NTHREADS, 1, 1)]
void main(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    const uint tid       = gtid.x;
#if BT_LAYOUT == 1
    // each thread owns consecutive rows and columns (the layout of the 64 x 32 kernel)
    const uint tx        = tid % TH_X;
    const uint ty        = tid / TH_X;
#define BT_ROW(i) (ty * REG_M + (i))
#define BT_COL(j) (tx * REG_N + (j))
#else
    const uint ty        = tid % TH_Y;   // row group: rows ty + TH_Y * i
    const uint tx        = tid / TH_Y;   // column group: columns tx + TH_X * j
#define BT_ROW(i) (ty + TH_Y * (i))
#define BT_COL(j) (tx + TH_X * (j))
#endif
    const uint wg_linear = gid.y * nwg_x + gid.x;

    const uint tiles_m = (m + TILE_M - 1) / TILE_M;
    const uint tiles_n = (n + TILE_N - 1) / TILE_N;
    const uint tile_id = wg_linear % (tiles_m * tiles_n);
    const uint batch   = wg_linear / (tiles_m * tiles_n);
    if (batch >= n_batches) {
        return;
    }
    const uint dst2_idx  = batch % ne2;
    const uint dst3_idx  = batch / ne2;
    const uint src02_idx = dst2_idx / broadcast2;
    const uint src03_idx = dst3_idx / broadcast3;

    const uint row0 = (tile_id % tiles_m) * TILE_M;
    const uint col0 = (tile_id / tiles_m) * TILE_N;

    const uint a_batch = offset_src0 + src03_idx * stride_03 + src02_idx * stride_02;
    const uint b_batch = offset_src1 + dst3_idx * stride_13 + dst2_idx * stride_12;

    // staging roles
    const uint a_row  = tid % TILE_M;
    const uint a_part = tid / TILE_M;          // elements a_part * A_ELEMS .. + A_ELEMS - 1 of the block
    const uint a_r    = row0 + a_row;
    const uint b_col  = tid / B_PARTS;         // the B_PARTS threads of one column are neighbours: contiguous src1 reads
    const uint b_part = tid % B_PARTS;         // k range b_part * B_ELEMS .. + B_ELEMS - 1
    const bool b_ok   = col0 + b_col < n;
    const uint b_base = b_batch + (col0 + b_col) * stride_11;

    float acc[REG_M][REG_N];
    [unroll] for (uint im = 0; im < REG_M; im++) {
        [unroll] for (uint jn = 0; jn < REG_N; jn++) {
            acc[im][jn] = 0.0f;
        }
    }

    for (uint kt = 0; kt < k; kt += TILE_K) {
        if (a_part < A_SPLIT) {
        float vals[A_ELEMS];
        if (a_r < m) {
#if defined(SRC0_Q4_0)
            // block: f16 d, 16 bytes of nibbles; low nibbles are elements 0..15, high nibbles 16..31
            const uint base = (a_batch + a_r * stride_01 + kt / 32) * 18;
#if BT_AMODE >= 2 && A_ELEMS == 32
            uint w[5];
            const uint a0 = base & ~3u;
            [unroll] for (uint z = 0; z < 5; z++) {
                w[z] = src0.Load(a0 + 4 * z);
            }
            if ((base & 2u) != 0u) {
                [unroll] for (uint z2 = 0; z2 < 4; z2++) {
                    w[z2] = (w[z2] >> 16) | (w[z2 + 1] << 16);
                }
                w[4] = w[4] >> 16;
            }
            const float d = f16tof32(w[0] & 0xFFFFu);
            [unroll] for (uint j = 0; j < 4; j++) {
                const uint q = (w[j] >> 16) | (w[j + 1] << 16);
                [unroll] for (uint b = 0; b < 4; b++) {
                    const uint byte = byte_of(q, b);
                    vals[j * 4 + b]      = ((float) (byte & 0xFu) - 8.0f) * d;
                    vals[16 + j * 4 + b] = ((float) (byte >> 4) - 8.0f) * d;
                }
            }
#else
            uint dbits;
            LOAD_U16_UNALIGNED(src0, base, dbits);
            const float d = f16tof32(dbits);
#if A_ELEMS == 32
            [unroll] for (uint j = 0; j < 4; j++) {
                uint q;
                LOAD_U32_UNALIGNED(src0, base + 2 + 4 * j, q);
                [unroll] for (uint b = 0; b < 4; b++) {
                    const uint byte = byte_of(q, b);
                    vals[j * 4 + b]      = ((float) (byte & 0xFu) - 8.0f) * d;
                    vals[16 + j * 4 + b] = ((float) (byte >> 4) - 8.0f) * d;
                }
            }
#else
            const uint e0     = a_part * A_ELEMS;       // first element of this thread
            const bool high   = e0 >= 16;
            const uint byte0  = e0 % 16;
            [unroll] for (uint j = 0; j < A_ELEMS / 4; j++) {
                uint q;
                LOAD_U32_UNALIGNED(src0, base + 2 + byte0 + 4 * j, q);
                [unroll] for (uint b = 0; b < 4; b++) {
                    const uint byte = byte_of(q, b);
                    vals[j * 4 + b] = ((float) (high ? (byte >> 4) : (byte & 0xFu)) - 8.0f) * d;
                }
            }
#endif
#endif
#else
            // Q8_0 block: f16 d, 32 int8
            const uint base = (a_batch + a_r * stride_01 + kt / 32) * 34;
            uint dbits;
            LOAD_U16_UNALIGNED(src0, base, dbits);
            const float d = f16tof32(dbits);
            [unroll] for (uint j = 0; j < A_ELEMS / 4; j++) {
                uint q;
                LOAD_U32_UNALIGNED(src0, base + 2 + a_part * A_ELEMS + 4 * j, q);
                [unroll] for (uint b = 0; b < 4; b++) {
                    vals[j * 4 + b] = (float) sbyte_of(q, b) * d;
                }
            }
#endif
        } else {
            [unroll] for (uint z = 0; z < A_ELEMS; z++) {
                vals[z] = 0.0f;
            }
        }
#if HALF_ON
        [unroll] for (uint kp = 0; kp < A_ELEMS / 2; kp++) {
            Atile[a_part * (A_ELEMS / 2) + kp][a_row] = f32tof16(vals[2 * kp]) | (f32tof16(vals[2 * kp + 1]) << 16);
        }
#else
        [unroll] for (uint ki = 0; ki < A_ELEMS; ki++) {
            Atile[a_part * A_ELEMS + ki][a_row] = vals[ki];
        }
#endif
        }
#if BT_BVEC && !defined(SRC1_F16)
        [unroll] for (uint i4 = 0; i4 < B_ELEMS / 4; i4++) {
            const uint ki0 = b_part * B_ELEMS + 4 * i4;
            float4 bv = float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (b_ok) {
                bv = asfloat(src1.Load4((b_base + kt + ki0) * 4));
            }
#if HALF_ON
            Btile[ki0 / 2][b_col] = f32tof16(bv.x) | (f32tof16(bv.y) << 16);
            Btile[ki0 / 2 + 1][b_col] = f32tof16(bv.z) | (f32tof16(bv.w) << 16);
#else
            Btile[ki0][b_col] = bv.x;
            Btile[ki0 + 1][b_col] = bv.y;
            Btile[ki0 + 2][b_col] = bv.z;
            Btile[ki0 + 3][b_col] = bv.w;
#endif
        }
#else
        [unroll] for (uint i = 0; i < B_ELEMS; i++) {
            const uint ki = b_part * B_ELEMS + i;
#if defined(SRC1_F16)
            uint bits;
            LOAD_U16_UNALIGNED(src1, (b_base + kt + ki) * 2, bits);
            Btile[ki][b_col] = b_ok ? f16tof32(bits) : 0.0f;
#else
            Btile[ki][b_col] = b_ok ? LOAD_F32(src1, b_base + kt + ki) : 0.0f;
#endif
        }
#endif
        GroupMemoryBarrierWithGroupSync();

#if HALF_ON
        [unroll] for (uint kp = 0; kp < TILE_K / 2; kp++) {
            uint a2[REG_M], b2[REG_N];
            [unroll] for (uint im2 = 0; im2 < REG_M; im2++) {
                a2[im2] = Atile[kp][BT_ROW(im2)];
            }
            [unroll] for (uint in2 = 0; in2 < REG_N; in2++) {
                b2[in2] = Btile[kp][BT_COL(in2)];
            }
            [unroll] for (uint hh = 0; hh < 2; hh++) {
                float a[REG_M], b[REG_N];
                [unroll] for (uint im5 = 0; im5 < REG_M; im5++) {
                    a[im5] = f16tof32(hh == 0 ? (a2[im5] & 0xFFFFu) : (a2[im5] >> 16));
                }
                [unroll] for (uint in5 = 0; in5 < REG_N; in5++) {
                    b[in5] = f16tof32(hh == 0 ? (b2[in5] & 0xFFFFu) : (b2[in5] >> 16));
                }
                [unroll] for (uint im6 = 0; im6 < REG_M; im6++) {
                    [unroll] for (uint in6 = 0; in6 < REG_N; in6++) {
                        acc[im6][in6] += a[im6] * b[in6];
                    }
                }
            }
        }
#else
        [unroll] for (uint kk = 0; kk < TILE_K; kk++) {
            float a[REG_M], b[REG_N];
            [unroll] for (uint im2 = 0; im2 < REG_M; im2++) {
                a[im2] = Atile[kk][BT_ROW(im2)];
            }
            [unroll] for (uint in2 = 0; in2 < REG_N; in2++) {
                b[in2] = Btile[kk][BT_COL(in2)];
            }
            [unroll] for (uint im3 = 0; im3 < REG_M; im3++) {
                [unroll] for (uint in3 = 0; in3 < REG_N; in3++) {
                    acc[im3][in3] += a[im3] * b[in3];
                }
            }
        }
#endif
        GroupMemoryBarrierWithGroupSync();
    }

    const uint dst_base = offset_dst + dst3_idx * (m * n * ne2) + dst2_idx * (m * n);
    [unroll] for (uint jn4 = 0; jn4 < REG_N; jn4++) {
        const uint c = col0 + BT_COL(jn4);
        if (c >= n) {
            continue;
        }
        [unroll] for (uint im4 = 0; im4 < REG_M; im4++) {
            const uint r = row0 + BT_ROW(im4);
            if (r < m) {
                STORE_F32(dst, dst_base + c * m + r, acc[im4][jn4]);
            }
        }
    }
}
#endif   // BT_DP4A
#else
#define TILE_M 64
#define TILE_N 32
#define TILE_K 32

// 256 threads as a 16 x 16 grid; each thread owns 4 rows and 2 columns of the output tile
#define TH_X 16
#define TH_Y 16
#define REG_M (TILE_M / TH_Y)
#define REG_N (TILE_N / TH_X)
#define NTHREADS (TH_X * TH_Y)

RWByteAddressBuffer src1 : register(u0);
RWByteAddressBuffer src0 : register(u1);
RWByteAddressBuffer dst  : register(u2);
#if defined(MMID)
RWByteAddressBuffer scratch : register(u3);
#endif

cbuffer Params : register(b0) {
    uint offset_src0;
    uint offset_src1;
    uint offset_dst;
    uint m;           // rows of src0 / dst
    uint n;           // columns of src1 / dst
    uint k;           // shared dimension
    uint stride_01;   // src0 row stride, elements or blocks
    uint stride_02;
    uint stride_03;
    uint stride_11;   // src1 column stride, elements
    uint stride_12;
    uint stride_13;
    uint ne2;         // dst dim 2
    uint broadcast2;  // src1 dim 2 / src0 dim 2
    uint broadcast3;
    uint n_batches;   // dst ne2 * ne3; the dispatch is rounded up to a 2D grid, so it can overshoot
#if defined(MMID)
    uint n_used;      // ids->ne[0]
    uint ne11;        // src1->ne[1]; the src1 row of slot s is s % ne11
    uint dst_s1;      // dst->nb[1] in elements
    uint dst_s2;      // dst->nb[2] in elements
    uint list_base;   // first list entry in scratch
#endif
    uint nwg_x;
};

groupshared float Atile[TILE_K][TILE_M];
groupshared float Btile[TILE_K][TILE_N];

#if !defined(SRC0_F32) && !defined(SRC0_F16) && !defined(SRC0_Q8_0) && !defined(SRC0_Q4_0) && \
    !defined(SRC0_Q4_K) && !defined(SRC0_Q6_K)
// dot_row with lane = block number and a TPR above any block count runs its loop once, for that
// block only. ACC then writes each value straight into the Atile column of this thread.
#define TILED_DOT_ROW
#define TPR 0x1000000u
#define MAX_COLS 1
static uint g_row;
#define ACC(a, kidx) { Atile[(kidx) % TILE_K][g_row] = (a); }
#include "dequant_row.hlsli"
#endif

#if defined(SRC1_F16)
float load_src1(uint i) {
    uint bits;
    LOAD_U16_UNALIGNED(src1, i * 2, bits);
    return f16tof32(bits);
}
#else
float load_src1(uint i) {
    return LOAD_F32(src1, i);
}
#endif

// Dequantize the TILE_K elements of one src0 row starting at element k0 into vals[0..TILE_K-1].
// row_base is the row's offset in elements (float types) or blocks (quant types). k0 is a multiple of
// TILE_K, so for the block types it lands exactly on a block or K-quant sub-block boundary.
void load_a_row(uint row_base, uint k0, out float vals[TILE_K]) {
#if defined(SRC0_F32)
    [unroll] for (uint i = 0; i < TILE_K; i += 4) {
        const uint4 w = src0.Load4((row_base + k0 + i) * 4);
        vals[i + 0] = asfloat(w.x);
        vals[i + 1] = asfloat(w.y);
        vals[i + 2] = asfloat(w.z);
        vals[i + 3] = asfloat(w.w);
    }
#elif defined(SRC0_F16)
    [unroll] for (uint i = 0; i < TILE_K; i += 2) {
        uint w;
        LOAD_U32_UNALIGNED(src0, (row_base + k0 + i) * 2, w);
        vals[i + 0] = f16tof32(w & 0xFFFFu);
        vals[i + 1] = f16tof32(w >> 16);
    }
#elif defined(SRC0_Q8_0)
    // block: f16 d, 32 int8
    {
        const uint base = (row_base + k0 / 32) * 34;
        uint dbits;
        LOAD_U16_UNALIGNED(src0, base, dbits);
        const float d = f16tof32(dbits);
        [unroll] for (uint j = 0; j < 8; j++) {
            uint q;
            LOAD_U32_UNALIGNED(src0, base + 2 + 4 * j, q);
            [unroll] for (uint b = 0; b < 4; b++) {
                vals[j * 4 + b] = (float) sbyte_of(q, b) * d;
            }
        }
    }
#elif defined(SRC0_Q4_0)
    // block: f16 d, 16 bytes of nibbles; low nibbles are elements 0..15, high nibbles 16..31
    {
        const uint base = (row_base + k0 / 32) * 18;
        uint dbits;
        LOAD_U16_UNALIGNED(src0, base, dbits);
        const float d = f16tof32(dbits);
        [unroll] for (uint j = 0; j < 4; j++) {
            uint q;
            LOAD_U32_UNALIGNED(src0, base + 2 + 4 * j, q);
            [unroll] for (uint b = 0; b < 4; b++) {
                const uint byte = byte_of(q, b);
                vals[j * 4 + b]      = ((float) (byte & 0xFu) - 8.0f) * d;
                vals[16 + j * 4 + b] = ((float) (byte >> 4) - 8.0f) * d;
            }
        }
    }
#elif defined(SRC0_Q4_K)
    // super-block of 256: f16 d, f16 dmin, 12 bytes of 6-bit scales/mins, 128 bytes of nibbles.
    // sub-block s (32 values): pair s/2 uses bytes 16 + 32*(s/2), low nibbles for even s, high for odd s.
    {
        const uint blk  = k0 / 256;
        const uint s    = (k0 % 256) / 32;
        const uint base = (row_base + blk) * 144;
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
        const float dl    = d * (float) sc;
        const float ml    = dmin * (float) mn;
        const uint  shift = (s & 1u) * 4u;
        const uint  qbase = base + 16 + 32 * (s / 2);
        [unroll] for (uint j = 0; j < 8; j++) {
            uint q;
            LOAD_U32_UNALIGNED(src0, qbase + 4 * j, q);
            [unroll] for (uint b = 0; b < 4; b++) {
                vals[j * 4 + b] = dl * (float) ((byte_of(q, b) >> shift) & 0xFu) - ml;
            }
        }
    }
#elif defined(SRC0_Q6_K)
    // super-block of 256: 128 bytes ql, 64 bytes qh, 16 int8 scales, f16 d.
    // sub-block s: half h = s/4 (128 values), t = s%4 selects the quarter inside the half.
    // The 16 scales are one per 16 values, so a sub-block of 32 uses two of them.
    {
        const uint blk  = k0 / 256;
        const uint s    = (k0 % 256) / 32;
        const uint h    = s / 4;
        const uint t    = s % 4;
        const uint base = (row_base + blk) * 210;
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
                vals[j * 4 + b] = dsc * (float) q;
            }
        }
    }
#endif
}

[numthreads(NTHREADS, 1, 1)]
void main(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    const uint tid       = gtid.x;
    const uint tx        = tid % TH_X;          // column group
    const uint ty        = tid / TH_X;          // row group
    const uint wg_linear = gid.y * nwg_x + gid.x;

    const uint tiles_m = (m + TILE_M - 1) / TILE_M;
    // staging: threads 0..TILE_M-1 take one src0 row each, so a row is dequantized exactly once;
    // all NTHREADS threads share the src1 tile, 8 threads per column, 4 values each
    const uint b_col  = tid / 8;
    const uint b_part = tid % 8;

#if defined(MMID)
    // workgroups past the tile count do no work. They do not return early: on the MTT S80 an early
    // return before the barriers gave wrong results. Their k loop runs zero times instead.
    const uint row0    = (wg_linear % tiles_m) * TILE_M;
    const uint ct      = wg_linear / tiles_m;
    const bool active  = ct < scratch.Load(0);
    uint3      tile    = uint3(0, 0, 0);
    if (active) {
        tile = scratch.Load3((1 + 3 * ct) * 4);
    }
#if defined(GGML_D3D11)
    // FXC refuses barriers in a loop whose count comes from a buffer load (X3663); inactive workgroups
    // run the full loop on tile 0 and store nothing (n_cols is 0)
    const uint k_end   = k;
#else
    const uint k_end   = active ? k : 0;
#endif
    const uint a_batch = offset_src0 + tile.x * stride_02;
    const uint n_cols  = tile.z;
    const bool b_ok    = b_col < n_cols;
    uint       b_base  = 0;
    if (b_ok) {
        const uint pair = scratch.Load((list_base + tile.y + b_col) * 4);
        b_base = offset_src1 + ((pair % n_used) % ne11) * stride_11 + (pair / n_used) * stride_12;
    }
#else
    const uint tiles_n = (n + TILE_N - 1) / TILE_N;
    const uint tile_id = wg_linear % (tiles_m * tiles_n);
    const uint batch   = wg_linear / (tiles_m * tiles_n);

    if (batch >= n_batches) {
        return;
    }
    const uint batches2  = ne2;
    const uint dst2_idx  = batch % batches2;
    const uint dst3_idx  = batch / batches2;
    const uint src02_idx = dst2_idx / broadcast2;
    const uint src03_idx = dst3_idx / broadcast3;

    const uint row0 = (tile_id % tiles_m) * TILE_M;
    const uint col0 = (tile_id / tiles_m) * TILE_N;

    const uint a_batch = offset_src0 + src03_idx * stride_03 + src02_idx * stride_02;
    const uint b_batch = offset_src1 + dst3_idx * stride_13 + dst2_idx * stride_12;
    const uint k_end   = k;
    const bool b_ok    = col0 + b_col < n;
    const uint b_base  = b_batch + (col0 + b_col) * stride_11;
#endif

    float acc[REG_M][REG_N];
    [unroll] for (uint im = 0; im < REG_M; im++) {
        [unroll] for (uint jn = 0; jn < REG_N; jn++) {
            acc[im][jn] = 0.0f;
        }
    }

    for (uint kt = 0; kt < k_end; kt += TILE_K) {
        if (tid < TILE_M) {
            const uint r = row0 + tid;
#if defined(TILED_DOT_ROW)
            if (r < m) {
                uint  src1_base[MAX_COLS];
                float unused[MAX_COLS];
                src1_base[0] = 0;
                unused[0]    = 0.0f;
                g_row        = tid;
                dot_row(src0, a_batch + r * stride_01, kt / 32, 0, src1_base, unused);
            } else {
                [unroll] for (uint z = 0; z < TILE_K; z++) {
                    Atile[z][tid] = 0.0f;
                }
            }
#else
            float vals[TILE_K];
            if (r < m) {
                load_a_row(a_batch + r * stride_01, kt, vals);
            } else {
                [unroll] for (uint z = 0; z < TILE_K; z++) {
                    vals[z] = 0.0f;
                }
            }
            [unroll] for (uint ki = 0; ki < TILE_K; ki++) {
                Atile[ki][tid] = vals[ki];
            }
#endif
        }
        [unroll] for (uint i = 0; i < 4; i++) {
            const uint ki = b_part * 4 + i;
            Btile[ki][b_col] = b_ok ? load_src1(b_base + kt + ki) : 0.0f;
        }
        GroupMemoryBarrierWithGroupSync();

        [unroll] for (uint kk = 0; kk < TILE_K; kk++) {
            float a[REG_M], b[REG_N];
            [unroll] for (uint im2 = 0; im2 < REG_M; im2++) {
                a[im2] = Atile[kk][ty * REG_M + im2];
            }
            [unroll] for (uint in2 = 0; in2 < REG_N; in2++) {
                b[in2] = Btile[kk][tx * REG_N + in2];
            }
            [unroll] for (uint im3 = 0; im3 < REG_M; im3++) {
                [unroll] for (uint in3 = 0; in3 < REG_N; in3++) {
                    acc[im3][in3] += a[im3] * b[in3];
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }

#if !defined(MMID)
    const uint dst_base = offset_dst + dst3_idx * (m * n * batches2) + dst2_idx * (m * n);
#endif
    [unroll] for (uint im4 = 0; im4 < REG_M; im4++) {
        const uint r = row0 + ty * REG_M + im4;
        if (r >= m) {
            continue;
        }
        [unroll] for (uint in4 = 0; in4 < REG_N; in4++) {
#if defined(MMID)
            const uint j = tx * REG_N + in4;
            if (j < n_cols) {
                const uint pair = scratch.Load((list_base + tile.y + j) * 4);
                STORE_F32(dst, offset_dst + (pair % n_used) * dst_s1 + (pair / n_used) * dst_s2 + r, acc[im4][in4]);
            }
#else
            const uint c = col0 + tx * REG_N + in4;
            if (c < n) {
                STORE_F32(dst, dst_base + c * m + r, acc[im4][in4]);
            }
#endif
        }
    }
}
#endif   // BIG_TILE
