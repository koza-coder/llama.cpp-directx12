#include "common.hlsli"

// dst[idx[row]] = src[row]; src is f32, or f16 with SRC_F16; defines: DST_{F32,F16,BF16}, I64_IDX (low 32 bits of
// the index are used). F16 and BF16 (src or dst) need USE_16BIT.
// f32 -> bf16: round to nearest even, a NaN stays a NaN

#if defined(DST_F32)
#define STORE_DST(i, v) STORE_F32(dst, i, v)
#elif defined(DST_F16)
#define STORE_DST(i, v) STORE_F16(dst, i, v)
#elif defined(DST_BF16)
uint f32_to_bf16_bits(float f) {
    const uint x = asuint(f);
    if ((x & 0x7FFFFFFFu) > 0x7F800000u) {
        return (x >> 16) | 0x40u;
    }
    return (x + 0x7FFFu + ((x >> 16) & 1u)) >> 16;
}
#define STORE_DST(i, v) dst.Store<uint16_t>((i) * 2, (uint16_t) f32_to_bf16_bits(v))
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

    uint stride_dst1;
    uint stride_dst2;
    uint stride_dst3;

    uint ne0;
    uint n_rows;
    uint ne2;
    uint ne3;

    uint idx1;
    uint idx2;

    uint nwg_x;
};

[numthreads(WG_SIZE, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    const uint i = flat_index(gid, nwg_x);
    if (i >= ne3 * ne2 * n_rows * ne0) {
        return;
    }
    uint row = i / ne0;
    const uint col = i % ne0;

    const uint i_src3 = row / (ne2 * n_rows);
    row = row % (ne2 * n_rows);
    const uint i_src2 = row / n_rows;
    const uint i_src1 = row % n_rows;

    const uint i_idx2 = i_src3 % idx2;
    const uint i_idx1 = i_src2 % idx1;
    const uint i_idx0 = i_src1;

    const uint idx_elem = offset_idx + i_idx0 * stride_idx0 + i_idx1 * stride_idx1 + i_idx2 * stride_idx2;
#ifdef I64_IDX
    const uint idx_val = idx.Load(idx_elem * 8);
#else
    const uint idx_val = idx.Load(idx_elem * 4);
#endif

    const uint i_dst_row = offset_dst + idx_val * stride_dst1 + i_src2 * stride_dst2 + i_src3 * stride_dst3;
    const uint i_src_row = offset_src + i_src1 * stride_src1 + i_src2 * stride_src2 + i_src3 * stride_src3;

    STORE_DST(i_dst_row + col, LOAD_SRC(i_src_row + col));
}
