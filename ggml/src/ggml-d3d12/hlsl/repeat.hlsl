#include "common.hlsli"

// REPEAT as raw bits: dst[i0, i1, i2, i3] = src[i0 % ne00, i1 % ne01, i2 % ne02, i3 % ne03], any strides
// 4-byte elements (f32, i32); ELEM16 2-byte (f16, bf16, i16), ELEM8 1-byte (i8): strides and offsets are then in units
// of the element size, a half or byte is written with two atomics so the neighbouring elements are kept

RWByteAddressBuffer src : register(u0);
RWByteAddressBuffer dst : register(u1);

cbuffer Params : register(b0) {
    uint offset_src;
    uint offset_dst;
    uint stride_src0;
    uint stride_src1;
    uint stride_src2;
    uint stride_src3;
    uint stride_dst0;
    uint stride_dst1;
    uint stride_dst2;
    uint stride_dst3;
    uint src_ne0;
    uint src_ne1;
    uint src_ne2;
    uint src_ne3;
    uint ne0;
    uint ne1;
    uint ne2;
    uint ne;
    uint nwg_x;
};

[numthreads(WG_SIZE, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint i = flat_index(gid, nwg_x);
    if (i >= ne) {
        return;
    }
    const uint i3 = i / (ne2 * ne1 * ne0);
    i = i % (ne2 * ne1 * ne0);
    const uint i2 = i / (ne1 * ne0);
    i = i % (ne1 * ne0);
    const uint i1 = i / ne0;
    const uint i0 = i % ne0;
    const uint s = offset_src + (i0 % src_ne0) * stride_src0 + (i1 % src_ne1) * stride_src1 +
                   (i2 % src_ne2) * stride_src2 + (i3 % src_ne3) * stride_src3;
    const uint d = offset_dst + i0 * stride_dst0 + i1 * stride_dst1 + i2 * stride_dst2 + i3 * stride_dst3;
#if defined(ELEM16)
    uint bits;
    LOAD_U16_UNALIGNED(src, s * 2, bits);
    const uint sh = ((d * 2) & 2u) * 8u;
    dst.InterlockedAnd((d * 2) & ~3u, ~(0xFFFFu << sh));
    dst.InterlockedOr((d * 2) & ~3u, bits << sh);
#elif defined(ELEM8)
    const uint bits = (src.Load(s & ~3u) >> ((s & 3u) * 8u)) & 0xFFu;
    const uint sh = (d & 3u) * 8u;
    dst.InterlockedAnd(d & ~3u, ~(0xFFu << sh));
    dst.InterlockedOr(d & ~3u, bits << sh);
#else
    dst.Store(d * 4, src.Load(s * 4));
#endif
}
