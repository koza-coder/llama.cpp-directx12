#include "common.hlsli"

// CPY / DUP / CONT of one type to itself, copied as raw bits; any strides, src and dst may differ in shape.
// TS = bytes per element (a quantised block counts as one element, ne0 and the strides are in blocks).
// CH4: TS % 4 == 0, words; CH2: TS % 2 == 0, halves; CH1: bytes. Halves and bytes are written with two atomics so the
// neighbouring elements are kept.

RWByteAddressBuffer src : register(u0);
RWByteAddressBuffer dst : register(u1);

cbuffer Params : register(b0) {
    uint ne;
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

    uint dst_ne0;
    uint dst_ne1;
    uint dst_ne2;

    uint nwg_x;
};

[numthreads(WG_SIZE, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    const uint idx = flat_index(gid, nwg_x);
    if (idx >= ne) {
        return;
    }

    uint i = idx;
    const uint i3 = i / (src_ne2 * src_ne1 * src_ne0);
    i = i % (src_ne2 * src_ne1 * src_ne0);
    const uint i2 = i / (src_ne1 * src_ne0);
    i = i % (src_ne1 * src_ne0);
    const uint i1 = i / src_ne0;
    const uint i0 = i % src_ne0;

    uint j = idx;
    const uint j3 = j / (dst_ne2 * dst_ne1 * dst_ne0);
    j = j % (dst_ne2 * dst_ne1 * dst_ne0);
    const uint j2 = j / (dst_ne1 * dst_ne0);
    j = j % (dst_ne1 * dst_ne0);
    const uint j1 = j / dst_ne0;
    const uint j0 = j % dst_ne0;

    const uint sb = (offset_src + i0 * stride_src0 + i1 * stride_src1 + i2 * stride_src2 + i3 * stride_src3) * TS;
    const uint db = (offset_dst + j0 * stride_dst0 + j1 * stride_dst1 + j2 * stride_dst2 + j3 * stride_dst3) * TS;
#if defined(CH4)
    for (uint o = 0; o < TS; o += 4) {
        dst.Store(db + o, src.Load(sb + o));
    }
#elif defined(CH2)
    for (uint o = 0; o < TS; o += 2) {
        uint v;
        LOAD_U16_UNALIGNED(src, sb + o, v);
        const uint a  = db + o;
        const uint sh = (a & 2u) * 8u;
        dst.InterlockedAnd(a & ~3u, ~(0xFFFFu << sh));
        dst.InterlockedOr(a & ~3u, v << sh);
    }
#else
    for (uint o = 0; o < TS; o++) {
        const uint a  = sb + o;
        const uint v  = (src.Load(a & ~3u) >> ((a & 3u) * 8u)) & 0xFFu;
        const uint d  = db + o;
        const uint sh = (d & 3u) * 8u;
        dst.InterlockedAnd(d & ~3u, ~(0xFFu << sh));
        dst.InterlockedOr(d & ~3u, v << sh);
    }
#endif
}
