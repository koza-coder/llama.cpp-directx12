#include "common.hlsli"

// ARGSORT of long rows (f32 -> i32): bitonic sorting network over the index array held in dst, one thread per
// compare-exchange, one dispatch per step (the C++ side loops over stages k and distances h).
// Positions at or beyond ne0 stand for +infinity (they never move), so any ne0 works; every comparison puts the
// smaller element at the lower position, the first step of a stage compares position p with its mirror in the 2h block.
// Ties keep the smaller source index first. defines: SORT_DESC
// mode 0: dst[i] = i (one thread per element); mode 1: one compare-exchange step (one thread per pair)

RWByteAddressBuffer src : register(u0);
RWByteAddressBuffer dst : register(u1);

cbuffer Params : register(b0) {
    uint offset_src;
    uint offset_dst;
    uint stride_src1;
    uint ne0;
    uint half_n;   // padded row length / 2
    uint h;
    uint flip;
    uint mode;
    uint total;    // threads
    uint nwg_x;
};

[numthreads(WG_SIZE, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    const uint t = flat_index(gid, nwg_x);
    if (t >= total) {
        return;
    }
    if (mode == 0u) {
        const uint r = t / ne0;
        const uint i = t % ne0;
        STORE_I32(dst, offset_dst + r * ne0 + i, (int) i);
        return;
    }
    const uint r = t / half_n;
    const uint p = t % half_n;
    const uint i = (p / h) * 2u * h + (p % h);
    const uint j = flip != 0u ? (p / h) * 2u * h + (2u * h - 1u) - (p % h) : i + h;
    if (j >= ne0) {
        return;
    }
    const uint drow = offset_dst + r * ne0;
    const uint srow = offset_src + r * stride_src1;
    const uint a = (uint) LOAD_I32(dst, drow + i);
    const uint b = (uint) LOAD_I32(dst, drow + j);
    const float va = LOAD_F32(src, srow + a);
    const float vb = LOAD_F32(src, srow + b);
#if defined(SORT_DESC)
    const bool b_first = vb > va || (vb == va && b < a);
#else
    const bool b_first = vb < va || (vb == va && b < a);
#endif
    if (b_first) {
        STORE_I32(dst, drow + i, (int) b);
        STORE_I32(dst, drow + j, (int) a);
    }
}
