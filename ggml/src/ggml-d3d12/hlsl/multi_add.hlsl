#include "common.hlsli"

// a chain of f32 ADDs whose operands are all views of one tensor with the same strides (the MoE expert sum):
// dst = sum of n_src views, view k starting at element offs[k]

RWByteAddressBuffer src : register(u0);
RWByteAddressBuffer dst : register(u1);

cbuffer Params : register(b0) {
    uint ne;
    uint offset_dst;
    uint n_src;
    uint ne0;

    uint ne1;
    uint ne2;
    uint stride1;
    uint stride2;

    uint stride3;
    uint pad0;
    uint pad1;
    uint pad2;

    uint4 offs[4];   // element offsets of the views, 4 per vector

    uint nwg_x;
};

[numthreads(WG_SIZE, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    const uint idx = flat_index(gid, nwg_x);
    if (idx >= ne) {
        return;
    }

    uint i = idx;
    const uint i3 = i / (ne2 * ne1 * ne0);
    i = i % (ne2 * ne1 * ne0);
    const uint i2 = i / (ne1 * ne0);
    i = i % (ne1 * ne0);
    const uint i1 = i / ne0;
    const uint i0 = i % ne0;
    const uint base = i0 + i1 * stride1 + i2 * stride2 + i3 * stride3;

    // same order as the ADD chain: ((v0 + v1) + v2) + ...
    float sum = LOAD_F32(src, offs[0][0] + base);
    for (uint k = 1; k < n_src; k++) {
        sum += LOAD_F32(src, offs[k / 4][k % 4] + base);
    }
    STORE_F32(dst, offset_dst + idx, sum);
}
