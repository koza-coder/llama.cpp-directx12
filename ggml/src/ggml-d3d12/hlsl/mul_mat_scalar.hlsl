#include "common.hlsli"

// MUL_MAT for float weights (SRC0_F32 / SRC0_F16 / SRC0_BF16) with any k, including k % 4 != 0 which the
// vectorised mul_mat_vec / tiled kernels cannot read. One thread per dst element, scalar loads; slow but
// complete (parity with Vulkan's odd-k cases). SRC1_F16: src1 holds f16 instead of f32.
// Strides and offsets are in elements of their own type.

RWByteAddressBuffer src0 : register(u0);
RWByteAddressBuffer src1 : register(u1);
RWByteAddressBuffer dst  : register(u2);

cbuffer Params : register(b0) {
    uint offset_src0;
    uint offset_src1;
    uint offset_dst;
    uint k;
    uint m;
    uint n;
    uint ne2;
    uint stride_01;
    uint stride_02;
    uint stride_03;
    uint stride_11;
    uint stride_12;
    uint stride_13;
    uint br2;      // dst->ne[2] / src0->ne[2]
    uint br3;      // dst->ne[3] / src0->ne[3]
    uint ne;       // nelements(dst)
    uint nwg_x;
};

float load_a(uint i) {
#if defined(SRC0_F32)
    return LOAD_F32(src0, i);
#elif defined(SRC0_F16)
    uint bits;
    LOAD_U16_UNALIGNED(src0, i * 2, bits);
    return f16tof32(bits);
#else
    uint bits;
    LOAD_U16_UNALIGNED(src0, i * 2, bits);
    return asfloat(bits << 16);
#endif
}

float load_b(uint i) {
#if defined(SRC1_F16)
    uint bits;
    LOAD_U16_UNALIGNED(src1, i * 2, bits);
    return f16tof32(bits);
#else
    return LOAD_F32(src1, i);
#endif
}

[numthreads(WG_SIZE, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint i = flat_index(gid, nwg_x);
    if (i >= ne) {
        return;
    }
    const uint row = i % m;
    i /= m;
    const uint col = i % n;
    i /= n;
    const uint i2 = i % ne2;
    const uint i3 = i / ne2;

    const uint a_base = offset_src0 + (i3 / br3) * stride_03 + (i2 / br2) * stride_02 + row * stride_01;
    const uint b_base = offset_src1 + i3 * stride_13 + i2 * stride_12 + col * stride_11;
    float sum = 0.0f;
    for (uint x = 0; x < k; x++) {
        sum += load_a(a_base + x) * load_b(b_base + x);
    }
    STORE_F32(dst, offset_dst + ((i3 * ne2 + i2) * n + col) * m + row, sum);
}
