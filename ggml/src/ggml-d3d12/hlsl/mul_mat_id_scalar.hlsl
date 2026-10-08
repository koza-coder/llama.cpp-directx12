#include "common.hlsli"

// MUL_MAT_ID for float weights (SRC0_F32 / SRC0_F16 / SRC0_BF16) with any k, including k % 4 != 0 which the
// vectorised mul_mat_vec / tiled kernels cannot read. dst[row, slot, token] = as[ids[slot, token]][row, :] . src1[:, slot % b_ne1, token]. One thread per dst element, scalar loads; slow but
// complete (parity with Vulkan's odd-k cases). SRC1_F16: src1 holds f16 instead of f32.
// Strides and offsets are in elements of their own type.

RWByteAddressBuffer src0 : register(u0);
RWByteAddressBuffer src1 : register(u1);
RWByteAddressBuffer dst  : register(u2);
RWByteAddressBuffer ids  : register(u3);

cbuffer Params : register(b0) {
    uint offset_src0;
    uint offset_src1;
    uint offset_dst;
    uint offset_ids;
    uint k;
    uint m;
    uint n_used;
    uint stride_01;
    uint stride_02;
    uint stride_11;
    uint stride_12;
    uint b_ne1;    // src1->ne[1]
    uint ids_s1;   // ids->nb[1] in elements
    uint dst_s1;
    uint dst_s2;
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
    const uint row  = i % m;
    i /= m;
    const uint slot = i % n_used;
    const uint tok  = i / n_used;
    const uint expert = (uint) ids.Load((offset_ids + tok * ids_s1 + slot) * 4);

    const uint a_base = offset_src0 + expert * stride_02 + row * stride_01;
    const uint b_base = offset_src1 + tok * stride_12 + (slot % b_ne1) * stride_11;
    float sum = 0.0f;
    for (uint x = 0; x < k; x++) {
        sum += load_a(a_base + x) * load_b(b_base + x);
    }
    STORE_F32(dst, offset_dst + tok * dst_s2 + slot * dst_s1 + row, sum);
}
