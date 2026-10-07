#include "common.hlsli"

// CONCAT of two tensors along dim, copied as raw bits; any strides.
// 4-byte elements (f32, i32); ELEM16 2-byte (f16, bf16, i16), ELEM8 1-byte (i8), ELEM64 8-byte (i64): strides and
// offsets are then in units of the element size; a half or byte is read from its word and written with two
// atomics so the neighbouring bytes are kept; 8-byte elements move as two words.
// ELEMBLK=<bytes>: quantised blocks (even size), the element is one block and ne0 etc. count blocks; copied as halves

RWByteAddressBuffer src0 : register(u0);
RWByteAddressBuffer src1 : register(u1);
RWByteAddressBuffer dst  : register(u2);

cbuffer Params : register(b0) {
    uint offset_src0;
    uint offset_src1;
    uint offset_dst;

    uint stride_src00;
    uint stride_src01;
    uint stride_src02;
    uint stride_src03;
    uint stride_src10;
    uint stride_src11;
    uint stride_src12;
    uint stride_src13;
    uint stride_dst0;
    uint stride_dst1;
    uint stride_dst2;
    uint stride_dst3;

    uint ne;
    uint ne0;
    uint ne1;
    uint ne2;

    uint dim;
    uint src0_nedim;   // src0->ne[dim]

    uint nwg_x;
};

[numthreads(WG_SIZE, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint i = flat_index(gid, nwg_x);
    if (i >= ne) {
        return;
    }
    uint idx[4];
    idx[3] = i / (ne2 * ne1 * ne0);
    i = i % (ne2 * ne1 * ne0);
    idx[2] = i / (ne1 * ne0);
    i = i % (ne1 * ne0);
    idx[1] = i / ne0;
    idx[0] = i % ne0;

    const uint i_dst = offset_dst + idx[0] * stride_dst0 + idx[1] * stride_dst1 + idx[2] * stride_dst2 + idx[3] * stride_dst3;
    uint bits;
#ifdef ELEMBLK
    uint src_base;
#endif
#ifdef ELEM64
    uint2 bits2;
#endif
    const bool idx_is_src0 = idx[dim] < src0_nedim;
    if (idx_is_src0) {
#if defined(ELEM16)
        LOAD_U16_UNALIGNED(src0, (offset_src0 + idx[0] * stride_src00 + idx[1] * stride_src01 + idx[2] * stride_src02 + idx[3] * stride_src03) * 2, bits);
#elif defined(ELEM8)
        { const uint _a = (offset_src0 + idx[0] * stride_src00 + idx[1] * stride_src01 + idx[2] * stride_src02 + idx[3] * stride_src03); bits = (src0.Load(_a & ~3u) >> ((_a & 3u) * 8u)) & 0xFFu; }
#elif defined(ELEMBLK)
        src_base = (offset_src0 + idx[0] * stride_src00 + idx[1] * stride_src01 + idx[2] * stride_src02 + idx[3] * stride_src03) * ELEMBLK;
#elif defined(ELEM64)
        bits2 = src0.Load2((offset_src0 + idx[0] * stride_src00 + idx[1] * stride_src01 + idx[2] * stride_src02 + idx[3] * stride_src03) * 8);
#else
        bits = src0.Load((offset_src0 + idx[0] * stride_src00 + idx[1] * stride_src01 + idx[2] * stride_src02 + idx[3] * stride_src03) * 4);
#endif
    } else {
        idx[dim] -= src0_nedim;
#if defined(ELEM16)
        LOAD_U16_UNALIGNED(src1, (offset_src1 + idx[0] * stride_src10 + idx[1] * stride_src11 + idx[2] * stride_src12 + idx[3] * stride_src13) * 2, bits);
#elif defined(ELEM8)
        { const uint _a = (offset_src1 + idx[0] * stride_src10 + idx[1] * stride_src11 + idx[2] * stride_src12 + idx[3] * stride_src13); bits = (src1.Load(_a & ~3u) >> ((_a & 3u) * 8u)) & 0xFFu; }
#elif defined(ELEMBLK)
        src_base = (offset_src1 + idx[0] * stride_src10 + idx[1] * stride_src11 + idx[2] * stride_src12 + idx[3] * stride_src13) * ELEMBLK;
#elif defined(ELEM64)
        bits2 = src1.Load2((offset_src1 + idx[0] * stride_src10 + idx[1] * stride_src11 + idx[2] * stride_src12 + idx[3] * stride_src13) * 8);
#else
        bits = src1.Load((offset_src1 + idx[0] * stride_src10 + idx[1] * stride_src11 + idx[2] * stride_src12 + idx[3] * stride_src13) * 4);
#endif
    }
#if defined(ELEM16)
    {
        const uint sh = ((i_dst * 2) & 2u) * 8u;
        dst.InterlockedAnd((i_dst * 2) & ~3u, ~(0xFFFFu << sh));
        dst.InterlockedOr((i_dst * 2) & ~3u, bits << sh);
    }
#elif defined(ELEM8)
    {
        const uint sh = (i_dst & 3u) * 8u;
        dst.InterlockedAnd(i_dst & ~3u, ~(0xFFu << sh));
        dst.InterlockedOr(i_dst & ~3u, bits << sh);
    }
#elif defined(ELEM64)
    dst.Store2(i_dst * 8, bits2);
#elif defined(ELEMBLK)
    {
        const uint src_buf_is0 = (idx_is_src0) ? 1u : 0u;
        for (uint h = 0; h < ELEMBLK / 2; h++) {
            uint v;
            const uint a = src_base + h * 2;
            if (src_buf_is0 != 0u) { LOAD_U16_UNALIGNED(src0, a, v); } else { LOAD_U16_UNALIGNED(src1, a, v); }
            const uint da = i_dst * ELEMBLK + h * 2;
            const uint sh = (da & 2u) * 8u;
            dst.InterlockedAnd(da & ~3u, ~(0xFFFFu << sh));
            dst.InterlockedOr(da & ~3u, v << sh);
        }
    }
#else
    dst.Store(i_dst * 4, bits);
#endif
}
