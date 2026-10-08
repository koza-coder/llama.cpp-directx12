#include "common.hlsli"

// LEAKY_RELU (defines TYPE_F32 or TYPE_F16 (+USE_16BIT), math in f32): dst = x > 0 ? x : x * negative_slope, one workgroup per row

#if defined(TYPE_F16)
#define LOAD(buf, i)     LOAD_F16(buf, i)
#define STORE(buf, i, v) STORE_F16(buf, i, v)
#else
#define LOAD(buf, i)     LOAD_F32(buf, i)
#define STORE(buf, i, v) STORE_F32(buf, i, v)
#endif

RWByteAddressBuffer src : register(u0);
RWByteAddressBuffer dst : register(u1);

cbuffer Params : register(b0) {
    uint  offset_src;
    uint  offset_dst;
    uint  stride_src1;
    uint  stride_dst1;
    uint  ne0;
    uint  n_rows;
    float negative_slope;
    uint  nwg_x;
};

[numthreads(WG_SIZE, 1, 1)]
void main(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    const uint r = gid.y * nwg_x + gid.x;
    if (r >= n_rows) {
        return;
    }
    const uint src_row = offset_src + r * stride_src1;
    const uint dst_row = offset_dst + r * stride_dst1;
    for (uint c = gtid.x; c < ne0; c += WG_SIZE) {
        const float x = LOAD(src, src_row + c);
        STORE(dst, dst_row + c, x > 0.0f ? x : x * negative_slope);
    }
}
