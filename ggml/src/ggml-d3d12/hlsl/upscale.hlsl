#include "common.hlsli"

// UPSCALE (f32), nearest, bilinear or bicubic (alpha -0.75, with or without align corners), following the CPU reference; one thread
// per dst element, any strides. defines: BILINEAR, ANTIALIAS (bilinear + antialias) or BICUBIC

RWByteAddressBuffer src : register(u0);
RWByteAddressBuffer dst : register(u1);

cbuffer Params : register(b0) {
    uint  offset_src;
    uint  offset_dst;
    uint  stride_src0;
    uint  stride_src1;
    uint  stride_src2;
    uint  stride_src3;
    uint  stride_dst0;
    uint  stride_dst1;
    uint  stride_dst2;
    uint  stride_dst3;
    uint  src_ne0;
    uint  src_ne1;
    uint  ne0;
    uint  ne1;
    uint  ne2;
    uint  ne;
    float sf0;
    float sf1;
    float sf2;
    float sf3;
    float pixel_offset;
    uint  nwg_x;
};

float cubic_w1(float x) { return ((-0.75f + 2.0f) * x - (-0.75f + 3.0f)) * x * x + 1.0f; }
float cubic_w2(float x) { return ((-0.75f * x + 3.75f) * x - 6.0f) * x + 3.0f; }
float cubic(float p0, float p1, float p2, float p3, float x) {
    return p0 * cubic_w2(x + 1.0f) + p1 * cubic_w1(x) + p2 * cubic_w1(1.0f - x) + p3 * cubic_w2(2.0f - x);
}

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

    const uint i02  = (uint) ((float) i2 / sf2);
    const uint i03  = (uint) ((float) i3 / sf3);
    const uint base = offset_src + i02 * stride_src2 + i03 * stride_src3;
    const uint o    = offset_dst + i0 * stride_dst0 + i1 * stride_dst1 + i2 * stride_dst2 + i3 * stride_dst3;

#if defined(BICUBIC)
    const float y  = ((float) i1 + pixel_offset) / sf1 - pixel_offset;
    const int   y0 = (int) floor(y);
    const float dy = y - (float) y0;
    const float x  = ((float) i0 + pixel_offset) / sf0 - pixel_offset;
    const int   x0 = (int) floor(x);
    const float dx = x - (float) x0;
    float rows[4];
    [unroll] for (int r = 0; r < 4; r++) {
        const uint yy = (uint) clamp(y0 + r - 1, 0, (int) src_ne1 - 1);
        float p[4];
        [unroll] for (int c = 0; c < 4; c++) {
            const uint xx = (uint) clamp(x0 + c - 1, 0, (int) src_ne0 - 1);
            p[c] = LOAD_F32(src, base + xx * stride_src0 + yy * stride_src1);
        }
        rows[r] = cubic(p[0], p[1], p[2], p[3], dx);
    }
    STORE_F32(dst, o, cubic(rows[0], rows[1], rows[2], rows[3], dy));
#elif defined(ANTIALIAS)
    // bilinear with antialiasing (triangle filter, PyTorch-style), same order of operations as the CPU reference
    const float support1  = max(1.0f, 1.0f / sf1);
    const float invscale1 = 1.0f / support1;
    const float support0  = max(1.0f, 1.0f / sf0);
    const float invscale0 = 1.0f / support0;
    const float y = ((float) i1 + pixel_offset) / sf1;
    const float x = ((float) i0 + pixel_offset) / sf0;
    const int x_min = max((int) (x - support0 + pixel_offset), 0);
    const int x_max = min((int) (x + support0 + pixel_offset), (int) src_ne0);
    const int y_min = max((int) (y - support1 + pixel_offset), 0);
    const int y_max = min((int) (y + support1 + pixel_offset), (int) src_ne1);
    float val = 0.0f;
    float total_weight = 0.0f;
    for (int sy = y_min; sy < y_max; sy++) {
        const float weight_y = max(1.0f - abs(((float) sy - y + pixel_offset) * invscale1), 0.0f);
        for (int sx = x_min; sx < x_max; sx++) {
            const float weight_x = max(1.0f - abs(((float) sx - x + pixel_offset) * invscale0), 0.0f);
            const float weight = weight_x * weight_y;
            if (weight <= 0.0f) {
                continue;
            }
            val += LOAD_F32(src, base + (uint) sx * stride_src0 + (uint) sy * stride_src1) * weight;
            total_weight += weight;
        }
    }
    if (total_weight > 0.0f) {
        val /= total_weight;
    }
    STORE_F32(dst, o, val);
#elif defined(BILINEAR)
    const float y  = ((float) i1 + pixel_offset) / sf1 - pixel_offset;
    const int   fy = (int) floor(y);
    const int   y0 = clamp(fy, 0, (int) src_ne1 - 1);
    const int   y1 = clamp(fy + 1, 0, (int) src_ne1 - 1);
    const float dy = clamp(y - (float) y0, 0.0f, 1.0f);
    const float x  = ((float) i0 + pixel_offset) / sf0 - pixel_offset;
    const int   fx = (int) floor(x);
    const int   x0 = clamp(fx, 0, (int) src_ne0 - 1);
    const int   x1 = clamp(fx + 1, 0, (int) src_ne0 - 1);
    const float dx = clamp(x - (float) x0, 0.0f, 1.0f);

    const float a = LOAD_F32(src, base + (uint) x0 * stride_src0 + (uint) y0 * stride_src1);
    const float b = LOAD_F32(src, base + (uint) x1 * stride_src0 + (uint) y0 * stride_src1);
    const float c = LOAD_F32(src, base + (uint) x0 * stride_src0 + (uint) y1 * stride_src1);
    const float d = LOAD_F32(src, base + (uint) x1 * stride_src0 + (uint) y1 * stride_src1);
    STORE_F32(dst, o, a * (1 - dx) * (1 - dy) + b * dx * (1 - dy) + c * (1 - dx) * dy + d * dx * dy);
#else
    const uint i00 = (uint) ((float) i0 / sf0);
    const uint i01 = (uint) ((float) i1 / sf1);
    STORE_F32(dst, o, LOAD_F32(src, base + i00 * stride_src0 + i01 * stride_src1));
#endif
}
