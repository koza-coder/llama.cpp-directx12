#include "common.hlsli"

// COL2IM_1D (f32, or f16 with -DCOL_F16 / bf16 with -DCOL_BF16 for both columns and signal): scatter-add columns [K*OC, T_in] back to a signal [T_out, OC].
// The CPU already gathers rather than scatters, so this is a direct transcription: one thread owns
// one output sample and sums the at most ceil(K/s) columns that overlap it.

RWByteAddressBuffer src : register(u0);   // [K*OC, T_in]
RWByteAddressBuffer dst : register(u1);   // [T_out, OC]

#if defined(COL_BF16)
// bf16 columns and signal: the top 16 bits of an f32 (rounded to nearest even, NaN kept quiet); raw word access so no
// 16-bit loads are needed, and a half is written with two atomics so the other half is kept
uint f32_to_bf16(float v) {
    const uint u = asuint(v);
    if ((u & 0x7FFFFFFFu) > 0x7F800000u) {
        return (u >> 16) | 0x40u;
    }
    return (u + 0x7FFFu + ((u >> 16) & 1u)) >> 16;
}
#define LOAD_COL(b, i)     asfloat(((b).Load(((i) * 2) & ~3u) >> ((((i) * 2) & 2u) * 8u)) << 16)
#define STORE_COL(b, i, v) { \
    const uint _h  = f32_to_bf16(v); \
    const uint _sh = (((i) * 2) & 2u) * 8u; \
    (b).InterlockedAnd(((i) * 2) & ~3u, ~(0xFFFFu << _sh)); \
    (b).InterlockedOr(((i) * 2) & ~3u, _h << _sh); \
}
#elif defined(COL_F16)
#define LOAD_COL(b, i)     LOAD_F16(b, i)
#define STORE_COL(b, i, v) STORE_F16(b, i, v)
#else
#define LOAD_COL(b, i)     LOAD_F32(b, i)
#define STORE_COL(b, i, v) STORE_F32(b, i, v)
#endif

cbuffer Params : register(b0) {
    uint offset_src;
    uint offset_dst;
    uint k_oc;     // K * OC, the column height
    uint t_in;
    uint kk;       // K, taps per output channel
    uint t_out;
    int  s0;
    int  p0;
    uint ne;
    uint nwg_x;
};

[numthreads(WG_SIZE, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    const uint i = flat_index(gid, nwg_x);
    if (i >= ne) {
        return;
    }
    const uint oc = i / t_out;
    const uint t  = i % t_out;

    const int t_abs = (int) t + p0;   // position in the uncropped signal
    // the first column whose window still reaches t_abs, and the last one that starts at or before it
    int lo = (t_abs - (int) kk + 1 + s0 - 1) / s0;
    if (lo < 0) {
        lo = 0;
    }
    int hi = t_abs / s0;
    if (hi >= (int) t_in) {
        hi = (int) t_in - 1;
    }

    float sum = 0.0f;
    for (int c = lo; c <= hi; c++) {
        const int k = t_abs - c * s0;
        if (k >= 0 && k < (int) kk) {
            sum += LOAD_COL(src, offset_src + (oc * kk + (uint) k) + (uint) c * k_oc);
        }
    }
    STORE_COL(dst, offset_dst + i, sum);
}
