#include "common.hlsli"

// MoE router in one pass (softmax -> top-k -> get_rows [-> sum, clamp, div]), one workgroup per token:
//   ids[t][0..k)     = indices of the k largest softmax probs, descending (lowest index first on ties)
//   weights[t][0..k) = their probs, divided by clamp(sum, clamp_min, clamp_max) with NORM
// defines: NORM

RWByteAddressBuffer logits  : register(u0);
RWByteAddressBuffer ids     : register(u1);
RWByteAddressBuffer weights : register(u2);

cbuffer Params : register(b0) {
    uint  offset_logits;
    uint  offset_ids;
    uint  offset_weights;
    uint  n_expert;
    uint  n_used;
    uint  stride_logits;    // elements between token rows
    uint  stride_ids;
    uint  stride_weights;
    uint  n_tokens;
    float clamp_min;
    float clamp_max;
    uint  nwg_x;
};

#define MAX_EXPERT 1024

groupshared float p_sh[MAX_EXPERT];
groupshared float red_v[WG_SIZE];
groupshared uint  red_i[WG_SIZE];

[numthreads(WG_SIZE, 1, 1)]
void main(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    const uint t   = gid.x + nwg_x * gid.y;
    const uint tid = gtid.x;
    if (t >= n_tokens) {
        return;
    }
    const uint base = offset_logits + t * stride_logits;

    // softmax: max, then exp and sum
    float m = -3.4028235e38f;
    for (uint e = tid; e < n_expert; e += WG_SIZE) {
        const float v = LOAD_F32(logits, base + e);
        p_sh[e] = v;
        m = max(m, v);
    }
    red_v[tid] = m;
    GroupMemoryBarrierWithGroupSync();
    for (uint h = WG_SIZE / 2; h > 0; h /= 2) {
        if (tid < h) {
            red_v[tid] = max(red_v[tid], red_v[tid + h]);
        }
        GroupMemoryBarrierWithGroupSync();
    }
    m = red_v[0];
    GroupMemoryBarrierWithGroupSync();
    float s = 0.0f;
    for (uint e1 = tid; e1 < n_expert; e1 += WG_SIZE) {
        const float v = exp(p_sh[e1] - m);
        p_sh[e1] = v;
        s += v;
    }
    red_v[tid] = s;
    GroupMemoryBarrierWithGroupSync();
    for (uint h1 = WG_SIZE / 2; h1 > 0; h1 /= 2) {
        if (tid < h1) {
            red_v[tid] += red_v[tid + h1];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    const float inv = 1.0f / red_v[0];
    GroupMemoryBarrierWithGroupSync();
    for (uint e2 = tid; e2 < n_expert; e2 += WG_SIZE) {
        p_sh[e2] *= inv;
    }
    GroupMemoryBarrierWithGroupSync();

    // top-k: k rounds of argmax; the winner is marked taken with -1 (probs are >= 0)
    float wsum = 0.0f;
    for (uint k = 0; k < n_used; k++) {
        float bv = -2.0f;
        uint  bi = 0xFFFFFFFFu;
        for (uint e3 = tid; e3 < n_expert; e3 += WG_SIZE) {
            const float v = p_sh[e3];
            if (v > bv) {   // ascending e per thread: the first (lowest) index wins a tie
                bv = v;
                bi = e3;
            }
        }
        red_v[tid] = bv;
        red_i[tid] = bi;
        GroupMemoryBarrierWithGroupSync();
        for (uint h2 = WG_SIZE / 2; h2 > 0; h2 /= 2) {
            if (tid < h2) {
                const float ov = red_v[tid + h2];
                const uint  oi = red_i[tid + h2];
                if (ov > red_v[tid] || (ov == red_v[tid] && oi < red_i[tid])) {
                    red_v[tid] = ov;
                    red_i[tid] = oi;
                }
            }
            GroupMemoryBarrierWithGroupSync();
        }
        const float wv = red_v[0];
        const uint  wi = red_i[0];
        if (tid == 0) {
            STORE_I32(ids, offset_ids + t * stride_ids + k, (int) wi);
            STORE_F32(weights, offset_weights + t * stride_weights + k, wv);
            p_sh[wi] = -1.0f;
        }
        wsum += wv;
        GroupMemoryBarrierWithGroupSync();
    }

#if defined(NORM)
    if (tid == 0) {
        const float d = clamp(wsum, clamp_min, clamp_max);
        for (uint k1 = 0; k1 < n_used; k1++) {
            const uint wo = offset_weights + t * stride_weights + k1;
            STORE_F32(weights, wo, LOAD_F32(weights, wo) / d);
        }
    }
#endif
}
