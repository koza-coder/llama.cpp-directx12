#include "common.hlsli"

// TOP_K with a small k (at most 32) over long rows (exp970, exp971: no HLSL keywords as names; draft samplers run top_k 10 over a 248k vocab), in two passes instead of the
// ~170-step bitonic network. Pass 1 (default): each workgroup takes one chunk of WG_SIZE * PER_THREAD elements of one row and writes its k best
// (value, index) pairs to a scratch buffer. Pass 2 (define PASS2): one workgroup per row picks the k best of the n_chunks * k candidates and writes
// their indices to dst. A round picks the best element left (larger value first, the smaller index on a tie) with a workgroup reduction; the
// owning thread marks the pick in a bit mask, so equal values and infinities stay distinct. A chunk with fewer than k elements pads with
// "none" pairs (index 0xFFFFFFFF), which never win against an element.

#define PER_THREAD 16
#define PER_THREAD2 32
#define NONE 0xFFFFFFFFu

RWByteAddressBuffer src : register(u0);   // pass 1: the f32 rows; pass 2: the scratch pairs
RWByteAddressBuffer dst : register(u1);   // pass 1: the scratch pairs; pass 2: the i32 rows

cbuffer Params : register(b0) {
    uint offset_src;   // elements
    uint offset_dst;   // elements
    uint ne0;          // pass 1: row length; pass 2: candidates per row (n_chunks * k)
    uint k;
    uint n_chunks;
    uint n_rows;
    uint ne1;
    uint ne2;
    uint nb1;          // row strides in elements: pass 1 of src, pass 2 of dst
    uint nb2;
    uint nb3;
    uint nwg_x;
};

groupshared float sh_val[WG_SIZE];
groupshared uint  sh_idx[WG_SIZE];
groupshared uint  sh_pos[WG_SIZE];

bool better(float va, uint ia, float vb, uint ib) {
    return ia != NONE && (ib == NONE || va > vb || (va == vb && ia < ib));
}

// workgroup reduction of (val, idx, pos); every thread gets the winner
void reduce_best(uint t, inout float v, inout uint i, inout uint p) {
    sh_val[t] = v;
    sh_idx[t] = i;
    sh_pos[t] = p;
    GroupMemoryBarrierWithGroupSync();
    for (uint off = WG_SIZE / 2; off > 0; off /= 2) {
        if (t < off && better(sh_val[t + off], sh_idx[t + off], sh_val[t], sh_idx[t])) {
            sh_val[t] = sh_val[t + off];
            sh_idx[t] = sh_idx[t + off];
            sh_pos[t] = sh_pos[t + off];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    v = sh_val[0];
    i = sh_idx[0];
    p = sh_pos[0];
    GroupMemoryBarrierWithGroupSync();   // sh_* are written again by the next round
}

uint row_offset(uint row) {
    const uint i1 = row % ne1;
    const uint i2 = (row / ne1) % ne2;
    const uint i3 = row / (ne1 * ne2);
    return i1 * nb1 + i2 * nb2 + i3 * nb3;
}

[numthreads(WG_SIZE, 1, 1)]
void main(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    const uint t = gtid.x;
#if !defined(PASS2)
    const uint wg    = gid.y * nwg_x + gid.x;
    const uint row   = wg / n_chunks;
    const uint chunk = wg % n_chunks;
    if (row >= n_rows) {
        return;
    }
    const uint srow = offset_src + row_offset(row);
    const uint base = chunk * (WG_SIZE * PER_THREAD);
    float v[PER_THREAD];
    uint  picked = 0;
    [unroll] for (uint j = 0; j < PER_THREAD; j++) {
        const uint c = base + j * WG_SIZE + t;
        v[j] = c < ne0 ? LOAD_F32(src, srow + c) : 0.0f;
        if (c >= ne0) {
            picked |= 1u << j;
        }
    }
    const uint outb = offset_dst + wg * k * 2;
    for (uint r = 0; r < k; r++) {
        float bv = 0.0f;
        uint  bi = NONE;
        uint  bp = 0;
        [unroll] for (uint j = 0; j < PER_THREAD; j++) {
            const uint c = base + j * WG_SIZE + t;
            if ((picked & (1u << j)) == 0 && better(v[j], c, bv, bi)) {
                bv = v[j];
                bi = c;
                bp = j;
            }
        }
        reduce_best(t, bv, bi, bp);
        if (t == 0) {
            STORE_F32(dst, outb + r * 2, bv);
            dst.Store((outb + r * 2 + 1) * 4, bi);
        }
        if (bi != NONE && (bi - base) % WG_SIZE == t) {
            picked |= 1u << bp;
        }
    }
#else
    const uint row = gid.y * nwg_x + gid.x;
    if (row >= n_rows) {
        return;
    }
    const uint inb = offset_src + row * ne0 * 2;
    float v[PER_THREAD2];
    uint  id[PER_THREAD2];
    [unroll] for (uint j = 0; j < PER_THREAD2; j++) {
        const uint c = j * WG_SIZE + t;
        v[j]  = c < ne0 ? LOAD_F32(src, inb + c * 2) : 0.0f;
        id[j] = c < ne0 ? src.Load((inb + c * 2 + 1) * 4) : NONE;
    }
    uint picked = 0;
    const uint drow = offset_dst + row_offset(row);
    for (uint r = 0; r < k; r++) {
        float bv = 0.0f;
        uint  bi = NONE;
        uint  bp = 0;
        [unroll] for (uint j = 0; j < PER_THREAD2; j++) {
            if ((picked & (1u << j)) == 0 && better(v[j], id[j], bv, bi)) {
                bv = v[j];
                bi = id[j];
                bp = j * WG_SIZE + t;
            }
        }
        reduce_best(t, bv, bi, bp);
        if (t == 0) {
            STORE_I32(dst, drow + r, (int) bi);
        }
        if (bi != NONE && bp % WG_SIZE == t) {
            picked |= 1u << (bp / WG_SIZE);
        }
    }
#endif
}
