#include "common.hlsli"

// Groups the (token, slot) pairs of a mul_mat_id by expert, for the tiled kernel (mul_mat_tiled.hlsl, MMID).
// One workgroup. Output in scratch, in uints:
//   [0]                     number of tiles
//   [1 + 3 t .. 3 + 3 t]    tile t: expert, first list entry, column count (1..32)
//   [list_base + i]         pair index token * n_used + slot, grouped by expert
// Like the list that mm_ids_helper makes in the CUDA backend. The order inside an expert is not fixed;
// every column is computed on its own, so the result does not depend on it.

#define MAX_EXPERTS 1024
#define COLS_PER_TILE 32

RWByteAddressBuffer ids     : register(u0);
RWByteAddressBuffer scratch : register(u1);

cbuffer Params : register(b0) {
    uint offset_ids;
    uint ids_s1;      // ids->nb[1] in elements
    uint n_used;      // ids->ne[0]
    uint n_tokens;    // ids->ne[1]
    uint n_experts;   // at most MAX_EXPERTS
    uint list_base;
    uint nwg_x;
};

groupshared uint counts[MAX_EXPERTS];
groupshared uint starts[MAX_EXPERTS];

[numthreads(WG_SIZE, 1, 1)]
void main(uint3 gtid : SV_GroupThreadID) {
    const uint tid     = gtid.x;
    const uint n_pairs = n_used * n_tokens;

    for (uint e = tid; e < n_experts; e += WG_SIZE) {
        counts[e] = 0;
    }
    GroupMemoryBarrierWithGroupSync();

    for (uint p = tid; p < n_pairs; p += WG_SIZE) {
        const uint e = (uint) ids.Load((offset_ids + (p / n_used) * ids_s1 + p % n_used) * 4);
        if (e < n_experts) {
            InterlockedAdd(counts[e], 1u);
        }
    }
    GroupMemoryBarrierWithGroupSync();

    // one thread: list start of each expert, and the tile table
    if (tid == 0) {
        uint pos   = 0;
        uint tiles = 0;
        for (uint e = 0; e < n_experts; e++) {
            const uint c = counts[e];
            starts[e]    = pos;
            for (uint c0 = 0; c0 < c; c0 += COLS_PER_TILE) {
                scratch.Store3((1 + 3 * tiles) * 4, uint3(e, pos + c0, min(c - c0, (uint) COLS_PER_TILE)));
                tiles++;
            }
            pos += c;
        }
        scratch.Store(0, tiles);
    }
    GroupMemoryBarrierWithGroupSync();

    for (uint p2 = tid; p2 < n_pairs; p2 += WG_SIZE) {
        const uint e = (uint) ids.Load((offset_ids + (p2 / n_used) * ids_s1 + p2 % n_used) * 4);
        if (e < n_experts) {
            uint slot;
            InterlockedAdd(starts[e], 1u, slot);
            scratch.Store((list_base + slot) * 4, p2);
        }
    }
}
