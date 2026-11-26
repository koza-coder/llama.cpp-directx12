#pragma once

#include "llama.h"

#include "ggml-cpp.h"

#include <vector>

struct llama_model;

//
// llama_memt - per-sequence state of the MemT native memory
//
// every memory layer keeps two F32 tensors [S*S*H, n_seq_max]:
//   read  - snapshot that the memory attention reads
//   write - state that a commit pass updates
// one row holds the state of one sequence, layout [S_k, S_v, H] (S_k fastest)
//

struct llama_memt {
    bool init(const llama_model & model, uint32_t n_seq_max);

    bool enabled() const { return !layers.empty(); }

    ggml_tensor * read_for (int il) const;
    ggml_tensor * write_for(int il) const;

    // number of floats in the state of one sequence (all memory layers)
    size_t n_state() const;

    // data == nullptr clears the state of the sequence
    void set(llama_seq_id seq_id, const float * data);
    void get(llama_seq_id seq_id, float * data) const;

    // true when at least one sequence has a non-zero state
    bool active() const;

    std::vector<int32_t> layers; // indices of the memory layers

private:
    uint32_t n_seq_max = 0;

    std::vector<bool> seq_set;

    std::vector<ggml_tensor *> reads;  // per layer
    std::vector<ggml_tensor *> writes; // per layer

    std::vector<ggml_context_ptr>        ctxs;
    std::vector<ggml_backend_buffer_ptr> bufs;
};
