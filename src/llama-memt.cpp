#include "llama-memt.h"

#include "llama-impl.h"
#include "llama-model.h"

#include <map>

bool llama_memt::init(const llama_model & model, uint32_t n_seq_max) {
    const auto & hparams = model.hparams;

    GGML_ASSERT(layers.empty());

    for (uint32_t il = 0; il < hparams.n_layer(); ++il) {
        if (model.layers[il].memt.wq) {
            layers.push_back(il);
        }
    }

    if (layers.empty()) {
        return true;
    }

    this->n_seq_max = n_seq_max;
    seq_set.assign(n_seq_max, false);

    const int64_t S = hparams.n_embd_head_k();
    const int64_t H = hparams.n_head();

    std::map<ggml_backend_buffer_type_t, ggml_context *> ctx_map;
    auto ctx_for_buft = [&](ggml_backend_buffer_type_t buft) -> ggml_context * {
        auto it = ctx_map.find(buft);
        if (it != ctx_map.end()) {
            return it->second;
        }
        ggml_init_params params = {
            /*.mem_size   =*/ 2*layers.size()*ggml_tensor_overhead(),
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
        };
        ggml_context * ctx = ggml_init(params);
        if (!ctx) {
            return nullptr;
        }
        ctx_map[buft] = ctx;
        ctxs.emplace_back(ctx);
        return ctx;
    };

    reads.assign(hparams.n_layer(), nullptr);
    writes.assign(hparams.n_layer(), nullptr);

    for (int32_t il : layers) {
        ggml_context * ctx = ctx_for_buft(model.select_buft(il));
        if (!ctx) {
            LLAMA_LOG_ERROR("%s: failed to create context for memt state\n", __func__);
            return false;
        }
        reads[il]  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, S*S*H, n_seq_max);
        writes[il] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, S*S*H, n_seq_max);
        ggml_format_name(reads[il],  "memt_read_l%d",  il);
        ggml_format_name(writes[il], "memt_write_l%d", il);
    }

    size_t total = 0;
    for (auto [buft, ctx] : ctx_map) {
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        if (!buf) {
            LLAMA_LOG_ERROR("%s: failed to allocate buffer for memt state\n", __func__);
            return false;
        }
        ggml_backend_buffer_clear(buf, 0);
        total += ggml_backend_buffer_get_size(buf);
        bufs.emplace_back(buf);
    }

    LLAMA_LOG_INFO("%s: %zu memory layers, %u seqs, %.2f MiB\n", __func__, layers.size(), n_seq_max, total/1024.0/1024.0);

    return true;
}

ggml_tensor * llama_memt::read_for(int il) const {
    return il < (int) reads.size() ? reads[il] : nullptr;
}

ggml_tensor * llama_memt::write_for(int il) const {
    return il < (int) writes.size() ? writes[il] : nullptr;
}

size_t llama_memt::n_state() const {
    size_t res = 0;
    for (int32_t il : layers) {
        res += reads[il]->ne[0];
    }
    return res;
}

void llama_memt::set(llama_seq_id seq_id, const float * data) {
    GGML_ASSERT(seq_id >= 0 && (uint32_t) seq_id < n_seq_max);

    std::vector<float> zeros;
    size_t off = 0;
    for (int32_t il : layers) {
        const size_t n      = reads[il]->ne[0];
        const size_t offset = reads[il]->nb[1]*seq_id;
        const float * src = data ? data + off : nullptr;
        if (!src) {
            zeros.resize(n, 0.0f);
            src = zeros.data();
        }
        ggml_backend_tensor_set(reads[il],  src, offset, n*sizeof(float));
        ggml_backend_tensor_set(writes[il], src, offset, n*sizeof(float));
        off += n;
    }

    seq_set[seq_id] = data != nullptr;
}

void llama_memt::get(llama_seq_id seq_id, float * data) const {
    GGML_ASSERT(seq_id >= 0 && (uint32_t) seq_id < n_seq_max);

    size_t off = 0;
    for (int32_t il : layers) {
        const size_t n = writes[il]->ne[0];
        ggml_backend_tensor_get(writes[il], data + off, writes[il]->nb[1]*seq_id, n*sizeof(float));
        off += n;
    }
}

bool llama_memt::active() const {
    for (bool b : seq_set) {
        if (b) {
            return true;
        }
    }
    return false;
}
