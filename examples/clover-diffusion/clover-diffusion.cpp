// llama-clover-diffusion: block-diffusion text generation for Clover (clover_moe) GGUFs with block-diffusion tensors.
//
// Decoding as in the Diffuspace Clover model card ("Block diffusion with LowConfidence decoding"): each block starts
// as masked tokens; every step runs one bidirectional pass over the block and fills all masked positions whose
// confidence is >= threshold (at least the single most confident one); a resolved block becomes the causal prefix of
// the next block. The block batch is [last committed token, masks...] (the anchor carries the prefix state, as in
// draft-self), so a block of width W yields W-1 new tokens.
//
// Options: the usual model/context options, -p, -n, --temp 0 only (greedy), --diffusion-block-length W (default 32),
// --diffusion-steps S (max steps per block, then the rest is filled at once), --diffusion-visual (live view),
// --threshold T (default 0.95), --presence-penalty P and --repeat-penalty R on the generated response tokens
// (model card defaults: 0.9 and 1.05; the tool default is off = 0.0 / 1.0).

#include "arg.h"
#include "chat.h"
#include "common.h"
#include "llama.h"
#include "log.h"

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#endif

static std::string token_text(const llama_vocab * vocab, llama_token t) {
    char buf[256];
    const int n = llama_token_to_piece(vocab, t, buf, sizeof(buf), 0, false);
    return n > 0 ? std::string(buf, n) : std::string();
}

static std::string format_prompt(const std::string & prompt, const std::string & system_prompt, bool use_template, llama_model * model) {
    if (!use_template || prompt.find("<|im_start|>") != std::string::npos) {
        return prompt;
    }
    auto templates = common_chat_templates_init(model, "");
    common_chat_templates_inputs inputs;
    if (!system_prompt.empty()) {
        common_chat_msg msg;
        msg.role    = "system";
        msg.content = system_prompt;
        inputs.messages.push_back(msg);
    }
    common_chat_msg msg;
    msg.role    = "user";
    msg.content = prompt;
    inputs.messages.push_back(msg);
    inputs.add_generation_prompt = true;
    return common_chat_templates_apply(templates.get(), inputs).prompt.str();
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    ggml_time_init();

    float threshold = 0.95f;
    std::vector<char *> args;
    for (int i = 0; i < argc; ++i) {
        if (!strcmp(argv[i], "--threshold") && i + 1 < argc) {
            threshold = (float) atof(argv[++i]);
            continue;
        }
        args.push_back(argv[i]);
    }

    common_params params;
    common_init();
    if (!common_params_parse((int) args.size(), args.data(), params, LLAMA_EXAMPLE_DIFFUSION)) {
        return 1;
    }

#ifdef _WIN32
    {
        HANDLE h = GetStdHandle(STD_OUTPUT_HANDLE);
        DWORD  mode = 0;
        if (GetConsoleMode(h, &mode)) {
            SetConsoleMode(h, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
        }
        SetConsoleOutputCP(CP_UTF8);
    }
#endif

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers  = params.n_gpu_layers;
    mparams.devices       = params.devices.data();
    mparams.load_mode     = params.load_mode;
    mparams.check_tensors = params.check_tensors;

    llama_model * model = llama_model_load_from_file(params.model.path.c_str(), mparams);
    if (!model) {
        LOG_ERR("error: failed to load model '%s'\n", params.model.path.c_str());
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const llama_token   mask  = llama_vocab_mask(vocab);
    char arch[64] = "";
    llama_model_meta_val_str(model, "general.architecture", arch, sizeof(arch));
    if (strcmp(arch, "clover_moe") != 0 || mask == LLAMA_TOKEN_NULL) {
        LOG_ERR("error: llama-clover-diffusion needs a clover_moe GGUF with a mask token (got arch '%s')\n", arch);
        llama_model_free(model);
        return 1;
    }

    const int32_t width = params.diffusion.block_length > 1 ? params.diffusion.block_length : 32;

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx           = params.n_ctx;
    cparams.n_batch         = std::max<uint32_t>(params.n_batch, width);
    cparams.n_ubatch        = std::max<uint32_t>(params.n_ubatch, width);
    cparams.flash_attn_type = params.flash_attn_type;
    cparams.no_perf         = params.no_perf;
    cparams.type_k          = params.cache_type_k;
    cparams.type_v          = params.cache_type_v;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        LOG_ERR("error: failed to create context\n");
        llama_model_free(model);
        return 1;
    }
    llama_set_n_threads(ctx, params.cpuparams.n_threads, params.cpuparams_batch.n_threads);

    const std::string text   = format_prompt(params.prompt, params.system_prompt, params.enable_chat_template, model);
    std::vector<llama_token> prompt = common_tokenize(vocab, text, true, true);
    if (prompt.empty()) {
        LOG_ERR("error: empty prompt\n");
        return 1;
    }
    const int32_t n_ctx     = (int32_t) llama_n_ctx(ctx);
    const int32_t n_predict = params.n_predict < 0 ? 512 : params.n_predict;
    const int32_t max_steps = params.diffusion.steps > 0 ? params.diffusion.steps : width;
    const int32_t n_vocab   = llama_vocab_n_tokens(vocab);
    const int32_t n_batch   = (int32_t) llama_n_batch(ctx);
    const bool    visual    = params.diffusion.visual_mode;
    const float   pen_pres  = params.sampling.penalty_present;
    const float   pen_rep   = params.sampling.penalty_repeat;
    std::vector<char> seen(n_vocab, 0); // tokens of the generated response so far (penalty history, prompt excluded)

    LOG_INF("clover-diffusion: prompt %zu tokens, block width %d (%d new tokens per block), threshold %.3f, max steps/block %d, n_predict %d, presence penalty %.2f, repeat penalty %.2f\n",
            prompt.size(), width, width - 1, threshold, max_steps, n_predict, pen_pres, pen_rep);

    common_batch batch(ctx);

    // causal prefill of all prompt tokens but the last one (the first anchor)
    const int64_t t_start = ggml_time_us();
    const int32_t n_pre   = (int32_t) prompt.size() - 1;
    for (int32_t off = 0; off < n_pre; off += n_batch) {
        batch.clear();
        for (int32_t j = off; j < std::min(n_pre, off + n_batch); ++j) {
            batch.add(prompt[j], j, 0, false);
        }
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
            LOG_ERR("error: prompt prefill failed\n");
            return 1;
        }
    }
    const int64_t t_prefill = ggml_time_us();

    if (!visual) {
        printf("%s", text.c_str());
        fflush(stdout);
    }

    llama_token anchor = prompt.back();
    llama_pos   pos    = n_pre;

    std::vector<llama_token> out;
    std::vector<llama_token> blk(width);
    std::vector<llama_token> cand(width);
    std::vector<float>       conf(width);
    std::vector<const float *> lgp(width, nullptr);
    const int32_t n_threads = std::max(1, (int32_t) params.cpuparams.n_threads);
    int64_t t_dec = 0, t_rst = 0, t_smp = 0; // per-step time: block pass + logits read, state restore, sampling
    common_prompt_checkpoint ckpt;
    std::string              committed;
    int32_t n_blocks = 0, n_steps = 0;
    std::string steps_list;
    bool    done = false;

    while (!done && (int32_t) out.size() < n_predict) {
        if (pos + width >= n_ctx) {
            LOG_WRN("\nclover-diffusion: context full\n");
            break;
        }
        blk[0] = anchor;
        std::fill(blk.begin() + 1, blk.end(), mask);

        // the memory state before every step of this block is the same: save it once per block, restore after each step
        ckpt.update_tgt(ctx, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        if (n_blocks == 0) {
            LOG_INF("clover-diffusion: state checkpoint %.1f MiB per block\n", ckpt.size() / 1048576.0);
        }

        int32_t step = 0;
        while (true) {
            int32_t n_masked = 0;
            for (int32_t i = 1; i < width; ++i) {
                n_masked += blk[i] == mask;
            }
            if (n_masked == 0) {
                break;
            }

            // bidirectional pass over the block; the memory is restored afterwards
            batch.clear();
            // logits only where a token is still masked (less GPU->host readback; the anchor and filled positions
            // still take part in the bidirectional attention)
            for (int32_t i = 0; i < width; ++i) {
                batch.add(blk[i], pos + i, 0, blk[i] == mask);
            }
            llama_set_causal_attn(ctx, false);
            const int64_t ts0 = ggml_time_us();
            const int ret = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get());
            llama_set_causal_attn(ctx, true);
            // read the logits of the block (this waits for the GPU) before the restore, so the timings split cleanly
            for (int32_t i = 1; i < width && ret == 0; ++i) {
                if (blk[i] == mask) {
                    lgp[i] = llama_get_logits_ith(ctx, i);
                }
            }
            const int64_t ts1 = ggml_time_us();
            if (!ckpt.load_tgt(ctx, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) || !llama_memory_seq_rm(llama_get_memory(ctx), 0, pos, -1)) {
                LOG_ERR("error: failed to restore the state after a block pass\n");
                return 1;
            }
            if (ret != 0) {
                LOG_ERR("error: block pass failed (%d)\n", ret);
                return 1;
            }
            const int64_t ts2 = ggml_time_us();
            ++step;

            // LowConfidence: fill every masked position with confidence >= threshold, at least the most confident one
            // per masked position: argmax and confidence over the whole vocabulary (penalized logits); the positions
            // are independent, so they are split over n_threads (same math as one thread, bit-identical result)
            auto score = [&](int32_t i) {
                const float * lg = lgp[i];
                // penalized logit: repetition (divide positive / multiply negative) and presence, for seen tokens only
                auto pl = [&](int32_t v) -> float {
                    float x = lg[v];
                    if (seen[v]) {
                        x = x > 0.0f ? x / pen_rep : x * pen_rep;
                        x -= pen_pres;
                    }
                    return x;
                };
                int32_t am = mask == 0 ? 1 : 0;
                float   best_l = pl(am);
                for (int32_t v = 0; v < n_vocab; ++v) {
                    if (v != mask) {
                        const float x = pl(v);
                        if (x > best_l) {
                            best_l = x;
                            am     = v;
                        }
                    }
                }
                double sum = 0.0;
                for (int32_t v = 0; v < n_vocab; ++v) {
                    sum += std::exp((double) pl(v) - best_l);
                }
                cand[i] = am;
                conf[i] = (float) (1.0 / sum);
            };
            std::vector<int32_t> todo;
            for (int32_t i = 1; i < width; ++i) {
                if (blk[i] == mask) {
                    todo.push_back(i);
                }
            }
            const int32_t nt = std::max(1, std::min(n_threads, (int32_t) todo.size()));
            std::vector<std::thread> pool;
            for (int32_t t = 1; t < nt; ++t) {
                pool.emplace_back([&, t]() {
                    for (size_t k = t; k < todo.size(); k += nt) {
                        score(todo[k]);
                    }
                });
            }
            for (size_t k = 0; k < todo.size(); k += nt) {
                score(todo[k]);
            }
            for (auto & th : pool) {
                th.join();
            }
            int32_t best = -1;
            for (int32_t i : todo) {
                if (best < 0 || conf[i] > conf[best]) {
                    best = i;
                }
            }
            const int64_t ts3 = ggml_time_us();
            t_dec += ts1 - ts0; t_rst += ts2 - ts1; t_smp += ts3 - ts2;
            int32_t n_filled = 0;
            for (int32_t i = 1; i < width; ++i) {
                if (blk[i] == mask && (conf[i] >= threshold || step >= max_steps)) {
                    blk[i] = cand[i];
                    seen[cand[i]] = 1;
                    ++n_filled;
                }
            }
            if (n_filled == 0) {
                blk[best] = cand[best];
                seen[cand[best]] = 1;
            }

            if (visual) {
                std::string cur;
                for (int32_t i = 1; i < width; ++i) {
                    cur += blk[i] == mask ? std::string("_") : token_text(vocab, blk[i]);
                }
                printf("\033[2J\033[H");
                printf("block %d  step %d  masked %d/%d\n\n%s%s\033[7m%s\033[0m\n", n_blocks + 1, step,
                        n_masked - std::max(n_filled, 1), width - 1, text.c_str(), committed.c_str(), cur.c_str());
                fflush(stdout);
            }
        }
        ++n_blocks;
        n_steps += step;
        steps_list += (steps_list.empty() ? "" : " ") + std::to_string(step);

        // keep tokens up to the first end-of-generation token
        int32_t n_new = width - 1;
        for (int32_t i = 1; i < width; ++i) {
            if (llama_vocab_is_eog(vocab, blk[i])) {
                n_new = i - 1;
                done  = true;
                break;
            }
        }
        n_new = std::min(n_new, n_predict - (int32_t) out.size());
        std::string piece;
        for (int32_t i = 1; i <= n_new; ++i) {
            out.push_back(blk[i]);
            piece += token_text(vocab, blk[i]);
        }
        committed += piece;
        if (!visual) {
            printf("%s", piece.c_str());
            fflush(stdout);
        }
        if (done || (int32_t) out.size() >= n_predict) {
            break;
        }

        // commit the resolved block causally; its last token becomes the next anchor
        batch.clear();
        for (int32_t i = 0; i < width - 1; ++i) {
            batch.add(blk[i], pos + i, 0, false);
        }
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
            LOG_ERR("error: block commit failed\n");
            return 1;
        }
        anchor = blk[width - 1];
        pos   += width - 1;
    }
    const int64_t t_end = ggml_time_us();

    if (visual) {
        printf("\033[2J\033[H%s%s\n", text.c_str(), committed.c_str());
    } else {
        printf("\n");
    }
    const double gen_s = (t_end - t_prefill) / 1e6;
    LOG_INF("\nclover-diffusion: prompt prefill %.3f s; %zu tokens in %d blocks, %d denoising steps (%.2f tokens/step), %.3f s, %.2f t/s\n",
            (t_prefill - t_start) / 1e6, out.size(), n_blocks, n_steps, n_steps ? (double) out.size() / n_steps : 0.0, gen_s,
            gen_s > 0 ? out.size() / gen_s : 0.0);
    LOG_INF("clover-diffusion: steps per block: %s\n", steps_list.c_str());
    if (n_steps > 0) {
        LOG_INF("clover-diffusion: per step: block pass + logits %.2f ms, state restore %.2f ms, sampling %.2f ms (%d threads)\n",
                t_dec / 1e3 / n_steps, t_rst / 1e3 / n_steps, t_smp / 1e3 / n_steps, n_threads);
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
