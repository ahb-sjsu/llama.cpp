// Tiered KV cache (Sprint 4) — implementation.
//
// See llama-kv-tiered.h for the design overview.

#include "llama-kv-tiered.h"

#include "ggml.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>

namespace llama_kv_tq {

tiered_cache::tiered_cache(const tiered_cache_config & cfg)
    : cfg_(cfg)
    , rot_(init_rotation(cfg.head_dim, cfg.rotation_seed))
{
    if (cfg_.n_layers   <= 0) throw std::invalid_argument("n_layers must be > 0");
    if (cfg_.n_kv_heads <= 0) throw std::invalid_argument("n_kv_heads must be > 0");
    if (cfg_.head_dim   <= 0) throw std::invalid_argument("head_dim must be > 0");
    if (cfg_.hot_window <  0) throw std::invalid_argument("hot_window must be >= 0");
}

void tiered_cache::add_token(const float * k_all, const float * v_all) {
    const int slots = cfg_.n_layers * cfg_.n_kv_heads;
    const int D     = cfg_.head_dim;

    hot_token tok;
    tok.pos = n_tokens_;
    tok.k.resize(slots);
    tok.v.resize(slots);
    for (int s = 0; s < slots; ++s) {
        tok.k[s].assign(k_all + s * D, k_all + (s + 1) * D);
        tok.v[s].assign(v_all + s * D, v_all + (s + 1) * D);
    }
    hot_.push_back(std::move(tok));
    ++n_tokens_;

    while (static_cast<int>(hot_.size()) > cfg_.hot_window) {
        evict_oldest_hot_();
    }
}

void tiered_cache::evict_oldest_hot_() {
    hot_token & front = hot_.front();
    const int slots = cfg_.n_layers * cfg_.n_kv_heads;

    cold_token cold;
    cold.k.resize(slots);
    cold.v.resize(slots);
    for (int s = 0; s < slots; ++s) {
        cold.k[s] = compress_vector(front.k[s].data(), cfg_.head_dim,
                                    cfg_.cold_bits, rot_);
        cold.v[s] = compress_vector(front.v[s].data(), cfg_.head_dim,
                                    cfg_.cold_bits, rot_);
    }

    // Cold storage is densely indexed by absolute position. Because we
    // always evict the oldest hot token, cold_ grows by exactly one
    // entry per call and that entry sits at index front.pos.
    if (static_cast<int>(cold_.size()) != front.pos) {
        throw std::runtime_error(
            "tiered_cache: cold/hot index drift at pos " +
            std::to_string(front.pos));
    }
    cold_.push_back(std::move(cold));
    hot_.pop_front();
}

void tiered_cache::get_kv(int pos, int layer, int head,
                          float * k_out, float * v_out) const
{
    if (pos < 0 || pos >= n_tokens_) {
        throw std::out_of_range("tiered_cache::get_kv pos out of range");
    }
    if (layer < 0 || layer >= cfg_.n_layers) {
        throw std::out_of_range("tiered_cache::get_kv layer out of range");
    }
    if (head < 0 || head >= cfg_.n_kv_heads) {
        throw std::out_of_range("tiered_cache::get_kv head out of range");
    }
    const int slot = slot_index_(layer, head);
    const int D    = cfg_.head_dim;

    const int n_cold = static_cast<int>(cold_.size());
    if (pos < n_cold) {
        // Cold path: decompress on demand.
        decompress_vector(cold_[pos].k[slot], D, cfg_.cold_bits, rot_, k_out);
        decompress_vector(cold_[pos].v[slot], D, cfg_.cold_bits, rot_, v_out);
        return;
    }

    // Hot path: O(1) — hot front always sits at absolute position n_cold,
    // so the offset into the hot deque is simply (pos - n_cold).
    const int idx = pos - n_cold;
    const hot_token & tok = hot_[idx];
    // Defensive: the deque order should always match absolute positions.
    if (tok.pos != pos) {
        throw std::runtime_error("tiered_cache: hot index drift");
    }
    std::copy(tok.k[slot].begin(), tok.k[slot].end(), k_out);
    std::copy(tok.v[slot].begin(), tok.v[slot].end(), v_out);
}

void tiered_cache::read_token_k(int pos, std::vector<float> & out) const {
    if (pos < 0 || pos >= n_tokens_) {
        throw std::out_of_range(
            "tiered_cache::read_token_k pos=" + std::to_string(pos) +
            " outside [0, " + std::to_string(n_tokens_) + ")");
    }
    const int slots = cfg_.n_layers * cfg_.n_kv_heads;
    const int D     = cfg_.head_dim;
    out.assign((size_t) slots * D, 0.0f);

    const int n_cold = static_cast<int>(cold_.size());
    if (pos < n_cold) {
        // Cold path.
        std::vector<float> tmp_v(D);
        for (int s = 0; s < slots; ++s) {
            decompress_vector(cold_[pos].k[s], D, cfg_.cold_bits, rot_,
                              out.data() + (size_t) s * D);
            (void) tmp_v; // v side intentionally untouched
        }
    } else {
        // Hot path — copy verbatim.
        const int hot_idx = pos - n_cold;
        const hot_token & tok = hot_[hot_idx];
        for (int s = 0; s < slots; ++s) {
            std::copy(tok.k[s].begin(), tok.k[s].end(),
                      out.data() + (size_t) s * D);
        }
    }
}

void tiered_cache::read_token_v(int pos, std::vector<float> & out) const {
    if (pos < 0 || pos >= n_tokens_) {
        throw std::out_of_range(
            "tiered_cache::read_token_v pos=" + std::to_string(pos) +
            " outside [0, " + std::to_string(n_tokens_) + ")");
    }
    const int slots = cfg_.n_layers * cfg_.n_kv_heads;
    const int D     = cfg_.head_dim;
    out.assign((size_t) slots * D, 0.0f);

    const int n_cold = static_cast<int>(cold_.size());
    if (pos < n_cold) {
        for (int s = 0; s < slots; ++s) {
            decompress_vector(cold_[pos].v[s], D, cfg_.cold_bits, rot_,
                              out.data() + (size_t) s * D);
        }
    } else {
        const int hot_idx = pos - n_cold;
        const hot_token & tok = hot_[hot_idx];
        for (int s = 0; s < slots; ++s) {
            std::copy(tok.v[s].begin(), tok.v[s].end(),
                      out.data() + (size_t) s * D);
        }
    }
}

void tiered_cache::materialize_fp16_rows(int start_pos,
                                         int n_positions,
                                         bool is_v,
                                         std::vector<uint16_t> & out) const
{
    if (n_positions < 0) {
        throw std::out_of_range(
            "tiered_cache::materialize_fp16_rows n_positions=" +
            std::to_string(n_positions) + " must be >= 0");
    }
    if (n_positions == 0) {
        out.clear();
        return;
    }
    if (start_pos < 0 || start_pos + n_positions > n_tokens_) {
        throw std::out_of_range(
            "tiered_cache::materialize_fp16_rows [" +
            std::to_string(start_pos) + ", " +
            std::to_string(start_pos + n_positions) +
            ") outside [0, " + std::to_string(n_tokens_) + ")");
    }

    const int slots      = cfg_.n_layers * cfg_.n_kv_heads;
    const int D          = cfg_.head_dim;
    const size_t row_n   = (size_t) slots * D;
    const size_t tot_n   = row_n * (size_t) n_positions;

    out.assign(tot_n, 0);

    std::vector<float> row_fp32;
    row_fp32.reserve(row_n);

    for (int t = 0; t < n_positions; ++t) {
        const int pos = start_pos + t;
        if (is_v) {
            read_token_v(pos, row_fp32);
        } else {
            read_token_k(pos, row_fp32);
        }
        ggml_fp32_to_fp16_row(
            row_fp32.data(),
            reinterpret_cast<ggml_fp16_t *>(out.data() + (size_t) t * row_n),
            row_n);
    }
}

memory_stats tiered_cache::stats() const {
    memory_stats s;
    s.hot_tokens  = static_cast<int>(hot_.size());
    s.cold_tokens = static_cast<int>(cold_.size());

    const int slots = cfg_.n_layers * cfg_.n_kv_heads;

    // Hot: 2 (K+V) * slots * head_dim floats per token.
    // (We store fp32 internally for simplicity; "fp16 baseline" below
    // counts the model's actual fp16 cost so the ratio is honest.)
    s.hot_bytes = static_cast<size_t>(s.hot_tokens) *
                  slots * cfg_.head_dim * sizeof(float) * 2;

    // Cold: 2 * slots * block_size_bytes per token.
    const size_t per_block = block_size_bytes(cfg_.head_dim, cfg_.cold_bits);
    s.cold_bytes = static_cast<size_t>(s.cold_tokens) *
                   slots * per_block * 2;

    // Baseline: what fp16 storage of the same n_tokens() would cost.
    s.fp16_total_bytes = static_cast<size_t>(n_tokens_) *
                         slots * cfg_.head_dim * 2 /*sizeof(fp16)*/ * 2;

    const size_t actual = s.hot_bytes + s.cold_bytes;
    s.compression_ratio = actual > 0
        ? static_cast<double>(s.fp16_total_bytes) / static_cast<double>(actual)
        : 1.0;
    return s;
}

}  // namespace llama_kv_tq
