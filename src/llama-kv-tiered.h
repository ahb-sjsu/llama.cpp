#pragma once

// Tiered KV cache (Sprint 4) — hot fp16 + cold TurboQuant.
//
// This is a self-contained data structure that proves the
// hot/cold tiering model works end-to-end with the TurboQuant
// compression from Sprint 2/3. It is intentionally NOT yet
// derived from llama_memory_i / llama_kv_cache — Sprint 4b
// will adapt this storage to the existing class so the
// existing prefill and generation graphs route through it
// transparently.
//
// Threading: not internally synchronised. Callers must
// serialise add_token / get_kv with respect to each other.
//
// Storage note: the hot tier is currently stored as fp32 internally.
// This makes correctness checks trivial (bit-exact comparison), but
// inflates the hot byte count vs a real fp16 KV cache. Sprint 4b
// will switch hot to fp16; the public API (add_token / get_kv) is
// unchanged because it operates on fp32 in either case.

#include "llama-kv-turboquant.h"

#include <cstdint>
#include <deque>
#include <vector>

namespace llama_kv_tq {

struct tiered_cache_config {
    int      n_layers     = 1;
    int      n_kv_heads   = 1;
    int      head_dim     = 128;
    int      hot_window   = 512;     // tokens kept in fp16
    bits     cold_bits    = BITS_3;
    uint32_t rotation_seed = 42;
};

struct memory_stats {
    int    hot_tokens          = 0;
    int    cold_tokens         = 0;
    size_t hot_bytes           = 0;
    size_t cold_bytes          = 0;
    size_t fp16_total_bytes    = 0;   // what storage would cost at fp16
    double compression_ratio   = 1.0; // fp16_total_bytes / (hot+cold)
};

class tiered_cache {
public:
    explicit tiered_cache(const tiered_cache_config & cfg);

    // Append one token's K and V across all layers and heads.
    // Both pointers must be of length n_layers * n_kv_heads * head_dim.
    // Slot layout: [layer][head][head_dim] (row-major).
    void add_token(const float * k_all, const float * v_all);

    // Read back one (layer, head) of one token at absolute position pos.
    // pos must satisfy 0 <= pos < n_tokens(). Hot lookups are exact;
    // cold lookups decompress on demand.
    void get_kv(int pos, int layer, int head,
                float * k_out, float * v_out) const;

    int n_tokens()      const { return n_tokens_; }
    int n_hot_tokens()  const { return static_cast<int>(hot_.size()); }
    int n_cold_tokens() const { return static_cast<int>(cold_.size()); }

    memory_stats stats() const;

    const tiered_cache_config & config() const { return cfg_; }

private:
    void evict_oldest_hot_();   // moves hot front into cold tier
    int  slot_index_(int layer, int head) const {
        return layer * cfg_.n_kv_heads + head;
    }

    tiered_cache_config cfg_;
    rotation_matrix     rot_;

    // One vector<float>(head_dim) per (layer, head) slot.
    struct hot_token {
        int                              pos;
        std::vector<std::vector<float>>  k;  // [slot] -> head_dim floats
        std::vector<std::vector<float>>  v;
    };
    std::deque<hot_token> hot_;

    // Compressed blocks for tokens that have aged out of the hot window.
    // Indexed by absolute position (cold_[pos] is valid for pos < hot_.front().pos).
    struct cold_token {
        std::vector<compressed_block> k;  // [slot]
        std::vector<compressed_block> v;
    };
    std::vector<cold_token> cold_;

    int n_tokens_ = 0;
};

}  // namespace llama_kv_tq
