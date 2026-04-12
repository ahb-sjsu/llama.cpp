// Tests for the tiered KV cache (Sprint 4).
//
// Coverage:
//   1. Hot-only correctness (no eviction → exact byte equality)
//   2. Cold reconstruction (post-eviction → cosine similarity > target)
//   3. Eviction order: oldest-first
//   4. Memory stats accuracy (hot vs cold byte counts, ratio sanity)
//   5. Multi-layer / multi-head slot isolation (no cross-talk)
//   6. Stress: 2K tokens / 4 layers / 8 heads / window 64 — verify
//      tier balance, no exceptions, all positions readable.
//   7. Out-of-range pos / layer / head throw.

#include "../src/llama-kv-tiered.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using llama_kv_tq::BITS_2;
using llama_kv_tq::BITS_3;
using llama_kv_tq::BITS_4;
using llama_kv_tq::memory_stats;
using llama_kv_tq::tiered_cache;
using llama_kv_tq::tiered_cache_config;

// -------------------------------------------------------------------- //
// Test harness                                                          //
// -------------------------------------------------------------------- //

static int n_failed = 0;
static int n_total  = 0;

#define CHECK(cond, msg) do {                                            \
    n_total++;                                                           \
    if (!(cond)) {                                                       \
        n_failed++;                                                      \
        std::fprintf(stderr, "FAIL  %s:%d  %s -- %s\n",                  \
            __FILE__, __LINE__, #cond, msg);                             \
    }                                                                    \
} while (0)

#define CHECK_NEAR(a, b, tol, msg) do {                                  \
    n_total++;                                                           \
    float _a = (a), _b = (b);                                            \
    if (std::abs(_a - _b) > (tol)) {                                     \
        n_failed++;                                                      \
        std::fprintf(stderr,                                             \
            "FAIL  %s:%d  |%g - %g| > %g -- %s\n",                       \
            __FILE__, __LINE__, _a, _b, (float)(tol), msg);              \
    }                                                                    \
} while (0)

static float cosine_similarity(const float * a, const float * b, int n) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (int i = 0; i < n; ++i) {
        dot += static_cast<double>(a[i]) * b[i];
        na  += static_cast<double>(a[i]) * a[i];
        nb  += static_cast<double>(b[i]) * b[i];
    }
    double denom = std::sqrt(na) * std::sqrt(nb);
    return denom > 1e-12 ? static_cast<float>(dot / denom) : 0.0f;
}

// Generate a deterministic "token" — fills k_out and v_out with
// (n_layers * n_kv_heads * head_dim) floats. The seed is mixed
// from token position so each token is unique.
static void make_token(int pos, const tiered_cache_config & cfg,
                       std::vector<float> & k, std::vector<float> & v)
{
    const size_t total = static_cast<size_t>(cfg.n_layers) *
                         cfg.n_kv_heads * cfg.head_dim;
    k.resize(total);
    v.resize(total);
    std::mt19937 rng(static_cast<uint32_t>(pos) * 2654435761u);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    for (auto & x : k) x = normal(rng);
    for (auto & x : v) x = normal(rng);
}

// -------------------------------------------------------------------- //
// 1. Hot-only correctness                                               //
// -------------------------------------------------------------------- //

static void test_hot_only_exact() {
    std::printf("test_hot_only_exact\n");
    tiered_cache_config cfg;
    cfg.n_layers   = 2;
    cfg.n_kv_heads = 4;
    cfg.head_dim   = 64;
    cfg.hot_window = 32;
    cfg.cold_bits  = BITS_3;

    tiered_cache cache(cfg);

    std::vector<std::vector<float>> all_k(8), all_v(8);
    for (int t = 0; t < 8; ++t) {
        make_token(t, cfg, all_k[t], all_v[t]);
        cache.add_token(all_k[t].data(), all_v[t].data());
    }

    CHECK(cache.n_tokens()      == 8, "n_tokens after 8 add");
    CHECK(cache.n_hot_tokens()  == 8, "all hot before eviction");
    CHECK(cache.n_cold_tokens() == 0, "no cold tokens yet");

    // Read every (pos, layer, head) and verify exact equality with input.
    std::vector<float> k_out(cfg.head_dim), v_out(cfg.head_dim);
    int matched = 0;
    for (int pos = 0; pos < 8; ++pos) {
        for (int layer = 0; layer < cfg.n_layers; ++layer) {
            for (int head = 0; head < cfg.n_kv_heads; ++head) {
                cache.get_kv(pos, layer, head, k_out.data(), v_out.data());
                int slot = layer * cfg.n_kv_heads + head;
                const float * k_ref = all_k[pos].data() + slot * cfg.head_dim;
                const float * v_ref = all_v[pos].data() + slot * cfg.head_dim;
                bool ok = true;
                for (int i = 0; i < cfg.head_dim; ++i) {
                    if (k_out[i] != k_ref[i] || v_out[i] != v_ref[i]) {
                        ok = false; break;
                    }
                }
                CHECK(ok, "hot-tier exact byte equality");
                if (ok) ++matched;
            }
        }
    }
    std::printf("  matched %d hot slots (expected 64)\n", matched);
}

// -------------------------------------------------------------------- //
// 2. Cold reconstruction after eviction                                 //
// -------------------------------------------------------------------- //

static void test_cold_reconstruction() {
    std::printf("test_cold_reconstruction\n");
    tiered_cache_config cfg;
    cfg.n_layers   = 1;
    cfg.n_kv_heads = 2;
    cfg.head_dim   = 128;
    cfg.hot_window = 4;        // tiny so most tokens evict
    cfg.cold_bits  = BITS_3;

    tiered_cache cache(cfg);

    const int n_tok = 24;
    std::vector<std::vector<float>> all_k(n_tok), all_v(n_tok);
    for (int t = 0; t < n_tok; ++t) {
        make_token(t, cfg, all_k[t], all_v[t]);
        cache.add_token(all_k[t].data(), all_v[t].data());
    }
    CHECK(cache.n_hot_tokens()  == cfg.hot_window, "hot capped at window");
    CHECK(cache.n_cold_tokens() == n_tok - cfg.hot_window, "rest evicted");
    CHECK(cache.n_tokens()      == n_tok, "n_tokens preserved");

    // Cold tokens should reconstruct to ~the original (3-bit target).
    std::vector<float> k_out(cfg.head_dim), v_out(cfg.head_dim);
    double total_k_cos = 0.0, total_v_cos = 0.0;
    int    cold_reads  = 0;
    for (int pos = 0; pos < cache.n_cold_tokens(); ++pos) {
        for (int layer = 0; layer < cfg.n_layers; ++layer) {
            for (int head = 0; head < cfg.n_kv_heads; ++head) {
                cache.get_kv(pos, layer, head, k_out.data(), v_out.data());
                int slot = layer * cfg.n_kv_heads + head;
                const float * k_ref = all_k[pos].data() + slot * cfg.head_dim;
                const float * v_ref = all_v[pos].data() + slot * cfg.head_dim;
                total_k_cos += cosine_similarity(k_out.data(), k_ref, cfg.head_dim);
                total_v_cos += cosine_similarity(v_out.data(), v_ref, cfg.head_dim);
                ++cold_reads;
            }
        }
    }
    double mean_k = total_k_cos / cold_reads;
    double mean_v = total_v_cos / cold_reads;
    std::printf("  mean cold cosine: K=%.4f, V=%.4f over %d reads\n",
        mean_k, mean_v, cold_reads);
    CHECK(mean_k > 0.93, "cold K cosine > 0.93 at 3 bits");
    CHECK(mean_v > 0.93, "cold V cosine > 0.93 at 3 bits");

    // Hot tokens (the last hot_window) should still be exact.
    int hot_first = cache.n_cold_tokens();
    int exact = 0;
    for (int pos = hot_first; pos < n_tok; ++pos) {
        for (int layer = 0; layer < cfg.n_layers; ++layer) {
            for (int head = 0; head < cfg.n_kv_heads; ++head) {
                cache.get_kv(pos, layer, head, k_out.data(), v_out.data());
                int slot = layer * cfg.n_kv_heads + head;
                const float * k_ref = all_k[pos].data() + slot * cfg.head_dim;
                const float * v_ref = all_v[pos].data() + slot * cfg.head_dim;
                bool ok = true;
                for (int i = 0; i < cfg.head_dim; ++i) {
                    if (k_out[i] != k_ref[i] || v_out[i] != v_ref[i]) {
                        ok = false; break;
                    }
                }
                if (ok) ++exact;
            }
        }
    }
    int expected_exact = cfg.hot_window * cfg.n_layers * cfg.n_kv_heads;
    CHECK(exact == expected_exact, "all hot reads remain bit-exact");
    std::printf("  hot exact reads: %d / %d\n", exact, expected_exact);
}

// -------------------------------------------------------------------- //
// 3. Eviction order: front-to-back                                      //
// -------------------------------------------------------------------- //

static void test_eviction_order() {
    std::printf("test_eviction_order\n");
    tiered_cache_config cfg;
    cfg.n_layers   = 1;
    cfg.n_kv_heads = 1;
    cfg.head_dim   = 32;
    cfg.hot_window = 3;
    cfg.cold_bits  = BITS_4;

    tiered_cache cache(cfg);
    std::vector<std::vector<float>> ks(7), vs(7);
    for (int t = 0; t < 7; ++t) {
        make_token(t, cfg, ks[t], vs[t]);
        cache.add_token(ks[t].data(), vs[t].data());
    }

    // After 7 adds with window=3: positions 0..3 are cold, 4..6 are hot.
    CHECK(cache.n_cold_tokens() == 4, "first 4 evicted");
    CHECK(cache.n_hot_tokens()  == 3, "last 3 hot");

    // Verify positions 0..3 reconstruct (cold path) and 4..6 are exact.
    std::vector<float> k_out(cfg.head_dim), v_out(cfg.head_dim);
    for (int pos = 0; pos < 7; ++pos) {
        cache.get_kv(pos, 0, 0, k_out.data(), v_out.data());
        bool exact = true;
        for (int i = 0; i < cfg.head_dim; ++i) {
            if (k_out[i] != ks[pos][i]) { exact = false; break; }
        }
        if (pos < 4) {
            CHECK(!exact, "cold positions are not bit-exact");
            float c = cosine_similarity(k_out.data(), ks[pos].data(),
                                        cfg.head_dim);
            CHECK(c > 0.95f, "cold cosine > 0.95 at 4 bits");
        } else {
            CHECK(exact, "hot positions are bit-exact");
        }
    }
}

// -------------------------------------------------------------------- //
// 4. Memory stats accuracy                                              //
// -------------------------------------------------------------------- //

static void test_memory_stats() {
    std::printf("test_memory_stats\n");
    tiered_cache_config cfg;
    cfg.n_layers   = 4;
    cfg.n_kv_heads = 8;
    cfg.head_dim   = 128;
    cfg.hot_window = 16;
    cfg.cold_bits  = BITS_3;

    tiered_cache cache(cfg);
    const int n_tok = 100;
    std::vector<float> k, v;
    for (int t = 0; t < n_tok; ++t) {
        make_token(t, cfg, k, v);
        cache.add_token(k.data(), v.data());
    }

    auto s = cache.stats();
    CHECK(s.hot_tokens == cfg.hot_window, "hot stats matches window");
    CHECK(s.cold_tokens == n_tok - cfg.hot_window, "cold stats matches");

    // Check byte counts against hand calculation.
    int slots = cfg.n_layers * cfg.n_kv_heads;
    size_t per_block_3bit = sizeof(float) + (cfg.head_dim * 3 + 7) / 8;
    size_t expected_cold = static_cast<size_t>(s.cold_tokens) *
                           slots * per_block_3bit * 2;
    CHECK(s.cold_bytes == expected_cold, "cold bytes hand-calc match");

    // fp16 baseline is 2 bytes per element * 2 (K+V).
    size_t expected_fp16 = static_cast<size_t>(n_tok) *
                           slots * cfg.head_dim * 2 * 2;
    CHECK(s.fp16_total_bytes == expected_fp16, "fp16 baseline correct");

    // Compression ratio should be substantially > 1 since most tokens are cold.
    std::printf("  hot=%zu B, cold=%zu B, fp16=%zu B, ratio=%.2fx\n",
        s.hot_bytes, s.cold_bytes, s.fp16_total_bytes, s.compression_ratio);
    CHECK(s.compression_ratio > 1.5,
          "ratio > 1.5x with 84/16 cold/hot split at 3 bits");
}

// -------------------------------------------------------------------- //
// 5. Multi-layer slot isolation (no cross-talk)                         //
// -------------------------------------------------------------------- //

static void test_slot_isolation() {
    std::printf("test_slot_isolation\n");
    tiered_cache_config cfg;
    cfg.n_layers   = 3;
    cfg.n_kv_heads = 5;
    cfg.head_dim   = 64;
    cfg.hot_window = 2;
    cfg.cold_bits  = BITS_4;

    tiered_cache cache(cfg);
    // Build a token whose every slot is uniquely identifiable: slot s
    // gets an all-(s+1.0f) k vector and an all-(-(s+1.0f)) v vector.
    int slots = cfg.n_layers * cfg.n_kv_heads;
    std::vector<float> k(slots * cfg.head_dim), v(slots * cfg.head_dim);
    for (int s = 0; s < slots; ++s) {
        for (int i = 0; i < cfg.head_dim; ++i) {
            k[s * cfg.head_dim + i] = static_cast<float>(s + 1);
            v[s * cfg.head_dim + i] = -static_cast<float>(s + 1);
        }
    }
    // Add several so all become cold, exercising the compression path.
    for (int t = 0; t < 5; ++t) {
        cache.add_token(k.data(), v.data());
    }
    CHECK(cache.n_cold_tokens() >= 3, "at least 3 cold tokens");

    std::vector<float> k_out(cfg.head_dim), v_out(cfg.head_dim);
    for (int layer = 0; layer < cfg.n_layers; ++layer) {
        for (int head = 0; head < cfg.n_kv_heads; ++head) {
            cache.get_kv(0, layer, head, k_out.data(), v_out.data());
            int slot = layer * cfg.n_kv_heads + head;
            float expected_k =  static_cast<float>(slot + 1);
            float expected_v = -static_cast<float>(slot + 1);
            // Mean of reconstructed vector should be close to expected
            // (constant vectors lose almost no info under PolarQuant since
            // they are 1D in the rotated space).
            double mean_k = 0.0, mean_v = 0.0;
            for (int i = 0; i < cfg.head_dim; ++i) {
                mean_k += k_out[i];
                mean_v += v_out[i];
            }
            mean_k /= cfg.head_dim;
            mean_v /= cfg.head_dim;
            CHECK_NEAR((float)mean_k, expected_k, 0.5f, "K slot value preserved");
            CHECK_NEAR((float)mean_v, expected_v, 0.5f, "V slot value preserved");
        }
    }
}

// -------------------------------------------------------------------- //
// 6. Stress: many tokens, many layers, many heads                       //
// -------------------------------------------------------------------- //

static void test_stress() {
    std::printf("test_stress\n");
    tiered_cache_config cfg;
    cfg.n_layers   = 4;
    cfg.n_kv_heads = 8;
    cfg.head_dim   = 128;
    cfg.hot_window = 64;
    cfg.cold_bits  = BITS_3;

    tiered_cache cache(cfg);
    const int n_tok = 2000;

    std::vector<float> k, v;
    for (int t = 0; t < n_tok; ++t) {
        make_token(t, cfg, k, v);
        cache.add_token(k.data(), v.data());
    }

    CHECK(cache.n_tokens()      == n_tok,           "n_tokens after stress");
    CHECK(cache.n_hot_tokens()  == cfg.hot_window,  "hot capped");
    CHECK(cache.n_cold_tokens() == n_tok - cfg.hot_window, "cold count");

    // Spot-check 50 random positions for readability.
    std::mt19937 rng(424242);
    std::uniform_int_distribution<int> upos(0, n_tok - 1);
    std::uniform_int_distribution<int> ulay(0, cfg.n_layers - 1);
    std::uniform_int_distribution<int> uhd (0, cfg.n_kv_heads - 1);
    std::vector<float> k_out(cfg.head_dim), v_out(cfg.head_dim);
    int n_ok = 0;
    for (int trial = 0; trial < 50; ++trial) {
        int pos = upos(rng), layer = ulay(rng), head = uhd(rng);
        cache.get_kv(pos, layer, head, k_out.data(), v_out.data());
        // Re-derive the expected value from the same seed.
        std::vector<float> k_ref, v_ref;
        make_token(pos, cfg, k_ref, v_ref);
        int slot = layer * cfg.n_kv_heads + head;
        float c = cosine_similarity(k_out.data(),
                                    k_ref.data() + slot * cfg.head_dim,
                                    cfg.head_dim);
        if (c > 0.85f) ++n_ok;
    }
    std::printf("  stress: %d/50 random reads above cos=0.85\n", n_ok);
    CHECK(n_ok >= 48, ">= 48 of 50 random reads cosine > 0.85");

    // Memory ratio against the fp16 baseline.
    //
    // Note: this prototype stores hot tokens as fp32 internally (twice
    // the bytes of fp16). Sprint 4b will move hot to fp16, which roughly
    // doubles the effective ratio on the hot region. With the current
    // fp32-hot we expect ~3.8x at 3 bits, 64/1936 split; production
    // (fp16 hot) would land around ~7x.
    auto s = cache.stats();
    std::printf("  stress ratio = %.2fx (hot=%zuMB cold=%zuMB fp16=%zuMB)\n",
        s.compression_ratio,
        s.hot_bytes / (1024 * 1024),
        s.cold_bytes / (1024 * 1024),
        s.fp16_total_bytes / (1024 * 1024));
    CHECK(s.compression_ratio > 3.5,
          "stress ratio > 3.5x at 3 bits, 64/1936 split (fp32-hot prototype)");
}

// -------------------------------------------------------------------- //
// 7. Out-of-range input throws                                          //
// -------------------------------------------------------------------- //

static void test_out_of_range() {
    std::printf("test_out_of_range\n");
    tiered_cache_config cfg;
    cfg.n_layers   = 1;
    cfg.n_kv_heads = 2;
    cfg.head_dim   = 32;
    cfg.hot_window = 2;

    tiered_cache cache(cfg);
    std::vector<float> k, v;
    make_token(0, cfg, k, v);
    cache.add_token(k.data(), v.data());

    std::vector<float> k_out(cfg.head_dim), v_out(cfg.head_dim);
    bool threw = false;
    try { cache.get_kv(5, 0, 0, k_out.data(), v_out.data()); }
    catch (const std::out_of_range &) { threw = true; }
    CHECK(threw, "pos out of range throws");

    threw = false;
    try { cache.get_kv(0, 9, 0, k_out.data(), v_out.data()); }
    catch (const std::out_of_range &) { threw = true; }
    CHECK(threw, "layer out of range throws");

    threw = false;
    try { cache.get_kv(0, 0, 9, k_out.data(), v_out.data()); }
    catch (const std::out_of_range &) { threw = true; }
    CHECK(threw, "head out of range throws");

    // Invalid config also throws.
    threw = false;
    try {
        tiered_cache_config bad; bad.head_dim = 0;
        tiered_cache c(bad);
    } catch (const std::invalid_argument &) { threw = true; }
    CHECK(threw, "invalid config throws");
}

// -------------------------------------------------------------------- //
// 8. read_token_k / read_token_v correctness (Sprint 4c step 3b)       //
// -------------------------------------------------------------------- //

static void test_read_token_hot_exact() {
    std::printf("test_read_token_hot_exact\n");
    tiered_cache_config cfg;
    cfg.n_layers   = 1;
    cfg.n_kv_heads = 2;
    cfg.head_dim   = 16;
    cfg.hot_window = 8;
    cfg.cold_bits  = BITS_3;

    tiered_cache cache(cfg);

    // Push 3 tokens, all stay hot (window=8).
    std::vector<std::vector<float>> all_k(3), all_v(3);
    for (int t = 0; t < 3; ++t) {
        make_token(t, cfg, all_k[t], all_v[t]);
        cache.add_token(all_k[t].data(), all_v[t].data());
    }

    std::vector<float> buf;
    for (int pos = 0; pos < 3; ++pos) {
        cache.read_token_k(pos, buf);
        const int n = cfg.n_kv_heads * cfg.head_dim;
        CHECK((int) buf.size() == n, "read_token_k size");
        bool ok_k = true;
        for (int i = 0; i < n; ++i) {
            if (buf[i] != all_k[pos][i]) { ok_k = false; break; }
        }
        CHECK(ok_k, "read_token_k hot is bit-exact");

        cache.read_token_v(pos, buf);
        bool ok_v = true;
        for (int i = 0; i < n; ++i) {
            if (buf[i] != all_v[pos][i]) { ok_v = false; break; }
        }
        CHECK(ok_v, "read_token_v hot is bit-exact");
    }
}

static void test_read_token_cold_cosine() {
    std::printf("test_read_token_cold_cosine\n");
    tiered_cache_config cfg;
    cfg.n_layers   = 1;
    cfg.n_kv_heads = 2;
    cfg.head_dim   = 64;
    cfg.hot_window = 2;       // force most to evict
    cfg.cold_bits  = BITS_3;  // ~0.93 cosine target

    tiered_cache cache(cfg);

    const int n_tok = 16;
    std::vector<std::vector<float>> all_k(n_tok), all_v(n_tok);
    for (int t = 0; t < n_tok; ++t) {
        make_token(t, cfg, all_k[t], all_v[t]);
        cache.add_token(all_k[t].data(), all_v[t].data());
    }

    CHECK(cache.n_cold_tokens() == n_tok - cfg.hot_window,
        "expected cold tokens count");

    std::vector<float> buf;
    double tot_cos_k = 0, tot_cos_v = 0;
    int n_cold = cache.n_cold_tokens();
    for (int pos = 0; pos < n_cold; ++pos) {
        cache.read_token_k(pos, buf);
        tot_cos_k += cosine_similarity(
            buf.data(), all_k[pos].data(),
            cfg.n_kv_heads * cfg.head_dim);
        cache.read_token_v(pos, buf);
        tot_cos_v += cosine_similarity(
            buf.data(), all_v[pos].data(),
            cfg.n_kv_heads * cfg.head_dim);
    }
    double mean_k = tot_cos_k / n_cold;
    double mean_v = tot_cos_v / n_cold;
    std::printf("  cold cosine: K=%.4f V=%.4f over %d tokens\n",
        mean_k, mean_v, n_cold);
    CHECK(mean_k > 0.93, "read_token_k cold cosine > 0.93 at 3-bit");
    CHECK(mean_v > 0.93, "read_token_v cold cosine > 0.93 at 3-bit");
}

static void test_read_token_out_of_range() {
    std::printf("test_read_token_out_of_range\n");
    tiered_cache_config cfg;
    cfg.n_layers   = 1;
    cfg.n_kv_heads = 1;
    cfg.head_dim   = 8;
    cfg.hot_window = 4;
    tiered_cache cache(cfg);

    std::vector<float> k, v;
    make_token(0, cfg, k, v);
    cache.add_token(k.data(), v.data());

    std::vector<float> buf;
    bool threw;

    threw = false;
    try { cache.read_token_k(-1, buf); }
    catch (const std::out_of_range &) { threw = true; }
    CHECK(threw, "read_token_k(-1) throws");

    threw = false;
    try { cache.read_token_k(1, buf); }
    catch (const std::out_of_range &) { threw = true; }
    CHECK(threw, "read_token_k past end throws");

    threw = false;
    try { cache.read_token_v(99, buf); }
    catch (const std::out_of_range &) { threw = true; }
    CHECK(threw, "read_token_v past end throws");
}

// -------------------------------------------------------------------- //
// 9. materialize_fp16_rows correctness                                  //
// -------------------------------------------------------------------- //

// Helper: reverse an fp16 row back to fp32 for comparison.
static void fp16_row_to_fp32(const uint16_t * src,
                             int n, std::vector<float> & dst)
{
    dst.resize(n);
    // Minimal fp16 → fp32 reference (ggml_fp16_to_fp32_row is in a
    // separate TU; we reproduce the IEEE half semantics here to make
    // the test independent). Bit-layout: 1 sign | 5 exp | 10 mant.
    for (int i = 0; i < n; ++i) {
        uint16_t h  = src[i];
        uint32_t s  = (h >> 15) & 0x1;
        uint32_t e  = (h >> 10) & 0x1F;
        uint32_t m  = h & 0x3FF;
        uint32_t f32;
        if (e == 0) {
            if (m == 0) {
                f32 = s << 31;
            } else {
                // subnormal
                while ((m & 0x400) == 0) { m <<= 1; e -= 1; }
                e += 1;
                m &= 0x3FF;
                f32 = (s << 31) | ((e + 112) << 23) | (m << 13);
            }
        } else if (e == 31) {
            f32 = (s << 31) | 0x7F800000 | (m << 13);
        } else {
            f32 = (s << 31) | ((e + 112) << 23) | (m << 13);
        }
        float fp32;
        std::memcpy(&fp32, &f32, 4);
        dst[i] = fp32;
    }
}

static void test_materialize_empty() {
    std::printf("test_materialize_empty\n");
    tiered_cache_config cfg;
    cfg.n_layers   = 1;
    cfg.n_kv_heads = 1;
    cfg.head_dim   = 8;
    cfg.hot_window = 4;
    tiered_cache cache(cfg);

    std::vector<float> k, v;
    make_token(0, cfg, k, v);
    cache.add_token(k.data(), v.data());

    std::vector<uint16_t> out;
    cache.materialize_fp16_rows(0, 0, false, out);
    CHECK(out.empty(), "n_positions=0 gives empty output for K");
    cache.materialize_fp16_rows(0, 0, true,  out);
    CHECK(out.empty(), "n_positions=0 gives empty output for V");
}

static void test_materialize_negative_n_throws() {
    std::printf("test_materialize_negative_n_throws\n");
    tiered_cache_config cfg;
    cfg.n_layers   = 1;
    cfg.n_kv_heads = 1;
    cfg.head_dim   = 8;
    cfg.hot_window = 4;
    tiered_cache cache(cfg);
    std::vector<float> k, v;
    make_token(0, cfg, k, v);
    cache.add_token(k.data(), v.data());

    std::vector<uint16_t> out;
    bool threw = false;
    try { cache.materialize_fp16_rows(0, -1, false, out); }
    catch (const std::out_of_range &) { threw = true; }
    CHECK(threw, "negative n_positions throws");
}

static void test_materialize_out_of_range_throws() {
    std::printf("test_materialize_out_of_range_throws\n");
    tiered_cache_config cfg;
    cfg.n_layers   = 1;
    cfg.n_kv_heads = 1;
    cfg.head_dim   = 8;
    cfg.hot_window = 4;
    tiered_cache cache(cfg);
    std::vector<float> k, v;
    for (int t = 0; t < 3; ++t) {
        make_token(t, cfg, k, v);
        cache.add_token(k.data(), v.data());
    }

    std::vector<uint16_t> out;
    bool threw;

    threw = false;
    try { cache.materialize_fp16_rows(-1, 2, false, out); }
    catch (const std::out_of_range &) { threw = true; }
    CHECK(threw, "negative start_pos throws");

    threw = false;
    try { cache.materialize_fp16_rows(2, 5, false, out); }
    catch (const std::out_of_range &) { threw = true; }
    CHECK(threw, "range overrunning n_tokens() throws");
}

static void test_materialize_all_hot() {
    std::printf("test_materialize_all_hot\n");
    tiered_cache_config cfg;
    cfg.n_layers   = 1;
    cfg.n_kv_heads = 2;
    cfg.head_dim   = 8;
    cfg.hot_window = 8;
    tiered_cache cache(cfg);

    std::vector<std::vector<float>> all_k(4), all_v(4);
    for (int t = 0; t < 4; ++t) {
        make_token(t, cfg, all_k[t], all_v[t]);
        cache.add_token(all_k[t].data(), all_v[t].data());
    }

    std::vector<uint16_t> out;
    cache.materialize_fp16_rows(0, 4, false, out);
    const int n_row = cfg.n_kv_heads * cfg.head_dim;
    CHECK((int) out.size() == 4 * n_row, "output row count * width");

    // Convert fp16 back to fp32 and compare — allow tiny rounding since
    // fp16 has ~3 decimal digits of precision.
    std::vector<float> recon;
    for (int t = 0; t < 4; ++t) {
        fp16_row_to_fp32(out.data() + (size_t) t * n_row, n_row, recon);
        float c = cosine_similarity(recon.data(), all_k[t].data(), n_row);
        CHECK(c > 0.999f, "hot materialize: cos ≈ 1.0 (fp16 rounding only)");
    }

    // Same for V
    cache.materialize_fp16_rows(0, 4, true, out);
    for (int t = 0; t < 4; ++t) {
        fp16_row_to_fp32(out.data() + (size_t) t * n_row, n_row, recon);
        float c = cosine_similarity(recon.data(), all_v[t].data(), n_row);
        CHECK(c > 0.999f, "hot materialize V: cos ≈ 1.0");
    }
}

static void test_materialize_all_cold() {
    std::printf("test_materialize_all_cold\n");
    tiered_cache_config cfg;
    cfg.n_layers   = 1;
    cfg.n_kv_heads = 2;
    cfg.head_dim   = 64;
    cfg.hot_window = 2;
    cfg.cold_bits  = BITS_3;
    tiered_cache cache(cfg);

    const int n_tok = 10;
    std::vector<std::vector<float>> all_k(n_tok), all_v(n_tok);
    for (int t = 0; t < n_tok; ++t) {
        make_token(t, cfg, all_k[t], all_v[t]);
        cache.add_token(all_k[t].data(), all_v[t].data());
    }

    int n_cold = cache.n_cold_tokens();
    std::vector<uint16_t> out;
    cache.materialize_fp16_rows(0, n_cold, false, out);

    const int n_row = cfg.n_kv_heads * cfg.head_dim;
    std::vector<float> recon;
    double tot = 0;
    for (int t = 0; t < n_cold; ++t) {
        fp16_row_to_fp32(out.data() + (size_t) t * n_row, n_row, recon);
        tot += cosine_similarity(recon.data(), all_k[t].data(), n_row);
    }
    double mean = tot / n_cold;
    std::printf("  all-cold materialize K cosine = %.4f\n", mean);
    CHECK(mean > 0.93, "all-cold materialize cosine > 0.93 at 3-bit");
}

static void test_materialize_mixed_boundary() {
    std::printf("test_materialize_mixed_boundary\n");
    tiered_cache_config cfg;
    cfg.n_layers   = 1;
    cfg.n_kv_heads = 1;
    cfg.head_dim   = 32;
    cfg.hot_window = 4;
    cfg.cold_bits  = BITS_4;   // higher fidelity so cold tail near-exact
    tiered_cache cache(cfg);

    const int n_tok = 10;  // last 4 hot, first 6 cold
    std::vector<std::vector<float>> all_k(n_tok), all_v(n_tok);
    for (int t = 0; t < n_tok; ++t) {
        make_token(t, cfg, all_k[t], all_v[t]);
        cache.add_token(all_k[t].data(), all_v[t].data());
    }
    CHECK(cache.n_cold_tokens() == 6, "6 cold");
    CHECK(cache.n_hot_tokens()  == 4, "4 hot");

    // Materialize across boundary: [3, 3 + 5) = positions 3..7 (3 cold, 2 hot)
    std::vector<uint16_t> out;
    cache.materialize_fp16_rows(3, 5, false, out);

    const int n_row = cfg.n_kv_heads * cfg.head_dim;
    std::vector<float> recon;

    // Positions 3, 4, 5 are still cold (cosine > ~0.95 at 4-bit)
    for (int t = 0; t < 3; ++t) {
        fp16_row_to_fp32(out.data() + (size_t) t * n_row, n_row, recon);
        float c = cosine_similarity(recon.data(), all_k[3 + t].data(), n_row);
        CHECK(c > 0.93f, "mixed-range cold slice cosine > 0.93");
    }
    // Positions 6, 7 are hot → near-exact
    for (int t = 3; t < 5; ++t) {
        fp16_row_to_fp32(out.data() + (size_t) t * n_row, n_row, recon);
        float c = cosine_similarity(recon.data(), all_k[3 + t].data(), n_row);
        CHECK(c > 0.999f, "mixed-range hot slice cosine ≈ 1.0");
    }
}

static void test_materialize_all_bit_widths() {
    std::printf("test_materialize_all_bit_widths\n");
    for (llama_kv_tq::bits b : {BITS_2, BITS_3, BITS_4}) {
        tiered_cache_config cfg;
        cfg.n_layers   = 1;
        cfg.n_kv_heads = 1;
        cfg.head_dim   = 128;
        cfg.hot_window = 2;
        cfg.cold_bits  = b;
        tiered_cache cache(cfg);

        const int n_tok = 12;
        std::vector<std::vector<float>> all_k(n_tok), all_v(n_tok);
        for (int t = 0; t < n_tok; ++t) {
            make_token(t, cfg, all_k[t], all_v[t]);
            cache.add_token(all_k[t].data(), all_v[t].data());
        }
        int n_cold = cache.n_cold_tokens();

        std::vector<uint16_t> out;
        cache.materialize_fp16_rows(0, n_cold, false, out);

        const int n_row = cfg.head_dim;
        std::vector<float> recon;
        double tot = 0;
        for (int t = 0; t < n_cold; ++t) {
            fp16_row_to_fp32(out.data() + (size_t) t * n_row, n_row, recon);
            tot += cosine_similarity(recon.data(), all_k[t].data(), n_row);
        }
        double mean = tot / n_cold;
        // Thresholds mirror the round-trip targets in test-tq-kv.
        double min_cos = (b == BITS_2) ? 0.82
                       : (b == BITS_3) ? 0.92
                       :                 0.97;
        std::printf("  bits=%d mean cos = %.4f (need > %.2f)\n",
            (int)b, mean, min_cos);
        CHECK(mean > min_cos, "materialize cosine meets per-bit target");
    }
}

// -------------------------------------------------------------------- //
// 10. Push-readback round-trip semantics (Sprint 4c step 3c-1)         //
//                                                                       //
// The runtime validation in llama_kv_cache relies on exactly this      //
// invariant: the last-added token can be read back from tiered_cache   //
// via read_token_k/v and the cosine against what we pushed in is 1.0  //
// for hot tokens and meets the per-bit threshold for cold tokens.     //
// -------------------------------------------------------------------- //

static void test_push_readback_last_token_hot() {
    std::printf("test_push_readback_last_token_hot\n");
    tiered_cache_config cfg;
    cfg.n_layers   = 1;
    cfg.n_kv_heads = 2;
    cfg.head_dim   = 32;
    cfg.hot_window = 8;  // big enough that no eviction occurs
    cfg.cold_bits  = BITS_3;
    tiered_cache cache(cfg);

    std::vector<float> push_buf, readback_buf;
    for (int t = 0; t < 4; ++t) {
        make_token(t, cfg, push_buf, /*v=*/readback_buf);
        // Use same buf for K and V here — the readback test is per-side.
        cache.add_token(push_buf.data(), push_buf.data());
        const int last_pos = cache.n_tokens() - 1;
        cache.read_token_k(last_pos, readback_buf);
        const int n = cfg.n_kv_heads * cfg.head_dim;
        float c = cosine_similarity(
            push_buf.data(), readback_buf.data(), n);
        CHECK(c >= 0.99999f,
            "hot push-readback: cos = 1.0 up to fp32 storage rounding");
    }
}

static void test_push_readback_last_token_cold_path() {
    std::printf("test_push_readback_last_token_cold_path\n");
    tiered_cache_config cfg;
    cfg.n_layers   = 1;
    cfg.n_kv_heads = 1;
    cfg.head_dim   = 128;
    cfg.hot_window = 2;
    cfg.cold_bits  = BITS_3;
    tiered_cache cache(cfg);

    // After adding > hot_window tokens, the most recently added still
    // sits in hot (exact), but older ones evicted to cold will show
    // compression loss if we happen to read those instead.
    std::vector<float> push_buf, rb_k;
    std::vector<std::vector<float>> all;
    for (int t = 0; t < 8; ++t) {
        make_token(t, cfg, push_buf, rb_k);
        cache.add_token(push_buf.data(), push_buf.data());
        all.push_back(push_buf);
    }
    CHECK(cache.n_cold_tokens() == 6, "6 cold");
    CHECK(cache.n_hot_tokens()  == 2, "2 hot");

    // Read back the last-added (hot): must be exact.
    cache.read_token_k(cache.n_tokens() - 1, rb_k);
    float c_hot = cosine_similarity(
        all.back().data(), rb_k.data(), cfg.head_dim);
    CHECK(c_hot >= 0.99999f, "hot readback still exact after cold evictions");

    // Read back oldest (cold): should meet 3-bit threshold.
    cache.read_token_k(0, rb_k);
    float c_cold = cosine_similarity(
        all[0].data(), rb_k.data(), cfg.head_dim);
    std::printf("  cold readback cos = %.4f\n", c_cold);
    CHECK(c_cold > 0.93f, "cold readback above 3-bit threshold");
}

static void test_push_readback_independent_k_and_v() {
    // The runtime validation pushes the same buf for K and V (since
    // our per-layer tiered_cache stores both sides with the same
    // value — one side is wasted). This test pins that behaviour so a
    // future optimization doesn't silently break the runtime check.
    std::printf("test_push_readback_independent_k_and_v\n");
    tiered_cache_config cfg;
    cfg.n_layers   = 1;
    cfg.n_kv_heads = 1;
    cfg.head_dim   = 16;
    cfg.hot_window = 4;
    tiered_cache cache(cfg);

    std::vector<float> k_side(cfg.head_dim), v_side(cfg.head_dim);
    for (int i = 0; i < cfg.head_dim; ++i) {
        k_side[i] =  (float) i;
        v_side[i] = -(float) i;
    }
    cache.add_token(k_side.data(), v_side.data());
    const int last = cache.n_tokens() - 1;

    std::vector<float> rb;
    cache.read_token_k(last, rb);
    for (int i = 0; i < cfg.head_dim; ++i) {
        CHECK_NEAR(rb[i], k_side[i], 1e-6f, "K readback matches K push");
    }
    cache.read_token_v(last, rb);
    for (int i = 0; i < cfg.head_dim; ++i) {
        CHECK_NEAR(rb[i], v_side[i], 1e-6f, "V readback matches V push");
    }
}

// -------------------------------------------------------------------- //
// Entry point                                                           //
// -------------------------------------------------------------------- //

int main() {
    test_hot_only_exact();
    test_cold_reconstruction();
    test_eviction_order();
    test_memory_stats();
    test_slot_isolation();
    test_stress();
    test_out_of_range();
    // Sprint 4c step 3b: read methods + materialize
    test_read_token_hot_exact();
    test_read_token_cold_cosine();
    test_read_token_out_of_range();
    test_materialize_empty();
    test_materialize_negative_n_throws();
    test_materialize_out_of_range_throws();
    test_materialize_all_hot();
    test_materialize_all_cold();
    test_materialize_mixed_boundary();
    test_materialize_all_bit_widths();
    // Sprint 4c step 3c-1: round-trip semantics used by runtime validation
    test_push_readback_last_token_hot();
    test_push_readback_last_token_cold_path();
    test_push_readback_independent_k_and_v();

    std::printf("\n=== %d / %d checks passed ===\n",
        n_total - n_failed, n_total);
    return n_failed == 0 ? 0 : 1;
}
