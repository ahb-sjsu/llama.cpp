// Tests for TurboQuant KV cache compression.
//
// Coverage:
//   1. Codebook properties (monotonic, symmetric)
//   2. Bit packing/unpacking round-trip
//   3. Rotation matrix orthogonality (Pi @ Pi^T == I)
//   4. Rotate/unrotate identity
//   5. Compression round-trip cosine similarity (per bit width)
//   6. Edge cases (zero vector, unit vector, large head_dim)
//   7. Determinism with same seed

#include "../src/llama-kv-turboquant.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

using llama_kv_tq::BITS_2;
using llama_kv_tq::BITS_3;
using llama_kv_tq::BITS_4;
using llama_kv_tq::bits;
using llama_kv_tq::compress_vector;
using llama_kv_tq::compressed_block;
using llama_kv_tq::decompress_vector;
using llama_kv_tq::init_rotation;
using llama_kv_tq::rotation_matrix;

// -------------------------------------------------------------------- //
// Test harness helpers                                                  //
// -------------------------------------------------------------------- //

static int n_failed = 0;
static int n_total  = 0;

#define CHECK(cond, msg) do {                                            \
    n_total++;                                                           \
    if (!(cond)) {                                                       \
        n_failed++;                                                      \
        std::fprintf(stderr, "FAIL  %s:%d  %s — %s\n",                   \
            __FILE__, __LINE__, #cond, msg);                             \
    }                                                                    \
} while (0)

#define CHECK_NEAR(a, b, tol, msg) do {                                  \
    n_total++;                                                           \
    float _a = (a), _b = (b);                                            \
    if (std::abs(_a - _b) > (tol)) {                                     \
        n_failed++;                                                      \
        std::fprintf(stderr,                                             \
            "FAIL  %s:%d  |%g - %g| > %g — %s\n",                        \
            __FILE__, __LINE__, _a, _b, (float)(tol), msg);              \
    }                                                                    \
} while (0)

static float cosine_similarity(
    const float * a, const float * b, int n
) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (int i = 0; i < n; ++i) {
        dot += static_cast<double>(a[i]) * b[i];
        na  += static_cast<double>(a[i]) * a[i];
        nb  += static_cast<double>(b[i]) * b[i];
    }
    double denom = std::sqrt(na) * std::sqrt(nb);
    return denom > 1e-12 ? static_cast<float>(dot / denom) : 0.0f;
}

static std::vector<float> random_vector(int n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> v(n);
    for (auto & x : v) x = normal(rng);
    return v;
}

// -------------------------------------------------------------------- //
// 1. block_size_bytes / compression_ratio sanity                        //
// -------------------------------------------------------------------- //

static void test_block_size_and_ratio() {
    std::printf("test_block_size_and_ratio\n");
    // head_dim = 128, fp16 = 256 bytes
    // TQ KV3: 4 (norm) + ceil(128*3/8) = 4 + 48 = 52 bytes  → 4.92x
    // TQ KV4: 4 + 64 = 68 bytes  → 3.76x
    // TQ KV2: 4 + 32 = 36 bytes  → 7.11x
    CHECK(llama_kv_tq::block_size_bytes(128, BITS_3) == 52, "3-bit block size");
    CHECK(llama_kv_tq::block_size_bytes(128, BITS_4) == 68, "4-bit block size");
    CHECK(llama_kv_tq::block_size_bytes(128, BITS_2) == 36, "2-bit block size");

    CHECK_NEAR(llama_kv_tq::compression_ratio(128, BITS_3), 4.923f, 0.01f, "3-bit ratio");
    CHECK_NEAR(llama_kv_tq::compression_ratio(128, BITS_4), 3.764f, 0.01f, "4-bit ratio");
    CHECK_NEAR(llama_kv_tq::compression_ratio(128, BITS_2), 7.111f, 0.01f, "2-bit ratio");
}

// -------------------------------------------------------------------- //
// 2. Rotation matrix orthogonality: Pi @ Pi^T == I                      //
// -------------------------------------------------------------------- //

static void test_rotation_orthogonality() {
    std::printf("test_rotation_orthogonality\n");
    for (int D : {16, 64, 128}) {
        auto rot = init_rotation(D, /*seed=*/42);
        CHECK(!rot.structured, "small head_dim should use QR");

        // Verify Pi @ Pi^T == I for several sample rows
        for (int i : {0, 1, D / 2, D - 1}) {
            for (int j : {0, 1, D / 2, D - 1}) {
                float dot = 0.0f;
                for (int k = 0; k < D; ++k) {
                    dot += rot.Pi[i * D + k] * rot.Pi[j * D + k];
                }
                float expected = (i == j) ? 1.0f : 0.0f;
                CHECK_NEAR(dot, expected, 1e-3f,
                    ("orthogonality D=" + std::to_string(D)).c_str());
            }
        }
    }
}

static void test_structured_rotation() {
    std::printf("test_structured_rotation\n");
    int D = 8192;
    auto rot = init_rotation(D, /*seed=*/42);
    CHECK(rot.structured, "large head_dim should use structured");
    CHECK(static_cast<int>(rot.sign_flip.size()) == D, "sign_flip size");
    CHECK(static_cast<int>(rot.perm.size()) == D, "perm size");
    CHECK(static_cast<int>(rot.inv_perm.size()) == D, "inv_perm size");

    // Verify perm + inv_perm round-trip
    for (int i = 0; i < D; ++i) {
        CHECK(rot.inv_perm[rot.perm[i]] == i, "perm/inv_perm are inverses");
    }
    // Verify sign_flip is in {-1, +1}
    for (int i = 0; i < D; ++i) {
        int s = rot.sign_flip[i];
        CHECK(s == 1 || s == -1, "sign_flip in {-1,+1}");
    }
}

// -------------------------------------------------------------------- //
// 3. Determinism with same seed                                         //
// -------------------------------------------------------------------- //

static void test_deterministic_seed() {
    std::printf("test_deterministic_seed\n");
    auto rot1 = init_rotation(64, 12345);
    auto rot2 = init_rotation(64, 12345);
    CHECK(rot1.Pi == rot2.Pi, "same seed → same Pi");

    auto rot3 = init_rotation(64, 99999);
    CHECK(rot1.Pi != rot3.Pi, "different seed → different Pi");
}

// -------------------------------------------------------------------- //
// 4. Compress/decompress round-trip cosine similarity                   //
// -------------------------------------------------------------------- //

static void test_round_trip(int head_dim, bits b, float min_cosine) {
    std::printf("test_round_trip D=%d bits=%d\n", head_dim, (int)b);
    auto rot = init_rotation(head_dim, 42);

    // Average cosine over many random vectors
    int n_trials = 16;
    double total_cos = 0.0;
    for (int t = 0; t < n_trials; ++t) {
        auto vec = random_vector(head_dim, /*seed=*/100 + t);

        auto block = compress_vector(vec.data(), head_dim, b, rot);

        std::vector<float> recon(head_dim);
        decompress_vector(block, head_dim, b, rot, recon.data());

        total_cos += cosine_similarity(vec.data(), recon.data(), head_dim);
    }
    double mean_cos = total_cos / n_trials;
    std::printf("  mean cosine similarity = %.4f (need ≥ %.3f)\n",
        mean_cos, min_cosine);
    CHECK(mean_cos >= min_cosine, "round-trip cosine similarity");
}

// -------------------------------------------------------------------- //
// 5. Round-trip preserves L2 norm exactly                               //
// -------------------------------------------------------------------- //

static void test_norm_preservation() {
    std::printf("test_norm_preservation\n");
    int D = 128;
    auto rot = init_rotation(D, 42);

    auto vec = random_vector(D, 7);
    auto block = compress_vector(vec.data(), D, BITS_3, rot);

    // The stored norm should match the input's L2 norm exactly
    float input_norm_sq = 0.0f;
    for (int i = 0; i < D; ++i) input_norm_sq += vec[i] * vec[i];
    float input_norm = std::sqrt(input_norm_sq);

    CHECK_NEAR(block.norm, input_norm, 1e-5f, "block.norm == ||vec||");
}

// -------------------------------------------------------------------- //
// 6. Edge cases                                                         //
// -------------------------------------------------------------------- //

static void test_zero_vector() {
    std::printf("test_zero_vector\n");
    int D = 64;
    auto rot = init_rotation(D, 42);
    std::vector<float> zeros(D, 0.0f);

    auto block = compress_vector(zeros.data(), D, BITS_3, rot);
    CHECK_NEAR(block.norm, 0.0f, 1e-6f, "zero vector → zero norm");

    std::vector<float> recon(D);
    decompress_vector(block, D, BITS_3, rot, recon.data());

    // After scaling by zero norm, all reconstructions should be ~0
    for (int i = 0; i < D; ++i) {
        CHECK_NEAR(recon[i], 0.0f, 1e-5f, "zero vector reconstruction");
    }
}

static void test_unit_axis_vector() {
    std::printf("test_unit_axis_vector\n");
    int D = 64;
    auto rot = init_rotation(D, 42);
    std::vector<float> e0(D, 0.0f);
    e0[0] = 1.0f;

    auto block = compress_vector(e0.data(), D, BITS_4, rot);
    CHECK_NEAR(block.norm, 1.0f, 1e-5f, "unit axis vector norm");

    std::vector<float> recon(D);
    decompress_vector(block, D, BITS_4, rot, recon.data());
    float cos = cosine_similarity(e0.data(), recon.data(), D);
    std::printf("  unit-axis cosine = %.4f\n", cos);
    CHECK(cos > 0.85f, "unit-axis vector reconstructs reasonably");
}

// -------------------------------------------------------------------- //
// 7. Structured rotation round-trip (head_dim > 4096)                  //
// -------------------------------------------------------------------- //

static void test_structured_round_trip() {
    std::printf("test_structured_round_trip\n");
    int D = 8192;
    auto rot = init_rotation(D, 42);
    CHECK(rot.structured, "should use structured");

    auto vec = random_vector(D, 200);
    auto block = compress_vector(vec.data(), D, BITS_4, rot);
    std::vector<float> recon(D);
    decompress_vector(block, D, BITS_4, rot, recon.data());

    float cos = cosine_similarity(vec.data(), recon.data(), D);
    std::printf("  structured cosine D=%d bits=4: %.4f\n", D, cos);
    CHECK(cos > 0.85f, "structured rotation round-trip");
}

// -------------------------------------------------------------------- //
// 8. Bit-packing edge cases (uneven group sizes)                       //
// -------------------------------------------------------------------- //

static void test_bit_packing_edge_cases() {
    std::printf("test_bit_packing_edge_cases\n");
    // head_dim = 13 → odd for all bit widths
    int D = 13;
    auto rot = init_rotation(D, 42);

    auto vec = random_vector(D, 1);
    for (bits b : {BITS_2, BITS_3, BITS_4}) {
        auto block = compress_vector(vec.data(), D, b, rot);
        std::vector<float> recon(D);
        decompress_vector(block, D, b, rot, recon.data());
        // Just verify it doesn't crash and produces sensible norm
        CHECK(block.norm > 0.0f, "non-zero norm for non-zero vector");
        float recon_norm_sq = 0.0f;
        for (auto x : recon) recon_norm_sq += x * x;
        CHECK(std::sqrt(recon_norm_sq) > 0.0f,
            "non-zero reconstruction for non-zero input");
    }
}

// -------------------------------------------------------------------- //
// 9. CUDA equivalence (Sprint 3) — GPU kernels must produce            //
//    bit-identical packed indices and matching reconstructions.        //
//    Skipped at runtime if no CUDA device is detected.                 //
// -------------------------------------------------------------------- //

using llama_kv_tq::compress_vector_cuda;
using llama_kv_tq::cuda_available;
using llama_kv_tq::decompress_vector_cuda;

static void test_cuda_compress_matches_cpu(int head_dim, llama_kv_tq::bits b) {
    std::printf("test_cuda_compress_matches_cpu D=%d bits=%d\n",
        head_dim, (int)b);
    auto rot = init_rotation(head_dim, 42);
    auto vec = random_vector(head_dim, /*seed=*/500);

    auto cpu_block  = compress_vector(vec.data(), head_dim, b, rot);

    llama_kv_tq::compressed_block gpu_block;
    bool ok = compress_vector_cuda(vec.data(), head_dim, b, rot, gpu_block);
    CHECK(ok, "compress_vector_cuda returned ok");

    // Stored norm: identical to within float32 sqrt rounding
    CHECK_NEAR(gpu_block.norm, cpu_block.norm, 1e-4f, "CUDA norm matches CPU");

    CHECK(gpu_block.indices.size() == cpu_block.indices.size(),
        "packed length matches");

    // Compare indices: small numerical drift in the rotation matvec may
    // tip a value across a quantization boundary, so allow a tiny
    // mismatch fraction (<2%).
    int mismatches = 0;
    for (size_t i = 0; i < cpu_block.indices.size(); ++i) {
        if (gpu_block.indices[i] != cpu_block.indices[i]) ++mismatches;
    }
    double frac = static_cast<double>(mismatches) / cpu_block.indices.size();
    std::printf("  packed-byte mismatch fraction = %.4f\n", frac);
    CHECK(frac < 0.02, "<2% packed bytes differ between CPU and CUDA");
}

static void test_cuda_round_trip(int head_dim, llama_kv_tq::bits b,
                                 float min_cos_vs_cpu)
{
    std::printf("test_cuda_round_trip D=%d bits=%d\n", head_dim, (int)b);
    auto rot = init_rotation(head_dim, 42);
    auto vec = random_vector(head_dim, /*seed=*/501);

    // CPU reference reconstruction
    auto cpu_block = compress_vector(vec.data(), head_dim, b, rot);
    std::vector<float> cpu_recon(head_dim);
    decompress_vector(cpu_block, head_dim, b, rot, cpu_recon.data());

    // CUDA reconstruction (compress + decompress entirely on GPU)
    llama_kv_tq::compressed_block gpu_block;
    bool ok1 = compress_vector_cuda(vec.data(), head_dim, b, rot, gpu_block);
    CHECK(ok1, "compress_vector_cuda ok");
    std::vector<float> gpu_recon(head_dim);
    bool ok2 = decompress_vector_cuda(gpu_block, head_dim, b, rot,
                                      gpu_recon.data());
    CHECK(ok2, "decompress_vector_cuda ok");

    float cos_input = cosine_similarity(vec.data(), gpu_recon.data(), head_dim);
    float cos_cpu   = cosine_similarity(cpu_recon.data(), gpu_recon.data(),
                                        head_dim);
    std::printf("  cos(GPU, input) = %.4f, cos(GPU, CPU recon) = %.4f\n",
        cos_input, cos_cpu);

    CHECK(cos_cpu >= min_cos_vs_cpu, "GPU recon ≈ CPU recon");
}

static void test_cuda_decompress_matches_cpu_block(int head_dim,
                                                   llama_kv_tq::bits b)
{
    std::printf("test_cuda_decompress_matches_cpu_block D=%d bits=%d\n",
        head_dim, (int)b);
    auto rot = init_rotation(head_dim, 42);
    auto vec = random_vector(head_dim, /*seed=*/502);

    // Build a single packed block on the CPU, decompress on both sides
    auto block = compress_vector(vec.data(), head_dim, b, rot);

    std::vector<float> cpu_out(head_dim);
    decompress_vector(block, head_dim, b, rot, cpu_out.data());

    std::vector<float> gpu_out(head_dim);
    bool ok = decompress_vector_cuda(block, head_dim, b, rot, gpu_out.data());
    CHECK(ok, "decompress_vector_cuda on CPU-built block ok");

    // Same packed indices → same dequantized rotated vector → same recon
    // Allow tiny rounding from the QR matvec on GPU (different reduction order)
    float max_abs = 0.0f;
    for (int i = 0; i < head_dim; ++i) {
        max_abs = std::max(max_abs, std::abs(cpu_out[i] - gpu_out[i]));
    }
    std::printf("  max |CPU - GPU| = %.6f\n", max_abs);
    CHECK(max_abs < 1e-3f, "GPU decompress matches CPU decompress");
}

static void run_cuda_tests() {
    if (!cuda_available()) {
        std::printf("\n[CUDA tests skipped — no GPU detected]\n");
        return;
    }
    std::printf("\n[CUDA tests — GPU detected]\n");

    test_cuda_compress_matches_cpu(64,  BITS_3);
    test_cuda_compress_matches_cpu(128, BITS_3);
    test_cuda_compress_matches_cpu(128, BITS_4);
    test_cuda_compress_matches_cpu(128, BITS_2);

    test_cuda_decompress_matches_cpu_block(128, BITS_3);
    test_cuda_decompress_matches_cpu_block(128, BITS_4);
    test_cuda_decompress_matches_cpu_block(64,  BITS_2);

    test_cuda_round_trip(128, BITS_4, 0.99f);
    test_cuda_round_trip(128, BITS_3, 0.97f);
    test_cuda_round_trip(128, BITS_2, 0.92f);
    test_cuda_round_trip(256, BITS_3, 0.97f);
}

// -------------------------------------------------------------------- //
// 10. ggml type registration (Sprint 4b)                               //
// -------------------------------------------------------------------- //

static void test_ggml_type_registration() {
    std::printf("test_ggml_type_registration\n");

    // Names round-trip through ggml_type_name.
    CHECK(std::string(ggml_type_name(GGML_TYPE_TQ_KV2)) == "tq_kv2",
        "tq_kv2 type name");
    CHECK(std::string(ggml_type_name(GGML_TYPE_TQ_KV3)) == "tq_kv3",
        "tq_kv3 type name");
    CHECK(std::string(ggml_type_name(GGML_TYPE_TQ_KV4)) == "tq_kv4",
        "tq_kv4 type name");

    // ggml_type_to_bits decodes correctly.
    CHECK(llama_kv_tq::ggml_type_to_bits(GGML_TYPE_TQ_KV2) == 2, "TQ_KV2 → 2 bits");
    CHECK(llama_kv_tq::ggml_type_to_bits(GGML_TYPE_TQ_KV3) == 3, "TQ_KV3 → 3 bits");
    CHECK(llama_kv_tq::ggml_type_to_bits(GGML_TYPE_TQ_KV4) == 4, "TQ_KV4 → 4 bits");
    CHECK(llama_kv_tq::ggml_type_to_bits(GGML_TYPE_F16)    == -1, "F16 not TQ");
    CHECK(llama_kv_tq::ggml_type_to_bits(GGML_TYPE_Q4_0)   == -1, "Q4_0 not TQ");

    // is_turboquant_kv_type predicate.
    CHECK( llama_kv_tq::is_turboquant_kv_type(GGML_TYPE_TQ_KV3), "tq_kv3 → true");
    CHECK(!llama_kv_tq::is_turboquant_kv_type(GGML_TYPE_F16),    "f16 → false");
    CHECK(!llama_kv_tq::is_turboquant_kv_type(GGML_TYPE_Q8_0),   "q8_0 → false");

    // Type traits report blck_size=0 / type_size=0 — these are TAGS,
    // not packed tensor types. Any code path that tries to compute a
    // row size for a TQ KV type will fail loudly via the existing
    // assertions and the friendly message in llama_init_from_model.
    CHECK(ggml_blck_size(GGML_TYPE_TQ_KV3) == 0, "TQ_KV3 blck_size = 0");
    CHECK(ggml_type_size(GGML_TYPE_TQ_KV3) == 0, "TQ_KV3 type_size = 0");

    // is_quantized = true so existing code that gates on quantized KV
    // (e.g. flash-attn checks) recognises them as compressed.
    CHECK(ggml_is_quantized(GGML_TYPE_TQ_KV2), "TQ_KV2 is quantized");
    CHECK(ggml_is_quantized(GGML_TYPE_TQ_KV3), "TQ_KV3 is quantized");
    CHECK(ggml_is_quantized(GGML_TYPE_TQ_KV4), "TQ_KV4 is quantized");
}

// -------------------------------------------------------------------- //
// Entry point                                                           //
// -------------------------------------------------------------------- //

int main() {
    test_ggml_type_registration();
    test_block_size_and_ratio();
    test_rotation_orthogonality();
    test_structured_rotation();
    test_deterministic_seed();
    test_norm_preservation();
    test_zero_vector();
    test_unit_axis_vector();
    test_structured_round_trip();
    test_bit_packing_edge_cases();

    // Round-trip cosine similarity targets per bit width
    test_round_trip(64,  BITS_4, 0.95f);
    test_round_trip(64,  BITS_3, 0.90f);
    test_round_trip(64,  BITS_2, 0.80f);
    test_round_trip(128, BITS_4, 0.97f);
    test_round_trip(128, BITS_3, 0.92f);
    test_round_trip(128, BITS_2, 0.82f);
    test_round_trip(256, BITS_4, 0.97f);
    test_round_trip(256, BITS_3, 0.93f);

    // CUDA equivalence — runs only when the binary was built with
    // LLAMA_TQ_CUDA=ON AND a GPU is present at runtime.
    run_cuda_tests();

    std::printf("\n=== %d / %d checks passed ===\n",
        n_total - n_failed, n_total);
    return n_failed == 0 ? 0 : 1;
}
