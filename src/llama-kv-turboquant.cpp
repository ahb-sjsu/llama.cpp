// TurboQuant KV cache compression — CPU reference implementation (Sprint 2)
//
// Ports the rotation, quantization, and bit-packing from
// turboquant-pro Python implementation. CPU-only, no SIMD.
// Sprint 3 will add CUDA kernels.
//
// See: docs/turboquant-kv-design.md
// Reference: turboquant_pro/_kv_cache.py and
//            ahb-sjsu/agi-hpc src/agi/meta/llm/turboquant_kv.py

#include "llama-kv-turboquant.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>

namespace llama_kv_tq {

// -------------------------------------------------------------------- //
// Lloyd-Max codebook centroids for standard normal N(0,1)              //
// (scaled by 1/sqrt(d) at runtime, where d = head_dim)                 //
// -------------------------------------------------------------------- //

static const std::vector<float> CODEBOOK_2 = {
    -1.510f, -0.453f, 0.453f, 1.510f
};

static const std::vector<float> CODEBOOK_3 = {
    -1.748f, -1.050f, -0.500f, -0.069f,
    0.069f, 0.500f, 1.050f, 1.748f
};

static const std::vector<float> CODEBOOK_4 = {
    -2.401f, -1.844f, -1.437f, -1.099f,
    -0.800f, -0.524f, -0.262f, -0.066f,
    0.066f, 0.262f, 0.524f, 0.800f,
    1.099f, 1.437f, 1.844f, 2.401f
};

static const std::vector<float> & get_codebook(bits b) {
    switch (b) {
        case BITS_2: return CODEBOOK_2;
        case BITS_3: return CODEBOOK_3;
        case BITS_4: return CODEBOOK_4;
    }
    throw std::runtime_error("invalid bits");
}

// Get scaled codebook for a given head_dim.
// Centroids are stored for N(0,1); for N(0, 1/sqrt(d)) we multiply by 1/sqrt(d).
static std::vector<float> scaled_centroids(bits b, int head_dim) {
    const auto & raw = get_codebook(b);
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    std::vector<float> out(raw.size());
    for (std::size_t i = 0; i < raw.size(); ++i) {
        out[i] = raw[i] * scale;
    }
    return out;
}

// Boundaries between centroids (midpoints) for searchsorted-style quantization.
static std::vector<float> boundaries_from_centroids(const std::vector<float> & c) {
    std::vector<float> b(c.size() - 1);
    for (std::size_t i = 0; i < b.size(); ++i) {
        b[i] = 0.5f * (c[i] + c[i + 1]);
    }
    return b;
}

// -------------------------------------------------------------------- //
// QR decomposition via Householder reflections                          //
// (numpy reference: np.linalg.qr; here: classic Householder)           //
// -------------------------------------------------------------------- //

// QR decomp of square matrix A (D x D, row-major). Writes Q to out_Q.
// We only need Q (R is discarded since the input is random Gaussian).
// Standard Householder; not the fastest but correct.
static void householder_qr(
    std::vector<float> & A,    // (D, D), modified in place to become R
    std::vector<float> & Q,    // (D, D), output
    int                  D
) {
    Q.assign(static_cast<std::size_t>(D) * D, 0.0f);
    for (int i = 0; i < D; ++i) {
        Q[static_cast<std::size_t>(i) * D + i] = 1.0f;  // Q = I
    }

    std::vector<float> v(D);

    for (int k = 0; k < D - 1; ++k) {
        // Compute Householder vector v for column k of A[k:, k]
        float norm_x_sq = 0.0f;
        for (int i = k; i < D; ++i) {
            float a = A[static_cast<std::size_t>(i) * D + k];
            norm_x_sq += a * a;
        }
        float norm_x = std::sqrt(norm_x_sq);
        if (norm_x < 1e-12f) {
            continue;  // skip degenerate column
        }

        float alpha =
            -std::copysign(norm_x, A[static_cast<std::size_t>(k) * D + k]);

        for (int i = 0; i < D; ++i) v[i] = 0.0f;
        for (int i = k; i < D; ++i) {
            v[i] = A[static_cast<std::size_t>(i) * D + k];
        }
        v[k] -= alpha;

        float v_norm_sq = 0.0f;
        for (int i = k; i < D; ++i) v_norm_sq += v[i] * v[i];
        if (v_norm_sq < 1e-24f) continue;

        float two_over_v_dot_v = 2.0f / v_norm_sq;

        // A := (I - tau v v^T) A   for the k-th panel onward
        for (int j = k; j < D; ++j) {
            float dot = 0.0f;
            for (int i = k; i < D; ++i) {
                dot += v[i] * A[static_cast<std::size_t>(i) * D + j];
            }
            float scale = two_over_v_dot_v * dot;
            for (int i = k; i < D; ++i) {
                A[static_cast<std::size_t>(i) * D + j] -= scale * v[i];
            }
        }

        // Q := Q (I - tau v v^T)
        for (int i = 0; i < D; ++i) {
            float dot = 0.0f;
            for (int j = k; j < D; ++j) {
                dot += Q[static_cast<std::size_t>(i) * D + j] * v[j];
            }
            float scale = two_over_v_dot_v * dot;
            for (int j = k; j < D; ++j) {
                Q[static_cast<std::size_t>(i) * D + j] -= scale * v[j];
            }
        }
    }
}

// -------------------------------------------------------------------- //
// init_rotation                                                         //
// -------------------------------------------------------------------- //

rotation_matrix init_rotation(int head_dim, uint32_t seed) {
    rotation_matrix rot;
    rot.head_dim = head_dim;

    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0.0f, 1.0f);

    if (head_dim <= 4096) {
        rot.structured = false;

        // Generate D x D Gaussian matrix
        std::vector<float> G(static_cast<std::size_t>(head_dim) * head_dim);
        for (auto & g : G) g = normal(rng);

        // QR decomposition
        householder_qr(G, rot.Pi, head_dim);

        // Precompute transpose
        rot.Pi_T.assign(rot.Pi.size(), 0.0f);
        for (int i = 0; i < head_dim; ++i) {
            for (int j = 0; j < head_dim; ++j) {
                rot.Pi_T[static_cast<std::size_t>(j) * head_dim + i] =
                    rot.Pi[static_cast<std::size_t>(i) * head_dim + j];
            }
        }
    } else {
        rot.structured = true;

        // Sign flip: random ±1
        rot.sign_flip.resize(head_dim);
        std::bernoulli_distribution coin(0.5);
        for (int i = 0; i < head_dim; ++i) {
            rot.sign_flip[i] = coin(rng) ? 1 : -1;
        }

        // Permutation
        rot.perm.resize(head_dim);
        std::iota(rot.perm.begin(), rot.perm.end(), 0);
        std::shuffle(rot.perm.begin(), rot.perm.end(), rng);

        // Inverse permutation
        rot.inv_perm.resize(head_dim);
        for (int i = 0; i < head_dim; ++i) {
            rot.inv_perm[rot.perm[i]] = i;
        }
    }

    return rot;
}

// -------------------------------------------------------------------- //
// Rotation forward/inverse                                              //
// -------------------------------------------------------------------- //

// y = x @ Pi^T  (rotate x along last axis, length head_dim)
static void rotate(
    const float *           x,
    float *                 y,
    int                     head_dim,
    const rotation_matrix & rot
) {
    if (rot.structured) {
        // y = (x * sign_flip)[perm]
        for (int i = 0; i < head_dim; ++i) {
            y[i] = x[rot.perm[i]] * static_cast<float>(rot.sign_flip[rot.perm[i]]);
        }
        return;
    }
    // Matrix multiply: y[j] = sum_d x[d] * Pi_T[j, d]
    // Pi_T row-major (D, D): Pi_T[j*D + d]
    for (int j = 0; j < head_dim; ++j) {
        float sum = 0.0f;
        const float * pt_row = &rot.Pi_T[static_cast<std::size_t>(j) * head_dim];
        for (int d = 0; d < head_dim; ++d) {
            sum += x[d] * pt_row[d];
        }
        y[j] = sum;
    }
}

// x = y @ Pi  (inverse rotation)
static void unrotate(
    const float *           y,
    float *                 x,
    int                     head_dim,
    const rotation_matrix & rot
) {
    if (rot.structured) {
        // Inverse of forward: x[i] = y[inv_perm[i]] / sign_flip[i]
        for (int i = 0; i < head_dim; ++i) {
            x[i] = y[rot.inv_perm[i]] * static_cast<float>(rot.sign_flip[i]);
        }
        return;
    }
    // x[j] = sum_d y[d] * Pi[j, d]    (note: Pi row-major)
    for (int j = 0; j < head_dim; ++j) {
        float sum = 0.0f;
        const float * p_row = &rot.Pi[static_cast<std::size_t>(j) * head_dim];
        for (int d = 0; d < head_dim; ++d) {
            sum += y[d] * p_row[d];
        }
        x[j] = sum;
    }
}

// -------------------------------------------------------------------- //
// Bit-packing                                                           //
// -------------------------------------------------------------------- //

static std::vector<uint8_t> pack_bits(
    const std::vector<uint8_t> & indices,
    bits                         b
) {
    const std::size_t n = indices.size();

    if (b == BITS_2) {
        // 4 indices per byte
        std::size_t groups = (n + 3) / 4;
        std::vector<uint8_t> out(groups, 0);
        for (std::size_t g = 0; g < groups; ++g) {
            uint8_t v0 = (g * 4 + 0 < n) ? (indices[g * 4 + 0] & 0x3) : 0;
            uint8_t v1 = (g * 4 + 1 < n) ? (indices[g * 4 + 1] & 0x3) : 0;
            uint8_t v2 = (g * 4 + 2 < n) ? (indices[g * 4 + 2] & 0x3) : 0;
            uint8_t v3 = (g * 4 + 3 < n) ? (indices[g * 4 + 3] & 0x3) : 0;
            out[g] = v0 | (v1 << 2) | (v2 << 4) | (v3 << 6);
        }
        return out;
    }

    if (b == BITS_3) {
        // 8 indices per 3 bytes (24 bits)
        std::size_t groups = (n + 7) / 8;
        std::vector<uint8_t> out(groups * 3, 0);
        for (std::size_t g = 0; g < groups; ++g) {
            uint32_t bits24 = 0;
            for (int k = 0; k < 8; ++k) {
                std::size_t i = g * 8 + k;
                uint32_t v = (i < n) ? (indices[i] & 0x7u) : 0u;
                bits24 |= v << (k * 3);
            }
            out[g * 3 + 0] = static_cast<uint8_t>(bits24 & 0xFF);
            out[g * 3 + 1] = static_cast<uint8_t>((bits24 >> 8) & 0xFF);
            out[g * 3 + 2] = static_cast<uint8_t>((bits24 >> 16) & 0xFF);
        }
        return out;
    }

    if (b == BITS_4) {
        // 2 indices per byte
        std::size_t groups = (n + 1) / 2;
        std::vector<uint8_t> out(groups, 0);
        for (std::size_t g = 0; g < groups; ++g) {
            uint8_t v0 = (g * 2 + 0 < n) ? (indices[g * 2 + 0] & 0xF) : 0;
            uint8_t v1 = (g * 2 + 1 < n) ? (indices[g * 2 + 1] & 0xF) : 0;
            out[g] = v0 | (v1 << 4);
        }
        return out;
    }

    throw std::runtime_error("unsupported bits in pack_bits");
}

static std::vector<uint8_t> unpack_bits(
    const std::vector<uint8_t> & packed,
    bits                         b,
    std::size_t                  n_values
) {
    std::vector<uint8_t> out(n_values, 0);

    if (b == BITS_2) {
        for (std::size_t i = 0; i < n_values; ++i) {
            std::size_t byte_idx = i / 4;
            int         shift    = (i % 4) * 2;
            out[i] = (packed[byte_idx] >> shift) & 0x3;
        }
        return out;
    }

    if (b == BITS_3) {
        for (std::size_t i = 0; i < n_values; ++i) {
            std::size_t group     = i / 8;
            int         k         = static_cast<int>(i % 8);
            uint32_t    bits24 = static_cast<uint32_t>(packed[group * 3 + 0])
                        | (static_cast<uint32_t>(packed[group * 3 + 1]) << 8)
                        | (static_cast<uint32_t>(packed[group * 3 + 2]) << 16);
            out[i] = (bits24 >> (k * 3)) & 0x7;
        }
        return out;
    }

    if (b == BITS_4) {
        for (std::size_t i = 0; i < n_values; ++i) {
            std::size_t byte_idx = i / 2;
            int         shift    = (i % 2) * 4;
            out[i] = (packed[byte_idx] >> shift) & 0xF;
        }
        return out;
    }

    throw std::runtime_error("unsupported bits in unpack_bits");
}

// -------------------------------------------------------------------- //
// Quantization helper                                                   //
// -------------------------------------------------------------------- //

// searchsorted: find index i such that boundaries[i-1] <= value < boundaries[i]
// Returns value in [0, n_centroids - 1].
static uint8_t scalar_quantize(float value, const std::vector<float> & boundaries) {
    auto it = std::lower_bound(boundaries.begin(), boundaries.end(), value);
    return static_cast<uint8_t>(it - boundaries.begin());
}

// -------------------------------------------------------------------- //
// Public API: compress / decompress                                     //
// -------------------------------------------------------------------- //

compressed_block compress_vector(
    const float *           vec,
    int                     head_dim,
    bits                    b,
    const rotation_matrix & rot
) {
    // 1. L2 norm
    float norm_sq = 0.0f;
    for (int i = 0; i < head_dim; ++i) norm_sq += vec[i] * vec[i];
    float norm = std::sqrt(norm_sq);
    float safe_norm = std::max(norm, 1e-30f);

    // 2. Normalize to unit vector
    std::vector<float> unit(head_dim);
    for (int i = 0; i < head_dim; ++i) unit[i] = vec[i] / safe_norm;

    // 3. Rotate
    std::vector<float> rotated(head_dim);
    rotate(unit.data(), rotated.data(), head_dim, rot);

    // 4. Scalar quantize each coordinate
    auto centroids  = scaled_centroids(b, head_dim);
    auto boundaries = boundaries_from_centroids(centroids);

    std::vector<uint8_t> indices(head_dim);
    for (int i = 0; i < head_dim; ++i) {
        indices[i] = scalar_quantize(rotated[i], boundaries);
    }

    // 5. Bit-pack
    compressed_block block;
    block.norm    = norm;
    block.indices = pack_bits(indices, b);
    return block;
}

void decompress_vector(
    const compressed_block & block,
    int                      head_dim,
    bits                     b,
    const rotation_matrix &  rot,
    float *                  out
) {
    // 1. Unpack indices
    auto indices = unpack_bits(block.indices, b, head_dim);

    // 2. Look up centroids
    auto centroids = scaled_centroids(b, head_dim);
    std::vector<float> rotated(head_dim);
    for (int i = 0; i < head_dim; ++i) {
        rotated[i] = centroids[indices[i]];
    }

    // 3. Inverse rotation
    std::vector<float> unit(head_dim);
    unrotate(rotated.data(), unit.data(), head_dim, rot);

    // 4. Scale by stored norm
    for (int i = 0; i < head_dim; ++i) {
        out[i] = unit[i] * block.norm;
    }
}

// -------------------------------------------------------------------- //
// CUDA wrapper stubs                                                    //
//                                                                       //
// When LLAMA_TQ_CUDA is defined the real implementations live in        //
// llama-kv-turboquant-cuda.cu. Otherwise these stubs report no GPU      //
// available so callers fall back to the CPU path above.                 //
// -------------------------------------------------------------------- //

#ifndef LLAMA_TQ_CUDA
bool cuda_available() {
    return false;
}

bool compress_vector_cuda(
    const float *           /*vec*/,
    int                     /*head_dim*/,
    bits                    /*b*/,
    const rotation_matrix & /*rot*/,
    compressed_block &      /*out_block*/
) {
    return false;
}

bool decompress_vector_cuda(
    const compressed_block & /*block*/,
    int                      /*head_dim*/,
    bits                     /*b*/,
    const rotation_matrix &  /*rot*/,
    float *                  /*out*/
) {
    return false;
}
#endif  // LLAMA_TQ_CUDA

}  // namespace llama_kv_tq
