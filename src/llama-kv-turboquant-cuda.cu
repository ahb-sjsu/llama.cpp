// CUDA kernels for TurboQuant KV cache compression (Sprint 3)
//
// Self-contained — does not yet touch ggml's tensor system.
// Sprint 4 will wire these into the actual KV cache via ggml types.
//
// Targets: sm_70 (Volta/GV100) and sm_80+ (Ampere/Hopper).
//
// Each function has a CPU fallback in llama-kv-turboquant.cpp;
// the wrapper functions below dispatch to GPU when LLAMA_TQ_CUDA
// is defined and a CUDA device is available.

#include "llama-kv-turboquant.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <vector>

// Lloyd-Max codebook centroids — must match CPU reference in llama-kv-turboquant.cpp
__device__ static const float CODEBOOK_2_DEV[4] = {
    -1.510f, -0.453f, 0.453f, 1.510f
};
__device__ static const float CODEBOOK_3_DEV[8] = {
    -1.748f, -1.050f, -0.500f, -0.069f,
     0.069f,  0.500f,  1.050f, 1.748f
};
__device__ static const float CODEBOOK_4_DEV[16] = {
    -2.401f, -1.844f, -1.437f, -1.099f,
    -0.800f, -0.524f, -0.262f, -0.066f,
     0.066f,  0.262f,  0.524f,  0.800f,
     1.099f,  1.437f,  1.844f,  2.401f
};

#define CUDA_CHECK(call) do {                                            \
    cudaError_t _e = (call);                                             \
    if (_e != cudaSuccess) {                                             \
        std::fprintf(stderr, "CUDA error %s:%d: %s\n",                   \
            __FILE__, __LINE__, cudaGetErrorString(_e));                 \
        return false;                                                    \
    }                                                                    \
} while (0)

namespace llama_kv_tq {

// -------------------------------------------------------------------- //
// Kernel 1: rotate (matvec for QR mode, perm+sign-flip for structured) //
// -------------------------------------------------------------------- //

// y[j] = sum_d x[d] * Pi_T[j*D + d]
// One block per output element, threads collaborate via warp shuffle.
__global__ static void rotate_qr_kernel(
    const float * __restrict__ x,
    const float * __restrict__ Pi_T,
    float       * __restrict__ y,
    int                        D
) {
    const int j = blockIdx.x;
    if (j >= D) return;

    const float * pt_row = Pi_T + j * D;

    float sum = 0.0f;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        sum += x[d] * pt_row[d];
    }

    // Warp-level reduction (assumes blockDim.x is a power of 2 ≤ 32)
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        sum += __shfl_xor_sync(0xFFFFFFFF, sum, off);
    }

    if (threadIdx.x == 0) y[j] = sum;
}

__global__ static void rotate_structured_kernel(
    const float   * __restrict__ x,
    const int8_t  * __restrict__ sign_flip,
    const int32_t * __restrict__ perm,
    float         * __restrict__ y,
    int                          D
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= D) return;
    const int p = perm[i];
    y[i] = x[p] * static_cast<float>(sign_flip[p]);
}

__global__ static void unrotate_structured_kernel(
    const float   * __restrict__ y,
    const int8_t  * __restrict__ sign_flip,
    const int32_t * __restrict__ inv_perm,
    float         * __restrict__ x,
    int                          D
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= D) return;
    x[i] = y[inv_perm[i]] * static_cast<float>(sign_flip[i]);
}

// -------------------------------------------------------------------- //
// Kernel 2: L2 norm (sum of squares + sqrt)                            //
// -------------------------------------------------------------------- //

__global__ static void l2_norm_kernel(
    const float * __restrict__ x,
    float       * __restrict__ norm_out,
    int                        D
) {
    extern __shared__ float sdata[];
    const int tid = threadIdx.x;

    float acc = 0.0f;
    for (int d = tid; d < D; d += blockDim.x) {
        const float v = x[d];
        acc += v * v;
    }

    // Block reduction via shared mem
    sdata[tid] = acc;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) *norm_out = sqrtf(sdata[0]);
}

// -------------------------------------------------------------------- //
// Kernel 3: normalize then quantize                                    //
// (input: rotated unit vector → indices 0..2^bits-1)                   //
// -------------------------------------------------------------------- //

__device__ static inline uint8_t quantize_lookup(
    float           v,
    const float   * centroids,  // scaled centroids, n entries
    int             n
) {
    // searchsorted across boundaries (midpoints between consecutive centroids)
    // Use linear scan — fine for n ≤ 16.
    uint8_t idx = 0;
    for (int k = 1; k < n; ++k) {
        float boundary = 0.5f * (centroids[k - 1] + centroids[k]);
        if (v >= boundary) idx = k;
        else break;
    }
    return idx;
}

__global__ static void quantize_kernel(
    const float * __restrict__ rotated,  // (D,)
    uint8_t     * __restrict__ indices,  // (D,)
    int                        D,
    int                        bits,
    float                      scale     // 1/sqrt(D)
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= D) return;

    const float v = rotated[i];

    // Pick scaled codebook based on bits
    if (bits == 2) {
        float c[4] = {
            CODEBOOK_2_DEV[0] * scale, CODEBOOK_2_DEV[1] * scale,
            CODEBOOK_2_DEV[2] * scale, CODEBOOK_2_DEV[3] * scale
        };
        indices[i] = quantize_lookup(v, c, 4);
    } else if (bits == 3) {
        float c[8];
        for (int k = 0; k < 8; ++k) c[k] = CODEBOOK_3_DEV[k] * scale;
        indices[i] = quantize_lookup(v, c, 8);
    } else { // 4
        float c[16];
        for (int k = 0; k < 16; ++k) c[k] = CODEBOOK_4_DEV[k] * scale;
        indices[i] = quantize_lookup(v, c, 16);
    }
}

__global__ static void dequantize_kernel(
    const uint8_t * __restrict__ indices,
    float         * __restrict__ rotated,
    int                          D,
    int                          bits,
    float                        scale
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= D) return;

    if (bits == 2) {
        rotated[i] = CODEBOOK_2_DEV[indices[i] & 0x3] * scale;
    } else if (bits == 3) {
        rotated[i] = CODEBOOK_3_DEV[indices[i] & 0x7] * scale;
    } else {
        rotated[i] = CODEBOOK_4_DEV[indices[i] & 0xF] * scale;
    }
}

// -------------------------------------------------------------------- //
// Kernel 4: bit packing (port from Python CuPy version)                //
// -------------------------------------------------------------------- //

__global__ static void pack_2bit_kernel(
    const uint8_t * __restrict__ in,    // n indices
    uint8_t       * __restrict__ out,   // (n+3)/4 bytes
    int                          n
) {
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    const int groups = (n + 3) / 4;
    if (g >= groups) return;

    uint8_t v0 = (g * 4 + 0 < n) ? (in[g * 4 + 0] & 0x3) : 0;
    uint8_t v1 = (g * 4 + 1 < n) ? (in[g * 4 + 1] & 0x3) : 0;
    uint8_t v2 = (g * 4 + 2 < n) ? (in[g * 4 + 2] & 0x3) : 0;
    uint8_t v3 = (g * 4 + 3 < n) ? (in[g * 4 + 3] & 0x3) : 0;
    out[g] = v0 | (v1 << 2) | (v2 << 4) | (v3 << 6);
}

__global__ static void unpack_2bit_kernel(
    const uint8_t * __restrict__ in,
    uint8_t       * __restrict__ out,
    int                          n
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int byte_idx = i / 4;
    int shift    = (i % 4) * 2;
    out[i] = (in[byte_idx] >> shift) & 0x3;
}

__global__ static void pack_3bit_kernel(
    const uint8_t * __restrict__ in,    // n indices
    uint8_t       * __restrict__ out,   // groups*3 bytes
    int                          n
) {
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    const int groups = (n + 7) / 8;
    if (g >= groups) return;

    uint32_t bits24 = 0;
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        int i = g * 8 + k;
        uint32_t v = (i < n) ? static_cast<uint32_t>(in[i] & 0x7) : 0u;
        bits24 |= v << (k * 3);
    }
    out[g * 3 + 0] = static_cast<uint8_t>(bits24 & 0xFF);
    out[g * 3 + 1] = static_cast<uint8_t>((bits24 >> 8) & 0xFF);
    out[g * 3 + 2] = static_cast<uint8_t>((bits24 >> 16) & 0xFF);
}

__global__ static void unpack_3bit_kernel(
    const uint8_t * __restrict__ in,
    uint8_t       * __restrict__ out,
    int                          n
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int group = i / 8;
    int k     = i % 8;
    uint32_t bits24 = static_cast<uint32_t>(in[group * 3 + 0])
                    | (static_cast<uint32_t>(in[group * 3 + 1]) << 8)
                    | (static_cast<uint32_t>(in[group * 3 + 2]) << 16);
    out[i] = (bits24 >> (k * 3)) & 0x7;
}

__global__ static void pack_4bit_kernel(
    const uint8_t * __restrict__ in,
    uint8_t       * __restrict__ out,
    int                          n
) {
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    const int groups = (n + 1) / 2;
    if (g >= groups) return;

    uint8_t v0 = (g * 2 + 0 < n) ? (in[g * 2 + 0] & 0xF) : 0;
    uint8_t v1 = (g * 2 + 1 < n) ? (in[g * 2 + 1] & 0xF) : 0;
    out[g] = v0 | (v1 << 4);
}

__global__ static void unpack_4bit_kernel(
    const uint8_t * __restrict__ in,
    uint8_t       * __restrict__ out,
    int                          n
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int byte_idx = i / 2;
    int shift    = (i % 2) * 4;
    out[i] = (in[byte_idx] >> shift) & 0xF;
}

__global__ static void scale_by_norm_kernel(
    const float * __restrict__ unit_vec,
    float                      norm,
    float       * __restrict__ out,
    int                        D
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= D) return;
    out[i] = unit_vec[i] * norm;
}

__global__ static void normalize_kernel(
    const float * __restrict__ in,
    float                      safe_norm,
    float       * __restrict__ out,
    int                        D
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= D) return;
    out[i] = in[i] / safe_norm;
}

// -------------------------------------------------------------------- //
// Public C++ wrappers (host-side memory mgmt + kernel dispatch)        //
// -------------------------------------------------------------------- //

bool cuda_available() {
    int count = 0;
    cudaError_t e = cudaGetDeviceCount(&count);
    return e == cudaSuccess && count > 0;
}

bool compress_vector_cuda(
    const float *           vec_h,
    int                     head_dim,
    bits                    b,
    const rotation_matrix & rot,
    compressed_block &      out_block
) {
    const int    bits_int = static_cast<int>(b);
    const float  scale    = 1.0f / sqrtf(static_cast<float>(head_dim));
    const size_t D        = head_dim;

    // Allocate device memory
    float   * d_vec      = nullptr;
    float   * d_unit     = nullptr;
    float   * d_rotated  = nullptr;
    float   * d_norm     = nullptr;
    uint8_t * d_indices  = nullptr;
    uint8_t * d_packed   = nullptr;
    void    * d_rot_data = nullptr;  // Pi_T or sign+perm

    const size_t n_indices = D;
    const size_t n_packed  = (D * bits_int + 7) / 8;

    CUDA_CHECK(cudaMalloc(&d_vec,     D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_unit,    D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rotated, D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_norm,        sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_indices, n_indices));
    CUDA_CHECK(cudaMalloc(&d_packed,  n_packed));

    CUDA_CHECK(cudaMemcpy(d_vec, vec_h, D * sizeof(float),
                          cudaMemcpyHostToDevice));

    // 1) L2 norm
    {
        int threads = 256;
        l2_norm_kernel<<<1, threads, threads * sizeof(float)>>>(
            d_vec, d_norm, D);
    }

    // 2) Read norm back and compute safe_norm on host
    float h_norm = 0.0f;
    CUDA_CHECK(cudaMemcpy(&h_norm, d_norm, sizeof(float),
                          cudaMemcpyDeviceToHost));
    float safe_norm = h_norm > 1e-30f ? h_norm : 1e-30f;

    // 3) Normalize → unit
    {
        int threads = 256;
        int blocks  = (D + threads - 1) / threads;
        normalize_kernel<<<blocks, threads>>>(d_vec, safe_norm, d_unit, D);
    }

    // 4) Rotate
    if (rot.structured) {
        int8_t  * d_signs = nullptr;
        int32_t * d_perm  = nullptr;
        CUDA_CHECK(cudaMalloc(&d_signs, D * sizeof(int8_t)));
        CUDA_CHECK(cudaMalloc(&d_perm,  D * sizeof(int32_t)));
        CUDA_CHECK(cudaMemcpy(d_signs, rot.sign_flip.data(),
                              D * sizeof(int8_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_perm, rot.perm.data(),
                              D * sizeof(int32_t), cudaMemcpyHostToDevice));

        int threads = 256;
        int blocks  = (D + threads - 1) / threads;
        rotate_structured_kernel<<<blocks, threads>>>(
            d_unit, d_signs, d_perm, d_rotated, D);

        cudaFree(d_signs);
        cudaFree(d_perm);
    } else {
        // Upload Pi_T (one-time per call — Sprint 4 will cache on device)
        CUDA_CHECK(cudaMalloc(&d_rot_data, D * D * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_rot_data, rot.Pi_T.data(),
                              D * D * sizeof(float),
                              cudaMemcpyHostToDevice));

        int threads = 32;  // warp size — matches reduction in kernel
        rotate_qr_kernel<<<D, threads>>>(
            d_unit, static_cast<float *>(d_rot_data), d_rotated, D);
        cudaFree(d_rot_data);
    }

    // 5) Quantize
    {
        int threads = 256;
        int blocks  = (D + threads - 1) / threads;
        quantize_kernel<<<blocks, threads>>>(
            d_rotated, d_indices, D, bits_int, scale);
    }

    // 6) Pack
    if (b == BITS_2) {
        int groups  = (D + 3) / 4;
        int threads = 256;
        int blocks  = (groups + threads - 1) / threads;
        pack_2bit_kernel<<<blocks, threads>>>(d_indices, d_packed, D);
    } else if (b == BITS_3) {
        int groups  = (D + 7) / 8;
        int threads = 256;
        int blocks  = (groups + threads - 1) / threads;
        pack_3bit_kernel<<<blocks, threads>>>(d_indices, d_packed, D);
    } else {
        int groups  = (D + 1) / 2;
        int threads = 256;
        int blocks  = (groups + threads - 1) / threads;
        pack_4bit_kernel<<<blocks, threads>>>(d_indices, d_packed, D);
    }

    // 7) Copy packed result back
    out_block.norm = h_norm;
    out_block.indices.assign(n_packed, 0);
    CUDA_CHECK(cudaMemcpy(out_block.indices.data(), d_packed, n_packed,
                          cudaMemcpyDeviceToHost));

    cudaFree(d_vec);
    cudaFree(d_unit);
    cudaFree(d_rotated);
    cudaFree(d_norm);
    cudaFree(d_indices);
    cudaFree(d_packed);
    return true;
}

bool decompress_vector_cuda(
    const compressed_block & block,
    int                      head_dim,
    bits                     b,
    const rotation_matrix &  rot,
    float *                  out_h
) {
    const int    bits_int = static_cast<int>(b);
    const float  scale    = 1.0f / sqrtf(static_cast<float>(head_dim));
    const size_t D        = head_dim;
    const size_t n_packed = block.indices.size();

    uint8_t * d_packed   = nullptr;
    uint8_t * d_indices  = nullptr;
    float   * d_rotated  = nullptr;
    float   * d_unit     = nullptr;
    float   * d_out      = nullptr;

    CUDA_CHECK(cudaMalloc(&d_packed,  n_packed));
    CUDA_CHECK(cudaMalloc(&d_indices, D));
    CUDA_CHECK(cudaMalloc(&d_rotated, D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_unit,    D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out,     D * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_packed, block.indices.data(), n_packed,
                          cudaMemcpyHostToDevice));

    // 1) Unpack
    int threads = 256;
    int blocks  = (D + threads - 1) / threads;
    if (b == BITS_2) {
        unpack_2bit_kernel<<<blocks, threads>>>(d_packed, d_indices, D);
    } else if (b == BITS_3) {
        unpack_3bit_kernel<<<blocks, threads>>>(d_packed, d_indices, D);
    } else {
        unpack_4bit_kernel<<<blocks, threads>>>(d_packed, d_indices, D);
    }

    // 2) Dequantize
    dequantize_kernel<<<blocks, threads>>>(
        d_indices, d_rotated, D, bits_int, scale);

    // 3) Inverse rotation
    if (rot.structured) {
        int8_t  * d_signs    = nullptr;
        int32_t * d_inv_perm = nullptr;
        CUDA_CHECK(cudaMalloc(&d_signs,    D * sizeof(int8_t)));
        CUDA_CHECK(cudaMalloc(&d_inv_perm, D * sizeof(int32_t)));
        CUDA_CHECK(cudaMemcpy(d_signs, rot.sign_flip.data(),
                              D * sizeof(int8_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_inv_perm, rot.inv_perm.data(),
                              D * sizeof(int32_t), cudaMemcpyHostToDevice));

        unrotate_structured_kernel<<<blocks, threads>>>(
            d_rotated, d_signs, d_inv_perm, d_unit, D);
        cudaFree(d_signs);
        cudaFree(d_inv_perm);
    } else {
        // y = rotated, x = unit; x[j] = sum_d y[d] * Pi[j, d]
        // Reuse rotate_qr_kernel by passing Pi (not Pi_T)
        float * d_Pi = nullptr;
        CUDA_CHECK(cudaMalloc(&d_Pi, D * D * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_Pi, rot.Pi.data(),
                              D * D * sizeof(float),
                              cudaMemcpyHostToDevice));
        rotate_qr_kernel<<<D, 32>>>(d_rotated, d_Pi, d_unit, D);
        cudaFree(d_Pi);
    }

    // 4) Scale by norm
    scale_by_norm_kernel<<<blocks, threads>>>(d_unit, block.norm, d_out, D);

    CUDA_CHECK(cudaMemcpy(out_h, d_out, D * sizeof(float),
                          cudaMemcpyDeviceToHost));

    cudaFree(d_packed);
    cudaFree(d_indices);
    cudaFree(d_rotated);
    cudaFree(d_unit);
    cudaFree(d_out);
    return true;
}

}  // namespace llama_kv_tq
