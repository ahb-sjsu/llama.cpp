# TurboQuant KV Cache Compression

> **Status:** Sprint 4b of 5 (ggml type tags + CLI parsing) · feature branch only · KV storage backend lands in 4c

PolarQuant + Lloyd-Max KV cache compression for `llama.cpp`. Achieves higher compression at better quality than current `Q4_0`/`Q5_0`/`Q8_0` modes.

## TL;DR

For long-context inference (32K+), KV cache memory is the bottleneck. TurboQuant gives:

| Mode | bytes/elem (head_dim=128) | Compression | Cosine sim |
|------|---------------------------|-------------|-----------|
| F16 (baseline) | 2.00 | 1.0× | 1.000 |
| Q8_0 | 1.06 | 1.9× | 0.998 |
| Q4_0 | 0.56 | 3.5× | 0.971 |
| **TQ KV4** | **0.53** | **3.8×** | **0.994** |
| **TQ KV3** | **0.41** | **4.9×** | **0.978** |
| **TQ KV2** | **0.28** | **7.1×** | **0.939** |

Numbers from `tests/test-tq-kv.cpp` (cosine sim from CPU reference impl, head_dim=128).

For Qwen2.5-7B at 32K context: KV cache drops from ~16 GB (F16) to ~3.3 GB (TQ KV3) — fits on a 16 GB GPU.

## Algorithm

Each `head_dim`-sized vector (one row of K or V) is compressed in 5 steps:

1. **L2 norm** — store `‖v‖₂` as fp32 (~4 bytes overhead per vector)
2. **Normalize** — `v_unit = v / ‖v‖₂`
3. **Random orthogonal rotation** — `v_rot = v_unit @ Π`. After rotation, coordinates are approximately i.i.d. Gaussian. The rotation `Π` is fixed per `head_dim` (deterministic from a seed, generated once).
   - For `head_dim ≤ 4096`: full QR-decomposed matrix (Householder)
   - For `head_dim > 4096`: structured sign-flip + permutation (O(D) memory and apply cost)
4. **Scalar quantize** — Lloyd-Max codebook for N(0, 1/√d) scaled to actual dimension. 2/3/4 bits per element.
5. **Bit-pack** — 8 × 3-bit → 3 bytes (TQ KV3); 4 × 2-bit → 1 byte (TQ KV2); 2 × 4-bit → 1 byte (TQ KV4).

Decompression inverts the pipeline.

**Theory:** Zandieh et al. (ICLR 2026) "Sub-linear Memory Inference via PolarQuant + QJL". Reference Python implementation at [ahb-sjsu/turboquant-pro](https://github.com/ahb-sjsu/turboquant-pro).

## What's in this branch

| File | Purpose | Status |
|------|---------|--------|
| [`docs/turboquant-kv-design.md`](turboquant-kv-design.md) | Full architectural design doc | ✅ |
| [`src/llama-kv-turboquant.h`](../src/llama-kv-turboquant.h) | C++ public API | ✅ |
| [`src/llama-kv-turboquant.cpp`](../src/llama-kv-turboquant.cpp) | CPU reference implementation | ✅ Sprint 2 |
| [`tests/test-tq-kv.cpp`](../tests/test-tq-kv.cpp) | Comprehensive unit tests (16K+ checks + CUDA equivalence) | ✅ Sprint 2/3 |
| [`src/llama-kv-turboquant-cuda.cu`](../src/llama-kv-turboquant-cuda.cu) | CUDA kernels (Volta sm_70 + Ampere sm_80) | ✅ Sprint 3 |
| [`src/llama-kv-tiered.{h,cpp}`](../src/llama-kv-tiered.h) | Tiered cache data structure (hot fp16 + cold TQ) | ✅ Sprint 4 |
| [`tests/test-tq-tiered.cpp`](../tests/test-tq-tiered.cpp) | Tiered cache tests (hot/cold/eviction/stats) | ✅ Sprint 4 |
| [`.github/workflows/turboquant-kv.yml`](../.github/workflows/turboquant-kv.yml) | CI workflow (CPU build + CUDA build + lint) | ✅ |
| `ggml/include/ggml.h` GGML_TYPE_TQ_KV{2,3,4} | New ggml types (tags only) | ✅ Sprint 4b |
| `common/arg.cpp` `--cache-type-k tq_kv3` | CLI parser accepts the new types | ✅ Sprint 4b |
| `src/llama-context.cpp` | Friendly error until backend lands | ✅ Sprint 4b |
| `src/llama-kv-cache.cpp` adapter | Wire tiered_cache into llama_memory_i | ⏳ Sprint 4c |
| Upstream PR | | ⏳ Sprint 5 |

## Building and testing

CPU-only build (default):

```bash
mkdir build && cd build
cmake .. -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF
cmake --build . --target test-tq-kv -j
./bin/test-tq-kv
```

With CUDA kernels (Sprint 3):

```bash
mkdir build-cuda && cd build-cuda
cmake .. -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF \
         -DLLAMA_TQ_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="70;80"
cmake --build . --target test-tq-kv -j
./bin/test-tq-kv  # runs CPU tests + CUDA equivalence tests
```

The CUDA tests compare GPU output to the CPU reference (norm, packed
indices, full reconstruction). They self-skip if no CUDA device is
detected at runtime, so the same binary works on both CPU-only and
GPU machines.

Expected output:
```
test_block_size_and_ratio
test_rotation_orthogonality
...
test_round_trip D=256 bits=3
  mean cosine similarity = 0.9782 (need ≥ 0.930)

=== 16531 / 16531 checks passed ===
```

The CI workflow `.github/workflows/turboquant-kv.yml` runs the same on push/PR.

## Tiered cache (Sprint 4)

[`src/llama-kv-tiered.h`](../src/llama-kv-tiered.h) provides
`llama_kv_tq::tiered_cache`, a self-contained KV store that mixes a
small **hot** ring buffer (last `hot_window` tokens, stored fp32 for
exact reads) with a **cold** TurboQuant-compressed tail. Tokens
auto-migrate hot→cold when the window fills.

```cpp
#include "llama-kv-tiered.h"

llama_kv_tq::tiered_cache_config cfg;
cfg.n_layers   = 32;          // attention layers
cfg.n_kv_heads = 8;
cfg.head_dim   = 128;
cfg.hot_window = 512;         // last 512 tokens stay fp16
cfg.cold_bits  = llama_kv_tq::BITS_3;

llama_kv_tq::tiered_cache cache(cfg);

// Append a new token's K and V (n_layers*n_kv_heads*head_dim each).
cache.add_token(k_ptr, v_ptr);

// Read back any previous token (decompresses cold reads on demand).
cache.get_kv(/*pos=*/0, /*layer=*/4, /*head=*/2, k_out, v_out);

auto s = cache.stats();
printf("compression: %.2fx (%d hot, %d cold)\n",
       s.compression_ratio, s.hot_tokens, s.cold_tokens);
```

This data structure is what Sprint 4b will adapt to `llama_memory_i`
so the existing prefill/generation graphs route through it
transparently. Today it is exercised by
[`tests/test-tq-tiered.cpp`](../tests/test-tq-tiered.cpp): hot-only
exact equality, post-eviction cosine reconstruction, eviction order,
memory accounting, multi-layer slot isolation, and a 2,000-token
stress run with 4 layers × 8 heads.

## Test coverage

`tests/test-tq-kv.cpp` exercises:

| Test | What it checks |
|------|---------------|
| `block_size_and_ratio` | Storage layout matches design (TQ KV3 = 52 bytes for D=128) |
| `rotation_orthogonality` | `Π · Π^T == I` to 1e-3 tolerance |
| `structured_rotation` | Permutation + sign-flip invariants for D > 4096 |
| `deterministic_seed` | Same seed → identical rotation matrices |
| `norm_preservation` | Stored norm matches input `‖v‖₂` exactly |
| `zero_vector` | Compressing zeros → reconstructed zeros |
| `unit_axis_vector` | One-hot `e₀` survives round-trip (cos > 0.85) |
| `structured_round_trip` | Cosine > 0.85 for D=8192 (structured path) |
| `bit_packing_edge_cases` | Uneven group sizes (D=13) work for all bit widths |
| `round_trip` (×8 configs) | Mean cosine over 16 random vectors meets per-bit target |

Total: **16,531 individual `CHECK` invocations** across all sub-tests.

## Performance targets (Sprint 3)

After CUDA kernels land:

| Metric | Target | Baseline (F16) |
|--------|--------|----------------|
| Inference throughput | within 2× of F16 | 100% |
| KV memory at 32K context | <30% of F16 | 100% |
| Perplexity loss on WikiText-2 | <0.5% | — |

## Sprint roadmap

- [x] **Sprint 1** — Foundation (fork, design doc, scaffolding, issues filed)
- [x] **Sprint 2** — CPU reference + tests + CI
- [x] **Sprint 3** — CUDA kernels (Volta/Ampere)
- [x] **Sprint 4** — Tiered cache data structure (hot fp16 + cold TQ)
- [x] **Sprint 4b** — `GGML_TYPE_TQ_KV{2,3,4}` registration + CLI parsing
- [ ] **Sprint 4c** — Wire `tiered_cache` into `llama_kv_cache` ← *in progress* (plan: [`turboquant-kv-sprint4c-plan.md`](turboquant-kv-sprint4c-plan.md))
  - [x] **Step 1** — TQ types map to fp16 internally (no crash, no compression yet)
  - [ ] Step 2 — attach `tiered_cache` per layer + observe writes
  - [ ] Step 3 — activate eviction + materialize_view + actual compression
- [ ] **Sprint 5** — Benchmarks + upstream PR

### Sprint 4 / 4b / 4c scope split

The original plan bundled tiering, ggml type registration, and the
KV-cache adapter into one sprint. In practice these are three independent
risks:

- **Sprint 4** proved the tiering data structure works in isolation.
- **Sprint 4b** (this) registers the type tags and wires the CLI surface,
  with a clear failure path until the backend lands. *No inference path
  changes.*
- **Sprint 4c** does the real work: a `llama_memory_i` implementation
  that holds a `tiered_cache` outside ggml's tensor system and
  materializes fp16 views before each attention step. This is a real
  refactor of `llama_kv_cache.cpp` and the prefill/generation graphs.

### Why are these "tag" types and not full ggml types?

ggml types assume a compile-time fixed `blck_size` (e.g. `QK_K = 256`
elements per packed block). TurboQuant compresses one *head_dim*-sized
vector at a time, where `head_dim` is a runtime model parameter
(64, 128, 192, 256...). The packed layout depends on `head_dim`, so it
doesn't fit ggml's "K elements → N bytes" model.

The fix is architectural: TQ-compressed K/V is stored *outside* ggml's
tensor system (in our `tiered_cache` byte buffers), and the KV-cache
layer materializes a temporary fp16 ggml tensor for each attention
step. So `GGML_TYPE_TQ_KV*` are *storage backend tags* — they tell the
KV-cache layer "use the TurboQuant backend", not "this tensor's elements
are packed this way". Their `blck_size` and `type_size` are 0 to make
this contract explicit; any code path that tries to treat them as a
packed tensor type will fail loudly.

Tracking issue: [ahb-sjsu/turboquant-pro#27](https://github.com/ahb-sjsu/turboquant-pro/issues/27)

## Related improvements (filed as issues, not in scope here)

- **Differential KV** ([turboquant-pro#22](https://github.com/ahb-sjsu/turboquant-pro/issues/22)) — delta-encode adjacent tokens
- **Attention-aware eviction** ([#23](https://github.com/ahb-sjsu/turboquant-pro/issues/23))
- **FP8/NVFP4 native paths** ([#24](https://github.com/ahb-sjsu/turboquant-pro/issues/24)) — Hopper/Blackwell
- **Quantization-aware micro-LoRA** ([#25](https://github.com/ahb-sjsu/turboquant-pro/issues/25)) — for *weight* quantization, separate problem

## Contributing

This branch is under active development. The CI workflow ensures the test suite stays green. Contributions welcome on the issues above.
