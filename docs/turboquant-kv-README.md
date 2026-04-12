# TurboQuant KV Cache Compression

> **Status:** Sprint 2 of 5 (CPU reference complete) · feature branch only · not yet wired to inference

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
| [`tests/test-tq-kv.cpp`](../tests/test-tq-kv.cpp) | Comprehensive unit tests (16K+ checks) | ✅ Sprint 2 |
| [`.github/workflows/turboquant-kv.yml`](../.github/workflows/turboquant-kv.yml) | CI workflow | ✅ |
| `ggml/include/ggml.h` GGML_TYPE_TQ_KV{2,3,4} | New ggml types | ⏳ Sprint 3 |
| `ggml/src/ggml-cuda/turboquant.cu` | CUDA kernels | ⏳ Sprint 3 |
| `src/llama-kv-cache.cpp` integration | Hot/cold tiering | ⏳ Sprint 4 |
| Upstream PR | | ⏳ Sprint 5 |

## Building and testing

```bash
mkdir build && cd build
cmake .. -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF
cmake --build . --target test-tq-kv -j
./bin/test-tq-kv
```

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
- [x] **Sprint 2** — CPU reference + tests + CI ← *you are here*
- [ ] **Sprint 3** — CUDA kernels (Volta/Ampere)
- [ ] **Sprint 4** — Hot/cold tiering integrated into `llama_kv_cache`
- [ ] **Sprint 5** — Documentation + benchmarks + upstream PR

Tracking issue: [ahb-sjsu/turboquant-pro#27](https://github.com/ahb-sjsu/turboquant-pro/issues/27)

## Related improvements (filed as issues, not in scope here)

- **Differential KV** ([turboquant-pro#22](https://github.com/ahb-sjsu/turboquant-pro/issues/22)) — delta-encode adjacent tokens
- **Attention-aware eviction** ([#23](https://github.com/ahb-sjsu/turboquant-pro/issues/23))
- **FP8/NVFP4 native paths** ([#24](https://github.com/ahb-sjsu/turboquant-pro/issues/24)) — Hopper/Blackwell
- **Quantization-aware micro-LoRA** ([#25](https://github.com/ahb-sjsu/turboquant-pro/issues/25)) — for *weight* quantization, separate problem

## Contributing

This branch is under active development. The CI workflow ensures the test suite stays green. Contributions welcome on the issues above.
