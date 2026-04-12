# TurboQuant KV — pre-Sprint-5 cleanup and validation plan

> Status: **in progress**. Items 1–3 are CPU-side cleanups and perf wins.
> Items 4–6 are the validation and GPU work that must land before any
> upstream PR.

## Motivation

Sprint 4c shipped a 26× total memory reduction end-to-end, but the
implementation has rough edges (orphaned backbone tensor, redundant
writebacks, slow transposed-V observation) and unproven claims (single-
prompt agreement on Qwen2.5-0.5B is not a quality signal). This plan
addresses both before polish and upstream submission.

## Items, in execution order

### Phase 1 — CPU cleanups (low risk, measurable perf)

#### Step A: skip writeback for hot positions (item 2)

The writeback currently walks every mapped slot and writes `fp32(fp16-
observed) → fp16` back into the view tensor. For hot-tier positions
this round-trip is identity with what `cpy_k` already wrote, so every
write is a no-op `ggml_backend_tensor_set` call. At 512-slot hot window
× 24 layers × 2 sides that is ~25 K no-op writes per generation step.

Fix: in `tq_apply_readthrough_`'s inner loop, `continue` when
`tc_pos >= tq->n_cold_tokens()`.

Verify: benchmark gen tok/s before/after. Expect 5–15% improvement on
short contexts, larger on long ones. Sampled-token output must remain
identical.

#### Step B: eliminate the backbone when view-bind is on (item 1)

With view-bind, `cpy_k`/`cpy_v` write to the view, `observe` reads from
the view, writeback targets the view. `layers[il].k` / `layers[il].v`
are allocated but never accessed on the hot path.

Fix: when `view_bind` is active, don't allocate the backbone tensors at
all. Audit the non-hot-path consumers:
- `state_write` / `state_read` (cache serialization)
- `size_k_bytes` / `size_v_bytes`
- `k_stream` / `v_stream` views
- `seq_*` methods (should already go through cells bookkeeping only)

Route each to the view tensor when view-bind is on.

`LLAMA_TQ_SHRINK_BACKBONE` becomes a no-op knob (backbone is gone) —
remove it or leave as ignored with a warn-once.

Verify: same benchmark, same disagreement %, KV MiB shrinks further.

#### Step C: batch-read transposed V observation (item 3)

Current v_trans observe does `n_embd_v_gqa` individual
`ggml_backend_tensor_get` calls per token per layer. For a 30-token
prompt across 24 layers that is 92 K small reads.

Better: for each of the `n_embd_v_gqa` elements, read the full row
(`kv_size` fp16) in one call, then extract the N token columns for
this flush's pending slots. Total reads drop from `N * n_embd_v_gqa`
per layer to `n_embd_v_gqa` per layer.

Verify: prompt tok/s improvement, especially when `v_trans=true` (no
`--flash-attn`). Cold-validate cosines must still meet per-bit
thresholds.

### Phase 2 — Real quality evaluation (item 4)

Single-prompt sampled-token agreement is a weak signal. Before any
upstream PR we need the standard quality numbers:

1. **Perplexity grid**: `llama-perplexity` on WikiText-2 raw, sliding
   window, `n_ctx=2048` (standard). Grid:
   - Baseline f16
   - tq_kv4, tq_kv3, tq_kv2
   - Hot windows 32, 128, 512, 2048 (= all hot)

2. **Logit-level fidelity**: for a few real prompts (long-context, code,
   multi-turn), compute full-vocab logit cosine similarity and top-K
   agreement (K=5, 20, 100) vs f16. Sampled-top-1 is too coarse.

3. **Record in a single markdown table**. This becomes the evidence
   table for the Sprint 5 PR description.

### Phase 3 — GPU backend verification (item 5)

All testing so far has been `-ngl 0` (CPU). `ggml_backend_tensor_set` /
`ggml_backend_tensor_get` on CUDA have different semantics
(async queues, device↔host copies). Race conditions and correctness
gaps that CPU never showed are likely to surface on GPU.

Steps:
1. Run tq_kv3 with `-ngl 999` on Atlas's GV100.
2. If output crashes or diverges: bpftrace `cudaMemcpy*` /
   `cudaStreamSynchronize` entry/exit to find missing syncs.
3. If correct: measure tok/s and memory vs baseline.

### Phase 4 — Wire Sprint 3 CUDA kernels (item 6)

Sprint 3 shipped CUDA kernels (`compress_vector_cuda`,
`decompress_vector_cuda`) that were proven bit-identical to CPU in
unit tests but **never wired into the inference path**. The observe →
compress → decompress cycle currently runs entirely on CPU.

Steps:
1. In `tq_apply_readthrough_`'s writeback, call
   `decompress_vector_cuda` when data is on GPU.
2. Similarly in `materialize_fp16_rows`.
3. In the observe path, call `compress_vector_cuda` directly from the
   device-resident view tensor (no device→host bounce).
4. Benchmark: expect 5–50× speedup on the compression path for long
   contexts.

This is the "selling point" of TurboQuant-on-GPU and is what will make
the upstream PR attractive to anyone running on NVIDIA hardware.

## Verification strategy

Each step ships with:
- Before/after benchmark (tok/s, memory, disagreement %)
- If behavior changed, `LLAMA_TQ_DIAG_GETK`-style pointer/state tracing
- Atlas run to confirm, not just local compile

Commits are small and one-concept-each so any regression is bisectable.

## Exit criteria for polishing to begin (Sprint 5)

- [x] Sprint 4c step 3c-5d (26× memory reduction) shipped.
- [ ] Steps A–C complete, CI green, disagreement unchanged.
- [ ] Perplexity table populated for at least one model (Qwen2.5-0.5B
      or the small Gemma) across bit widths.
- [ ] GPU path (`-ngl 999`) either works cleanly or has a documented
      limitation with a known fix path.
- [ ] CUDA kernels from Sprint 3 at least partially wired in.

Only then do we write the PR description and benchmark graphs. Not
before.
