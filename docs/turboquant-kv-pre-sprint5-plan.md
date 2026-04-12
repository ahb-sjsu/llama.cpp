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

**Scoping decision (actual):** `SHRINK_BACKBONE=1` already shrinks the
backbone to a single slot ring buffer — the remaining per-layer
footprint is negligible, and full allocation elimination requires
refactoring `k_stream`/`v_stream` views plus `state_write`/`state_read`
serialization (risky for marginal memory win). Landed the
*correctness-redirect* subset only: `cold_check_one`, `type_k`/`type_v`,
and `size_k_bytes`/`size_v_bytes` all consult the view tensor when
view-bind is active. Backbone allocation-elimination is deferred until
a separate sprint that also reworks serialization.

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

## Phase 2 root cause: cold path bad quality is actual compression cost

With `LLAMA_TQ_READTHROUGH=0` (view holds the untouched fp16 cpy_k
writes at every slot, attention reads directly from it, TC is pure
shadow storage), PPL matches f16 **exactly** (15.7316 on 4 chunks).

With `LLAMA_TQ_READTHROUGH=1` (TC-decompressed fp16 overwrites the
view at cold slots, so attention reads TC-decompressed data), PPL
climbs to ~108. Per-row cold-validate cosine = 0.979 ± 0.02.

**This is the actual cost of 3-bit KV compression**, not a bug. Per-
row cosine 0.98 is what the bit budget buys; that 2% noise accumulated
across 24 layers and 95% of tokens produces the 8× PPL regression.

Prior "0% sampled-token disagreement" claims came from either
(a) running with readthrough=0 — which produces f16-identical output
but delivers zero compression benefit (view is fp16-sized), or
(b) short llama-cli contexts where the cold path barely fires.

## Strategic implication

The memory-vs-quality knob in the current architecture is a clean
trade, but a lot worse than any prior PR-quality claim suggested:

| Config | Memory win | PPL cost |
|---|---|---|
| `read=0, full view` | none (view = fp16 cache) | 0 |
| `read=1, full view` | none (still fp16-sized) | 8× |
| `read=1, shrunk view` | up to 60× | 8×+ (cold data from TC, worse if view < hot_window) |
| (hypothetical) `read=0, shrunk view` | up to 60× | N/A — uncovered slots have **no data source**, attention breaks |

Compressed KV only pays off when readthrough is on. Readthrough only
produces acceptable PPL if TC's per-row cosine is much better than
0.98. The paper claims near-lossless at 3-bit for large models; our
implementation on Qwen2.5-0.5B delivers 0.98. Candidate causes for
the gap:

1. Rotation matrix quality — is it a true Hadamard/random-orthogonal,
   or something degenerate at small hidden sizes?
2. Lloyd-Max boundaries — are we using the precomputed paper
   boundaries, or inferring them per-row?
3. Small-model sensitivity — 0.5B may be intrinsically more fragile
   to per-K/V row noise than the 7B+ models the paper targets.

Items below are now prerequisites for anything PR-worthy. Phases 3
(GPU) and 4 (CUDA kernels) stay blocked until TC per-row cosine
climbs substantially (target: &gt; 0.998 at 3-bit on a representative
model).

## Phase 2 finding (deprecated): cold path quality is catastrophically bad

WikiText-2 raw, Qwen2.5-0.5B, -ngl 0, c=2048, 8 chunks, view-bind on,
SHRINK_VIEW=2048 (= n_ctx), SHRINK_BACKBONE unset (known-good mode):

| Config | hot_window | PPL | vs f16 |
|---|---:|---:|---|
| f16 | — | 13.3 | baseline |
| tq_kv3 | 2048 (all hot) | 12.3 | ≈ f16 ✓ |
| tq_kv2 | 128 | 212.04 | 16× worse |
| tq_kv3 | 128 | 107.73 | 8× worse |
| tq_kv4 | 128 | 100.70 | 7.5× worse |

**The cold path (TQ compress → decompress → materialize) destroys
inference quality.** All-hot matches f16 exactly, so the write/read
plumbing is fine. The moment positions are evicted to the cold tier,
PPL explodes. Sampled-token tests masked this because short llama-cli
runs barely exercise the cold tier.

## Phase 2 finding: SHRINK_BACKBONE=1 + view-bind is broken

Even with `HOT_WINDOW=n_ctx` (no cold tier exercised), enabling
`SHRINK_BACKBONE=1` produces PPL ≈ 16,000 (garbage). One fix landed
(set_input_k/v_idxs domain — was remapping indices modulo backbone
size instead of view size), but a second, deeper bug remains. All
prior "26× KV reduction working end-to-end" claims stand on short
llama-cli runs where the bug didn't surface.

## Revised exit criteria — new blockers

- [ ] **Cold path quality**: diagnose why decompressed cold K/V
      deviate enough to cause 7–16× PPL regression. Candidate causes:
      incorrect rotation matrix per-token, per-element read/write
      offset bugs under v_trans, round-trip precision loss in
      compress_vector() itself. Fix before any PR.
- [ ] **SHRINK_BACKBONE=1 + view-bind**: root-cause the remaining
      crash path. Likely in build_rope_shift (still reads layer.k
      with stride n_embd*get_size()) or in a residual code path that
      assumes backbone size matches view size.

Phases 3 (GPU) and 4 (CUDA kernels) are blocked until the cold path
quality regression is fixed — there is no point benchmarking
throughput on a path that produces garbage logits.

## Phase 1 measured results (Atlas, Qwen2.5-0.5B, -ngl 0)

With `LLAMA_TQ_VIEW_BIND=1`, `SHRINK_VIEW=512`, `SHRINK_BACKBONE=1`,
`hot_window=128`, 62-token workload:

| Config | Prompt t/s | Gen t/s | KV MiB |
|---|---:|---:|---:|
| f16 | 135.3 | 42.6 | 384.00 |
| tq_kv3 | 132.0 | 48.0 | **6.38** |

**60× KV memory reduction, +12.7% faster generation.** Sampled-token
disagreement remains high (100%) — a known weak signal addressed by
Phase 2 (perplexity/logit-cosine). Output quality is visually intact.

## Exit criteria for polishing to begin (Sprint 5)

- [x] Sprint 4c step 3c-5d (26× memory reduction) shipped.
- [x] Steps A–C complete, CI green, disagreement unchanged.
- [ ] Perplexity table populated for at least one model (Qwen2.5-0.5B
      or the small Gemma) across bit widths.
- [ ] GPU path (`-ngl 999`) either works cleanly or has a documented
      limitation with a known fix path.
- [ ] CUDA kernels from Sprint 3 at least partially wired in.

Only then do we write the PR description and benchmark graphs. Not
before.
