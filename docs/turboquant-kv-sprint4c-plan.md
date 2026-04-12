# Sprint 4c plan: wire `tiered_cache` into `llama_kv_cache`

> Status: **plan only** — no `llama_kv_cache.cpp` changes yet. This document
> exists because Sprint 4c is large enough (~1-2 weeks of focused work for
> someone with deep llama.cpp familiarity) that it deserves a written design
> reviewed against trace evidence before any code lands.

## Goal

Make `--cache-type-k tq_kv3` (and `tq_kv2`, `tq_kv4`) actually work end-to-end:
load a model, run inference, get logits within ~1% of the fp16 baseline, with
substantial KV memory savings on long contexts.

The CLI flag is already accepted (Sprint 4b). Today it fails fast in
`llama_init_from_model` with a clear "backend not yet wired" error. Sprint 4c
removes that gate and lights up the storage backend.

## Trace evidence (Atlas, Quadro GV100, Qwen2.5-0.5B Q4_K_M, prefill of "hi")

`bpftrace` instrumented `libllama.so` symbols during a real inference run.
Counts of method invocations:

| Method                              | Count | Why it matters |
|-------------------------------------|------:|----------------|
| `llama_kv_cache::ctor`              |     2 | Cache instances created at context init |
| `llama_kv_cache::init_full`         |     2 | Initial allocation pass |
| `llama_kv_cache::init_batch`        |     6 | Per ubatch admission |
| `llama_kv_cache::prepare`           |     6 | Per forward pass setup |
| `llama_kv_cache::apply_ubatch`      |    12 | Slot bookkeeping (write side) |
| **`llama_kv_cache::cpy_k`**         | **360** | **Per layer, per forward pass — write hot path** |
| **`llama_kv_cache::cpy_v`**         | **360** | **Per layer, per forward pass — write hot path** |
| `llama_kv_cache::clear`             |     3 | Reset paths |
| `llama_kv_cache::update`            |     0 | Not exercised here |

360 cpy_k = 24 layers × 15 forward passes during prefill. **The splice point
is `cpy_k` and `cpy_v`.** Everything else is bookkeeping that already
operates in terms of slot indices, not raw tensor data.

## What `cpy_k` actually does (read of llama-kv-cache.cpp:1196-1228)

It is a **graph builder**, not an immediate copy:

```cpp
ggml_tensor * llama_kv_cache::cpy_k(ggml_context * ctx,
                                    ggml_tensor * k_cur,
                                    ggml_tensor * k_idxs,
                                    int32_t       il,
                                    const slot_info & sinfo) const
{
    const int32_t ikv = map_layer_ids.at(il);
    ggml_tensor * k = layers[ikv].k;          // (1) the cache tensor
    // ... shape merging ...
    return ggml_set_rows(ctx, k, k_cur, k_idxs);   // (2) graph node
}
```

Two facts decide the design:

1. **`layers[ikv].k` is a single big ggml tensor** allocated at construction
   time — `[n_embd_gqa, kv_size, n_stream]` typed as `cache_type_k`. ggml
   ops read/write it directly during attention.

2. **`cpy_k` returns a graph node**, not an immediate write. The actual byte
   movement happens later when `ggml_backend_graph_compute` runs. So any
   TQ-side write must hook into that compute step, not the build step.

For TurboQuant this layout doesn't fit because:
- `layers[ikv].k` can't be a typed ggml tensor (`blck_size=0`, `type_size=0`
  for our tag types).
- ggml ops can't read TQ-packed bytes directly during attention compute —
  they only understand fp16/fp32/Q*/IQ* layouts.

## Architecture: hot+cold storage with materialized fp16 views

```
                    ┌─────────────────────────────┐
   write (cpy_k)    │   tiered_cache (per layer)  │   read (build_attn)
   ───────────────► │   ┌──────────┐  ┌────────┐  │ ──────────────────►
                    │   │ hot ring │  │ cold   │  │
                    │   │ fp16     │  │ TQ pkg │  │
                    │   └──────────┘  └────────┘  │
                    └──────────────┬──────────────┘
                                   │
                  materialize_view(layer, positions)
                                   │
                                   ▼
                    ┌─────────────────────────────┐
                    │  scratch fp16 ggml tensor   │  ── normal ggml attention
                    │  [n_embd_gqa, n_positions]  │     compute path
                    └─────────────────────────────┘
```

- **Storage**: per-layer `tiered_cache` byte buffer outside ggml's tensor
  system. Hot tier holds the most recent `hot_window` tokens as fp16; cold
  tier holds older tokens as TurboQuant-packed bytes.
- **Write (`cpy_k` / `cpy_v`)**: returns a *no-op ggml graph node* with a
  registered post-compute callback. The callback runs after
  `ggml_backend_graph_compute` evaluates `k_cur`, reads the resulting fp16
  values from CPU/GPU memory, and calls `tiered_cache::add_token` to push
  them into the hot tier (auto-evicting to cold).
- **Read (before each forward pass)**: `materialize_view(layer, positions)`
  allocates a temporary fp16 ggml tensor, copies hot positions verbatim
  (already fp16), decompresses cold positions into the tail (CPU or CUDA
  path from Sprint 3), then patches `layers[ikv].k` to point at this
  scratch for the duration of the forward pass. Old `layers[ikv].k` is
  null for TQ types — the materialized view fully replaces it.

This keeps every code path *above* `cpy_k`/`cpy_v`/the cache tensor
unchanged: `build_attn`, `ggml_set_rows`, the CUDA attention kernels — all
operate on the materialized fp16 view, exactly as they would on a normal
fp16 cache.

## Implementation steps

### Step 1 — Map TQ types → fp16 internally (shipped)

**Status: done.** Sprint 4c step 1 ships the minimum end-to-end change:
`llama_init_from_model` substitutes `GGML_TYPE_TQ_KV*` with `GGML_TYPE_F16`
in the `llama_context_params` after warning the user, and the
`llama_kv_cache` constructor does the same substitution defensively for
any direct construction path. The rest of the pipeline sees fp16 exactly
as it always has.

Result: `--cache-type-k tq_kv3` now runs end-to-end without crashing, and
(by definition — it's literally fp16 internally) produces byte-identical
generated tokens to `--cache-type-k f16`.

Verified on Atlas (Quadro GV100, Qwen2.5-0.5B): both runs produce
`"Hello!"` for prompt `"hi"`; `diff` on the generated token region is
empty (only differences are the loading spinner and timing stats).

No compression yet. That's steps 2 and 3 below.

### Step 1b — `is_tq()` helper + tiered_cache allocation (next)

In `llama-kv-cache.cpp`, add a private `is_tq()` predicate. In
`llama_kv_cache::llama_kv_cache(...)`:

- If `type_k` is TQ: skip the `ggml_new_tensor_*` for `layers[i].k` and
  instead allocate a `tiered_cache` per layer (kept in a parallel
  `std::vector<std::unique_ptr<llama_kv_tq::tiered_cache>>` member).
  Same for V.
- Otherwise: existing path (no change).

The constructor signature already takes `type_k` and `type_v` as
`ggml_type`, so no API change is needed. `n_embd_head_k`, `n_embd_head_v`,
`n_kv_heads` come from `model->hparams` — these are the exact inputs
`tiered_cache_config` needs.

### Step 2 — `cpy_k` / `cpy_v` TQ path

Add an early branch:

```cpp
if (is_tq()) {
    return tq_cpy_k_via_callback(ctx, k_cur, il, sinfo);
}
```

`tq_cpy_k_via_callback` builds a ggml graph node whose op is a custom
"identity-with-callback" op (or uses `ggml_map_custom1` if its semantics
fit). The callback reads `k_cur->data` after compute, calls
`tiered_caches[il]->add_token(k_ptr, v_ptr)` per token in the ubatch, and
returns. The returned tensor is a placeholder consumed by no downstream op.

Open question: `ggml_map_custom1` runs on CPU; for GPU-resident `k_cur`
we may need to add a `cudaMemcpyDeviceToHost` first. Sprint 3 already has
the device-side `compress_vector_cuda` — we'd actually prefer to call that
directly on `k_cur->data` while it's still on GPU. This is a perf
optimization to do in the second iteration; the first iteration uses CPU
compression for correctness, then GPU for speed.

### Step 3 — `materialize_view(layer, positions)`

New method on `llama_kv_cache`. Called from `prepare()` once per layer per
forward pass (the trace shows `prepare` fires 6 times for 6 ubatches; this
is the correct cadence — once per ubatch, not once per cpy_k).

Allocates a fp16 ggml tensor of shape `[n_embd_gqa, n_positions]` in the
context's compute buffer. Hot tokens: `memcpy` from
`tiered_cache.hot_storage()`. Cold tokens: call `decompress_vector` (or
`decompress_vector_cuda`) per (head, position) into the tail of the
scratch.

Then assigns `layers[ikv].k = scratch_k`. Existing downstream code reads
this tensor unchanged.

### Step 4 — Hook into `prepare()`

`prepare(const std::vector<llama_ubatch> & ubatches)` already iterates the
ubatches to compute slot bookkeeping. Add a final loop:

```cpp
if (is_tq()) {
    for (int il = 0; il < layers.size(); ++il) {
        layers[il].k = materialize_view_k(il, all_positions_in_ubatch);
        layers[il].v = materialize_view_v(il, all_positions_in_ubatch);
    }
}
```

The scratch tensors live for one ubatch and are reclaimed when the ggml
context resets.

### Step 5 — `seq_rm` / `seq_cp` / `seq_keep` / `clear` / `update`

These manipulate slots, not raw tensor data. For TQ they must call into
`tiered_cache` (drop tokens, copy ranges, etc.). The hot-tier deque + cold
vector layout already supports all these operations naturally.

## Validation

End-to-end smoke (must pass before merge):

1. Load Qwen2.5-0.5B with `--cache-type-k f16` (baseline). Compute logits
   on a 100-token prompt. Save `baseline_logits[100][vocab]`.
2. Load same model with `--cache-type-k tq_kv3`. Same prompt. Save
   `tq_logits[100][vocab]`.
3. Per-token cosine similarity `cos(baseline_logits[i], tq_logits[i])`
   should be > 0.999 for `tq_kv4`, > 0.998 for `tq_kv3`, > 0.99 for
   `tq_kv2`.

Memory check (long context):

1. Load Qwen2.5-7B with `-c 32768 --cache-type-k f16`. Record peak VRAM.
2. Same with `--cache-type-k tq_kv3`. Expect ~5x reduction in KV memory
   (matches Sprint 2 `compression_ratio` table for D=128, 3-bit).

Perf gate: tokens/sec at `tq_kv3` should be within 2x of fp16 baseline on
the GV100 (CUDA decompression carries the cost; the tracing showed
`cpy_k`/`cpy_v` fire 24 times per forward pass, so even ~10us of overhead
per call lands inside this budget).

## What's intentionally NOT in this sprint

- **Flash attention path**: ggml's flash attention kernels read K and V
  directly without going through `cpy_k`/`cpy_v` indices the same way.
  Sprint 4c targets the non-FA path first; FA support is Sprint 4d.
- **Multi-stream KV cache** (`n_stream > 1`): the current `cpy_k` has a
  branch for this; we skip that branch in the TQ path initially and fail
  loudly if anyone selects multi-stream + TQ.
- **`llama_kv_cache_iswa` (sliding window)**: it's a wrapper around two
  `llama_kv_cache` instances; once the base class works, iswa works
  transparently. Verify with a Gemma model in Sprint 4d.
- **`seq_cp` across streams** — same multi-stream caveat as above.

## Risk summary

| Risk | Mitigation |
|------|------------|
| Custom ggml op for cpy_k callback may not exist with the right semantics | Fall back to writing to a scratch fp16 tensor + reading it after compute via `ggml_backend_tensor_get` |
| Materialize-view allocation cost dominates at short contexts | Hot tier covers short contexts entirely; cold path only fires when `n_tokens > hot_window` |
| GPU↔CPU copies for compression hurt prefill throughput | Sprint 3's `compress_vector_cuda` already runs on GPU; wire it into the callback once correctness is proven |
| `prepare()` is called per ubatch but materialization needs the *full* sequence positions | Use `apply_ubatch` callback (already runs at the right time per the trace) instead — re-trace to confirm |
| Quant-aware flash attention is the actual production path | Defer to Sprint 4d after correctness is proven on the slow path |

## Trace methodology (for Sprint 4d / 4e re-runs)

1. `~/llama.cpp/build/bin/llama-cli` exists at Atlas (release build, but
   symbols suffice for uprobes by mangled name).
2. `/tmp/kv_trace.bt` — bpftrace script with `uprobe:` lines on the
   `llama_kv_cache::*` mangled symbols, accumulating into `@evt[name]` and
   printing in `END`.
3. Orchestrate via `/tmp/trace_kv.sh`: `pkill` old runs, launch bpftrace,
   sleep 5 to let probes attach, run a *short bounded* inference (`-n 4
   -p hi --simple-io`, `< /dev/null`, `timeout 30`), `kill -INT` bpftrace,
   sleep 4 to flush.
4. Always redirect llama-cli stdout to `/dev/null` or a small log — naive
   defaults can balloon to multi-GB on what should be a 4-token run.
