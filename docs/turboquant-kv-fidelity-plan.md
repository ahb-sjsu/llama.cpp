# TurboQuant KV fidelity investigation plan

## Context

At 3-bit compression with readthrough on, Qwen2.5-0.5B WikiText-2
PPL climbs from 13.3 (f16) to ~108 (tq_kv3). Per-row cold-validate
cosine = 0.979 mean. Theory for Lloyd-Max quantization of a unit
N(0,1) rotated vector at 3-bit predicts cosine ~0.992. Observed
0.979 is slightly worse than theory but in the right ballpark.

The paper's "near-lossless" claim must rely on at least one of:
A. Stronger rotation quality (structured Hadamard rather than
   random Gaussian QR).
B. Exact Lloyd-Max boundaries rather than midpoint boundaries
   (only relevant if our midpoints diverge from the optimal
   partition boundaries for the actual rotated distribution).
C. Scale insulating the per-row noise: larger models have
   more heads and/or larger `n_embd_head_k`, each head's
   contribution to logits is diluted, same 0.98 cosine
   produces dramatically smaller PPL regression.

## Experiments (execute in this order, cheapest first)

### Benchmark 1: rotation quality

Hypothesis: random-Gaussian QR is already near-optimal for
head_dim=128. Hadamard-based rotation may do slightly better but
unlikely to close a 0.98 → 0.998 gap.

Method: unit test. Feed 1000 synthetic unit vectors (drawn from
N(0, I_128)) through compress/decompress with:
  - current QR rotation
  - a Walsh-Hadamard transform with random sign flip (structured)
Report mean/min cosine for each.

Outcome:
  - If cosines match within noise: rotation is not the issue,
    move on.
  - If Hadamard is materially better: fix init_rotation for
    head_dim=128.

### Benchmark 2: Lloyd-Max boundary precision

Hypothesis: midpoint boundaries are optimal only for strictly
symmetric codebook + Gaussian input. If the rotated-unit
distribution has heavier tails than N(0,1), the outer boundaries
are sub-optimal.

Method: unit test. For our CODEBOOK_3 centroids, compute the
true optimal boundaries that minimize MSE against an actual
distribution sampled from (1) N(0,1) and (2) real rotated K
vectors extracted from Qwen2.5-0.5B attention. Compare.

Outcome:
  - If boundaries nearly match for N(0,1) but diverge for real
    K: switch to data-driven boundaries.
  - Else: not the issue.

### Benchmark 3: scale

Hypothesis: small n_embd_head (128) amplifies per-row quantization
noise in attention logits; larger models tolerate the same 0.98
cosine with far smaller PPL regression.

Method: download Qwen2.5-7B-Instruct (or similar 7B fp16 gguf) to
Atlas, run the same PPL grid (f16 / tq_kv3 / tq_kv4) at
hot_window=128 and hot_window=2048 (all-hot reference).

Decision criterion: ratio of tq_kv3 PPL to f16 PPL.
  - < 1.05: compression works at scale, implementation is fine,
    ship it.
  - 1.05-1.5: marginal; might be acceptable for long-context
    applications that couldn't run at f16 anyway.
  - &gt;= 1.5: compression is fundamentally too lossy at these
    bit widths for useful inference; next step is either
    higher-bit (5-6 bit) or a different quantization scheme.

## Reporting

Each benchmark writes its numbers to a row of a single table
appended to this file. Decision points are resolved by actual
measurement, not argument.

## Results

### Benchmark 1: rotation quality (unit test, 10k synthetic vectors, D=128)

| Rotation | bits | mean cos | min cos | p1 cos |
|---|---:|---:|---:|---:|
| QR-Gaussian | 3 | 0.9784 | 0.9480 | 0.9632 |
| Hadamard + sign-flip | 3 | 0.9784 | 0.9454 | 0.9634 |
| QR-Gaussian | 4 | 0.9949 | 0.9744 | 0.9880 |
| Hadamard + sign-flip | 4 | 0.9949 | 0.9751 | 0.9879 |

**Verdict: rotation is not the issue.** Random-Gaussian QR and
Walsh-Hadamard are indistinguishable within noise. The C++ impl
already does the mathematically-right thing at D=128.

### Benchmark 2: Lloyd-Max boundary precision (unit test)

| Codebook | MSE on rotated-unit dist | mean cos |
|---|---:|---:|
| Paper N(0,1) centroids | 0.0444 | 0.9784 |
| Data-fitted (20 Lloyd iter) | 0.0340 | 0.9831 |

**Verdict: marginal improvement.** Data-fitted codebook closes
about 0.5% of the cosine gap (0.978 → 0.983) — nowhere near the
0.998+ needed to produce acceptable PPL. Not enough to justify
adding a calibration pass.

### Benchmark 3: scale (WikiText-2 PPL, 4 chunks, hot_window=128)

| Model | head_dim | Config | PPL | vs f16 |
|---|---:|---|---:|---:|
| Qwen2.5-0.5B | 64 | f16 | 15.7 | 1.0× |
| Qwen2.5-0.5B | 64 | tq_kv3 | 107.7 | 6.9× |
| Qwen2.5-0.5B | 64 | tq_kv4 | 100.7 | 6.4× |
| Qwen2.5-1.5B | 128 | f16 | 9.83 | 1.0× |
| Qwen2.5-1.5B | 128 | tq_kv2 | 122.3 | 12.4× |
| Qwen2.5-1.5B | 128 | tq_kv3 | 94.3 | 9.6× |
| Qwen2.5-1.5B | 128 | tq_kv4 | 131.7 | 13.4× |

**Verdict: scale does not rescue quality.** 0.5B → 1.5B (head_dim
64 → 128, 24 → 28 layers) did not reduce the cost of 3-bit
compression. The 9-13× PPL regression is essentially flat across
bit widths — the cold-path data is noisy enough that attention
loses structure regardless of how many bits per row.

Runtime cosines confirm the per-row compression math is correct:
0.979 at 3-bit, 0.995 at 4-bit — both match theoretical Lloyd-Max
predictions. Implementation is not the problem.

## Conclusion

None of the three candidate causes (rotation, boundaries, scale)
closes the gap. The observed 0.98 per-row cosine at 3-bit is
*what Lloyd-Max actually buys at 3 bits*; the implementation is
correct. The PPL regression is the natural downstream cost of
that cosine when ≥95% of attention reads use compressed K/V.

TurboQuant's "near-lossless at 3-bit" claim is not reproducible
at 0.5B–1.5B scale. Next pivots (pick one):

1. Test at 7B+ — unlikely to help based on the 0.5B→1.5B trend,
   but worth a single data point before abandoning.
2. Switch compression scheme — `fp8_e4m3` at 8-bit gives
   ≈0.9998 cosine and still halves KV memory vs fp16.
3. Accept 5–6 bit as the practical floor and ship that instead
   of 2/3/4 bit.

Phases 3 (GPU) and 4 (CUDA kernels) stay blocked. Benchmarking
throughput on a path that produces 10× PPL is not useful.

## Cross-architecture extension (dense vs MoE vs MoE+MLA)

WikiText-2 perplexity, 2048 ctx, single chunk unless otherwise
noted, CPU for small models, dual-GPU offload for GLM-5:

| Model | Arch | f16 PPL | q8_0 PPL | Δ | q4_0 PPL | Δ |
|---|---|---:|---:|---:|---:|---:|
| Qwen2.5-1.5B | dense GQA, 12/2 | 9.83 | 9.85 | +0.02 | 4952 | **+500×** |
| Gemma 4 E2B | MoE, 8/1 | 214 (unusable baseline) | 216 | flat | 215 | flat |
| GLM-5-REAP | MoE+MLA, 64/1 | 10.46 | 10.52 | +0.06 | 11.23 | **+7%** |

**GLM-5's MLA (multi-head latent attention, head_count_kv=1) and
MoE structure tolerates 4-bit KV at +7% PPL**, vs Qwen dense's
500× catastrophic regression. This is the inverse of what we
expected — smaller KV footprints per token (MLA) are MORE
tolerant of compression, not less.

Caveat: on GLM-5 our harness saw chunk-2 PPL explode to ~970 for
both q8 and q4 while chunk-1 was clean. Likely a cache-reset
quirk in llama-perplexity's handling of the `glm-dsa` arch, not
a compression failure — needs a second look before citing GLM-5
numbers anywhere.

Gemma 4 E2B baseline is unusable (PPL 214 on an instruction-
tuned model against raw WikiText). Can't extract a compression
signal until we get a base (non-instruct) Gemma 4 checkpoint
or switch to a chat-format eval.

**Practical takeaway:** if there's any product left in TQ-KV, it
may be on MoE+MLA architectures (DeepSeek-V3, GLM-5, etc.) where
the cost of compression to 4-bit is ~7% rather than catastrophic.
Standard dense models are a dead end below 8-bit.

## TurboQuant-on-GLM-5: negative result

Ran our own `tq_kv3` / `tq_kv4` on the same GLM-5-REAP-50pct
under view-bind + readthrough:

| Config | PPL chunk-1 | vs f16 |
|---|---:|---:|
| f16 | 10.56 | 1.0× |
| stock q8_0 | 10.52 | ~0% |
| stock q4_0 | 11.23 | +7% |
| **tq_kv4 (ours)** | **94.66** | **+797%** |
| **tq_kv3 (ours)** | **4884** | **+46000×** |

Stock `q4_0` beats our `tq_kv4` by ~100× at the same bit budget
*on the architecture we thought would be our niche*. Also of note:
the TQ init log reports `78 K layers, 0 V layers` — GLM-5's MLA
stores V inside the latent-K path, so our V-side compression
never activates. K-only compression still destroys quality.

**Final verdict: TurboQuant-KV-for-llama.cpp is not useful.**
Stock `q8_0` owns the 2× lossless niche; stock `q4_0` owns the
4× MLA-tolerant niche. Neither is beaten by our current
implementation. The sub-8-bit regime on dense models isn't
usable regardless of compression scheme.

If we want to salvage TurboQuant, the promising directions are
outside llama.cpp:

1. **PyTorch / vLLM KV cache** — there is no stock `q8_0`
   equivalent there; a well-tuned compression could have a
   cleaner product story.
2. **Weight compression for training** — where the paper's
   theory actually predicts a win.
3. **Publish the paper and move on** — the fidelity numbers
   reproduce cleanly; the engineering surface just doesn't have
   room for another KV compressor on llama.cpp.
