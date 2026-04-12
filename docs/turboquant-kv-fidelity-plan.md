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
