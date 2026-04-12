#!/usr/bin/env python3
"""Differential benchmark harness for TurboQuant KV cache.

Runs llama-cli several times against the same prompt with different
--cache-type-k / --cache-type-v settings, parses timings and memory
output, and prints a markdown table comparing them.

Today (Sprint 4c step 3c-1) the TQ backend stores a parallel tiered
cache alongside the fp16 backbone — so KV VRAM does not yet shrink,
but we *do* have evidence of:

  - observed compression ratio inside the tiered_cache
  - end-to-end token correctness (same sampled tokens as fp16)
  - per-validation round-trip cosine similarity (= 1.0 for hot
    tokens when LLAMA_TQ_VALIDATE=1)

Once step 3c-3 shrinks the fp16 backbone to just the hot window, the
"KV memory (MiB)" column below will start showing real savings. The
harness is wired to surface that automatically.

Example:
    python3 benchmark-tq-kv.py \\
        --llama-cli /path/to/build/bin/llama-cli \\
        --model     /path/to/qwen2.5-0.5b.gguf \\
        --prompt    "hi" --n 16 \\
        --hot-window 4 \\
        --configs f16 tq_kv3 tq_kv4 tq_kv2

Exits non-zero if any run crashed or sampled-token disagreement
exceeds --max-disagreement-frac (default: 0.1 = 10%).
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ------------------------------------------------------------------ #
# Parsing                                                             #
# ------------------------------------------------------------------ #

# Regexes for llama.cpp log lines of interest.
# The log format is stable across recent upstream versions; keep these
# permissive enough to survive minor changes.
RX_PROMPT_TPS = re.compile(r"prompt eval.*?(\d+\.\d+)\s*tokens per second", re.I)
RX_GEN_TPS    = re.compile(r"eval time.*?(\d+\.\d+)\s*tokens per second", re.I)
RX_KV_MIB     = re.compile(r"KV (?:cache|self) size\s*=\s*([0-9.]+)\s*MiB", re.I)
RX_KV_LINE    = re.compile(r"llama_kv_cache.*?K \(.*?\):\s*([0-9.]+)\s*MiB,\s*V \(.*?\):\s*([0-9.]+)\s*MiB", re.I)
RX_TQ_ALLOC   = re.compile(r"TurboQuant shadow caches allocated.*?hot_window=(\d+)")
RX_TQ_FLUSH   = re.compile(
    r"flushed \d+ tokens -> K=(\d+) \(hot=(\d+) cold=(\d+) ratio=([0-9.]+)x\)\s*"
    r"V=(\d+) \(hot=(\d+) cold=(\d+) ratio=([0-9.]+)x\)")
RX_VALIDATE   = re.compile(
    r"TQ validate ([KV]):\s*mean cos = ([0-9.]+),\s*min cos = ([0-9.]+)\s*over\s*(\d+)")
RX_VIEW_VAL   = re.compile(
    r"TQ view-validate ([KV]):\s*mean cos = ([0-9.]+),\s*min cos = ([0-9.]+)\s*over\s*(\d+)")
RX_COLD_VAL   = re.compile(
    r"TQ cold-validate ([KV]):\s*mean cos = ([0-9.]+),\s*min cos = ([0-9.]+)\s*over\s*(\d+)")
RX_GEN_TOK    = re.compile(r"next token:\s*\d+\s*'([^']*)'")


@dataclass
class RunResult:
    """Parsed metrics from one llama-cli run."""
    config:       str
    prompt_tps:   Optional[float] = None
    gen_tps:      Optional[float] = None
    kv_mib:       Optional[float] = None   # total KV (K+V) in MiB
    kv_k_mib:     Optional[float] = None
    kv_v_mib:     Optional[float] = None
    tq_hot_window: Optional[int] = None
    tq_ratio_last_k: Optional[float] = None
    tq_ratio_last_v: Optional[float] = None
    tq_cold_max_k: int = 0
    tq_cold_max_v: int = 0
    tq_validate_mean_cos_k: Optional[float] = None
    tq_validate_min_cos_k:  Optional[float] = None
    tq_validate_n_k:        int = 0
    tq_view_mean_cos_k:     Optional[float] = None
    tq_view_min_cos_k:      Optional[float] = None
    tq_view_n_k:            int = 0
    tq_cold_mean_cos_k:     Optional[float] = None
    tq_cold_min_cos_k:      Optional[float] = None
    tq_cold_n_k:            int = 0
    sampled_tokens: list[str] = field(default_factory=list)
    returncode:    int = 0
    stderr_tail:   str = ""


def parse_run(output: str, config: str, returncode: int) -> RunResult:
    r = RunResult(config=config, returncode=returncode)
    for line in output.splitlines():
        m = RX_PROMPT_TPS.search(line)
        if m: r.prompt_tps = float(m.group(1))
        m = RX_GEN_TPS.search(line)
        if m: r.gen_tps = float(m.group(1))
        m = RX_KV_MIB.search(line)
        if m: r.kv_mib = float(m.group(1))
        m = RX_KV_LINE.search(line)
        if m:
            r.kv_k_mib = float(m.group(1))
            r.kv_v_mib = float(m.group(2))
            if r.kv_mib is None:
                r.kv_mib = r.kv_k_mib + r.kv_v_mib
        m = RX_TQ_ALLOC.search(line)
        if m: r.tq_hot_window = int(m.group(1))
        m = RX_TQ_FLUSH.search(line)
        if m:
            r.tq_ratio_last_k = float(m.group(4))
            r.tq_ratio_last_v = float(m.group(8))
            r.tq_cold_max_k = max(r.tq_cold_max_k, int(m.group(3)))
            r.tq_cold_max_v = max(r.tq_cold_max_v, int(m.group(7)))
        m = RX_VALIDATE.search(line)
        if m and m.group(1) == "K":
            # Keep the *last* report — earlier reports get reset.
            r.tq_validate_mean_cos_k = float(m.group(2))
            r.tq_validate_min_cos_k  = float(m.group(3))
            r.tq_validate_n_k        = int(m.group(4))
        m = RX_VIEW_VAL.search(line)
        if m and m.group(1) == "K":
            r.tq_view_mean_cos_k = float(m.group(2))
            r.tq_view_min_cos_k  = float(m.group(3))
            r.tq_view_n_k        = int(m.group(4))
        m = RX_COLD_VAL.search(line)
        if m and m.group(1) == "K":
            r.tq_cold_mean_cos_k = float(m.group(2))
            r.tq_cold_min_cos_k  = float(m.group(3))
            r.tq_cold_n_k        = int(m.group(4))
        m = RX_GEN_TOK.search(line)
        if m: r.sampled_tokens.append(m.group(1))

    # Keep a short stderr tail for triage on failure.
    if returncode != 0:
        r.stderr_tail = "\n".join(output.splitlines()[-20:])
    return r


# ------------------------------------------------------------------ #
# Running                                                             #
# ------------------------------------------------------------------ #

def run_one(llama_cli: Path,
            model: Path,
            cache_type: str,
            prompt: str,
            n_predict: int,
            hot_window: Optional[int],
            threads: int,
            ngl: int,
            timeout_s: int,
            validate: bool,
            view_validate: bool = False,
            cold_validate: bool = False) -> RunResult:
    cmd = [
        str(llama_cli),
        "-m", str(model),
        "-p", prompt,
        "-n", str(n_predict),
        "--no-warmup",
        "-no-cnv",              # exit after n tokens — no chat shell
        "--simple-io",
        "-t", str(threads),
        "-ngl", str(ngl),
        "--seed", "42",
        "--temp", "0",
        "-v",
        "--cache-type-k", cache_type,
        "--cache-type-v", cache_type,
    ]
    env = os.environ.copy()
    if hot_window is not None:
        env["LLAMA_TQ_HOT_WINDOW"] = str(hot_window)
    if validate:
        env["LLAMA_TQ_VALIDATE"] = "1"
    if view_validate:
        env["LLAMA_TQ_VIEW_VALIDATE"] = "1"
    if cold_validate:
        env["LLAMA_TQ_COLD_VALIDATE"] = "1"

    # llama-cli in this tree has a quirk: after `-n` tokens it emits
    # the perf stats and a short idle-prompt loop that can print for a
    # long time. We kill it early and cap captured bytes. The data we
    # care about — timings, TQ allocation, flush logs, perf table — is
    # all emitted BEFORE the idle loop starts, so a 256 KB cap is safe.
    CAP_BYTES = 256 * 1024
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, env=env, text=True,
            bufsize=1,
        )
        assert proc.stdout is not None  # PIPE was requested
        stdout = proc.stdout
        chunks: list[str] = []
        total = 0
        deadline = time.time() + timeout_s
        last_data_t = time.time()
        import select
        # Drain without deadlocking on the idle loop.
        while True:
            if proc.poll() is not None:
                # Read anything left.
                remaining = stdout.read()
                if remaining:
                    chunks.append(remaining)
                break
            remain = max(0.0, deadline - time.time())
            r, _, _ = select.select([stdout], [], [], min(remain, 0.5))
            if r:
                line = stdout.readline()
                if not line:
                    break
                chunks.append(line)
                total += len(line)
                last_data_t = time.time()
                # Early-exit signal: once we see the llama_perf final line
                # AND a subsequent idle "> " prompt, the useful data is in.
                if total >= CAP_BYTES:
                    break
                if "llama_perf_context_print" in line and total > 2000:
                    # Collect a tiny bit more then stop.
                    pass
            if time.time() >= deadline:
                break
            # If we see the idle-loop prompts landing, cut short.
            if (time.time() - last_data_t) > 5.0:
                break
        proc.kill()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        combined = "".join(chunks)[:CAP_BYTES]
        rc = proc.returncode if proc.returncode is not None else 0
        # rc=-9 (killed by us) is expected; treat as success for parsing.
        parsed_rc = 0 if rc in (0, -9, None) else rc
        return parse_run(combined, cache_type, parsed_rc)
    except Exception as e:  # pragma: no cover
        return RunResult(config=cache_type, returncode=125,
                         stderr_tail=f"UNEXPECTED: {e}")


# ------------------------------------------------------------------ #
# Reporting                                                           #
# ------------------------------------------------------------------ #

def fmt(x, fmt_="{:.2f}"):
    return fmt_.format(x) if x is not None else "—"


def print_markdown(results: list[RunResult], baseline: str) -> None:
    bl = next((r for r in results if r.config == baseline), None)
    print(f"\n## TurboQuant KV differential benchmark (baseline = `{baseline}`)\n")
    print("| Config | Prompt tok/s | Gen tok/s | KV MiB | Δ Gen % | Observed compression | Push round-trip cos | View vs fp16 cos | Cold vs fp16 cos |")
    print("|---|---:|---:|---:|---:|---|---|---|---|")
    for r in results:
        ratio = "—"
        if r.tq_ratio_last_k is not None:
            ratio = f"K={r.tq_ratio_last_k:.2f}x V={r.tq_ratio_last_v:.2f}x (cold_max={r.tq_cold_max_k})"
        val = "—"
        if r.tq_validate_n_k > 0:
            val = f"{r.tq_validate_mean_cos_k:.6f}/{r.tq_validate_min_cos_k:.6f} (n={r.tq_validate_n_k})"
        view = "—"
        if r.tq_view_n_k > 0:
            view = f"{r.tq_view_mean_cos_k:.6f}/{r.tq_view_min_cos_k:.6f} (n={r.tq_view_n_k})"
        cold = "—"
        if r.tq_cold_n_k > 0:
            cold = f"{r.tq_cold_mean_cos_k:.4f}/{r.tq_cold_min_cos_k:.4f} (n={r.tq_cold_n_k})"
        delta = "—"
        if bl and bl.gen_tps and r.gen_tps:
            delta = f"{(r.gen_tps - bl.gen_tps) / bl.gen_tps * 100:+.2f}%"
        print(f"| `{r.config}` | {fmt(r.prompt_tps)} | {fmt(r.gen_tps)} | "
              f"{fmt(r.kv_mib)} | {delta} | {ratio} | {val} | {view} | {cold} |")
    print()

    # Sampled-token agreement
    if bl is None:
        return
    print("### Sampled-token agreement vs baseline")
    print()
    print("| Config | First generated tokens | Disagreement |")
    print("|---|---|---:|")
    for r in results:
        if r.config == baseline:
            joined = " ".join(repr(t) for t in r.sampled_tokens[:8])
            print(f"| `{r.config}` (baseline) | {joined} | — |")
            continue
        dis = diff_fraction(bl.sampled_tokens, r.sampled_tokens)
        joined = " ".join(repr(t) for t in r.sampled_tokens[:8])
        print(f"| `{r.config}` | {joined} | {dis*100:.1f}% |")
    print()


def diff_fraction(a: list[str], b: list[str]) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    mismatches = sum(1 for i in range(n) if a[i] != b[i])
    return mismatches / n


# ------------------------------------------------------------------ #
# Entry point                                                         #
# ------------------------------------------------------------------ #

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llama-cli", type=Path, required=True)
    ap.add_argument("--model",     type=Path, required=True)
    ap.add_argument("--prompt",    default="hi")
    ap.add_argument("--n",         type=int, default=16)
    ap.add_argument("--threads",   type=int, default=4)
    ap.add_argument("--ngl",       type=int, default=0)
    ap.add_argument("--hot-window", type=int, default=None,
                    help="override LLAMA_TQ_HOT_WINDOW")
    ap.add_argument("--configs",   nargs="+", default=["f16", "tq_kv3"],
                    help="cache-type-k/v configs to compare")
    ap.add_argument("--baseline",  default="f16",
                    help="config to compute Δ against")
    ap.add_argument("--timeout-s", type=int, default=90)
    ap.add_argument("--validate",  action="store_true",
                    help="enable LLAMA_TQ_VALIDATE for TQ configs")
    ap.add_argument("--view-validate", action="store_true",
                    help="enable LLAMA_TQ_VIEW_VALIDATE for TQ configs")
    ap.add_argument("--cold-validate", action="store_true",
                    help="enable LLAMA_TQ_COLD_VALIDATE for TQ configs")
    ap.add_argument("--max-disagreement-frac", type=float, default=0.1)
    args = ap.parse_args()

    if not args.llama_cli.is_file():
        print(f"error: llama-cli not found at {args.llama_cli}", file=sys.stderr)
        return 2
    if not args.model.is_file():
        print(f"error: model not found at {args.model}", file=sys.stderr)
        return 2

    results = []
    for cfg in args.configs:
        print(f"[benchmark] running {cfg}...", file=sys.stderr, flush=True)
        r = run_one(args.llama_cli, args.model, cfg, args.prompt, args.n,
                    args.hot_window, args.threads, args.ngl,
                    args.timeout_s, args.validate, args.view_validate,
                    args.cold_validate)
        results.append(r)
        if r.returncode != 0:
            print(f"[benchmark] {cfg} FAILED (rc={r.returncode}):", file=sys.stderr)
            print(r.stderr_tail, file=sys.stderr)

    print_markdown(results, args.baseline)

    # Exit status: non-zero if any crash or excessive token disagreement.
    rc = 0
    bl = next((r for r in results if r.config == args.baseline), None)
    for r in results:
        if r.returncode != 0:
            rc = 1
            break
        if bl is None or r.config == args.baseline:
            continue
        dis = diff_fraction(bl.sampled_tokens, r.sampled_tokens)
        if dis > args.max_disagreement_frac:
            print(f"[benchmark] {r.config}: {dis*100:.1f}% token "
                  f"disagreement exceeds --max-disagreement-frac "
                  f"({args.max_disagreement_frac*100:.0f}%)", file=sys.stderr)
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
