#!/usr/bin/env python3
"""
eval_diversity.py
=================
Evaluate NeuralSAT-Div (use_diversity_sampling) on ACAS Xu vnncomp2026
benchmarks.

For each instance:
  1.  Run Verifier.sample_violations() to collect k diverse violation witnesses.
  2.  Verify every returned sample is a REAL violation (cs @ net(x) <= rhs).
  3.  Compute diversity metrics on the solution space:
        - Mean pairwise L2 distance  (higher = more spread-out)
        - Min pairwise L2 distance   (higher = no near-duplicate witnesses)
        - Coverage radius (max of min-dist to nearest neighbour)
        - Normalised versions (divided by input-box diameter)

Usage
-----
    conda run -n neuralsat python eval_diversity.py [options]

    --benchmark_dir   path to vnncomp2026_benchmarks/benchmarks/acasxu_2023/2.0
    --num_instances   how many instances to try  (default: 5)
    --k               target sample budget per instance  (default: 10)
    --timeout         per-instance timeout in seconds    (default: 60)
    --device          cpu / cuda                         (default: cpu)
    --batch           NeuralSAT branch batch size        (default: 200)
    --filter_prop     only run instances whose vnnlib path contains this string
                      e.g. "prop_2"
    --seed            random seed                        (default: 0)
"""

from __future__ import annotations
import argparse
import csv
import gzip
import itertools
import os
import random
import shutil
import sys
import tempfile
import time

import numpy as np
import torch

# ── NeuralSAT source on the path ────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR    = os.path.join(SCRIPT_DIR, 'src')
sys.path.insert(0, SRC_DIR)

from helper.network.read_onnx import parse_onnx
from helper.spec.objective    import parse_vnnlib, DnfObjectives
from helper.misc.result       import ReturnStatus
from helper.misc.logger       import logger
from verifier.verifier        import Verifier
from setting                  import Settings

import logging
logger.setLevel(logging.WARNING)  # keep NeuralSAT quiet during batch eval

# ── ANSI colour helpers ──────────────────────────────────────────────────────
RESET  = '\033[0m'
BOLD   = '\033[1m'
GREEN  = '\033[32m'
RED    = '\033[31m'
YELLOW = '\033[33m'
CYAN   = '\033[36m'
DIM    = '\033[2m'

def _c(text, *codes): return ''.join(codes) + str(text) + RESET
def _bar(frac, width=20, fill='█', empty='░'):
    n = round(frac * width)
    return fill * n + empty * (width - n)


# ── gz-aware file loading ────────────────────────────────────────────────────

def _resolve_and_decompress(rel_path: str, bench_dir: str, suffix: str) -> tuple[str, bool]:
    """
    Resolve rel_path relative to bench_dir, transparently handling .gz variants.
    Returns (local_path, needs_cleanup) where local_path is a plain (possibly
    temp-decompressed) file ready to pass to parsers.
    """
    full = os.path.join(bench_dir, rel_path)
    # Try as-is
    if os.path.exists(full):
        return full, False
    # Try with .gz appended
    gz = full + '.gz'
    if os.path.exists(gz):
        fd, tmp = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        with gzip.open(gz, 'rb') as fi, open(tmp, 'wb') as fo:
            shutil.copyfileobj(fi, fo)
        return tmp, True
    raise FileNotFoundError(f"Cannot find {full} or {gz}")


# ── Violation checker ────────────────────────────────────────────────────────

def _is_real_violation(
    net:    torch.nn.Module,
    sample: torch.Tensor,
    obj,                       # the raw objective from DnfObjectives
    device: str,
) -> bool:
    """Return True iff sample ∈ [lb, ub] AND cs @ net(sample) <= rhs for
    at least one disjunct in obj."""
    net.eval()
    lower = obj.lower_bounds.view(-1).to(device)
    upper = obj.upper_bounds.view(-1).to(device)
    x = sample.to(device).flatten()

    # Input-box check
    if not (torch.all(x >= lower - 1e-5) and torch.all(x <= upper + 1e-5)):
        return False

    x_shaped = sample.to(device).reshape(1, *sample.shape[1:]) if sample.dim() > 1 else sample.to(device).unsqueeze(0)
    with torch.no_grad():
        y = net(x_shaped).reshape(1, -1)  # (1, n_out)

    cs  = obj.cs.to(device)   # (n_disjunct, n_out) or (n_out,)
    rhs = obj.rhs.to(device)  # (n_disjunct,) or scalar

    if cs.dim() == 1:
        cs  = cs.unsqueeze(0)
        rhs = rhs.unsqueeze(0)

    # cs @ y^T  → (n_disjunct,)
    vals = (cs @ y.t()).squeeze(-1)
    if (vals <= rhs + 1e-5).all():
        return True
    return False


# ── Diversity metrics ────────────────────────────────────────────────────────

def _diversity_metrics(samples: list[torch.Tensor], lb: torch.Tensor, ub: torch.Tensor) -> dict:
    """Compute pairwise-L2 diversity statistics on a list of flat tensors."""
    n = len(samples)
    if n < 2:
        return {
            'n': n,
            'mean_pairwise_l2':  float('nan'),
            'min_pairwise_l2':   float('nan'),
            'coverage_radius':   float('nan'),
            'mean_pairwise_l2_norm': float('nan'),
            'min_pairwise_l2_norm':  float('nan'),
            'coverage_radius_norm':  float('nan'),
        }

    flat = torch.stack([s.cpu().flatten() for s in samples])  # (n, d)
    diameter = (ub.cpu().flatten() - lb.cpu().flatten()).norm().item()
    diameter = max(diameter, 1e-9)

    # All pairwise distances
    dists = []
    min_dists = []  # for each sample, distance to nearest other sample
    for i in range(n):
        row_dists = []
        for j in range(n):
            if i != j:
                d = (flat[i] - flat[j]).norm().item()
                dists.append(d)
                row_dists.append(d)
        min_dists.append(min(row_dists))

    mean_pw  = float(np.mean(dists))
    min_pw   = float(np.min(dists))
    coverage = float(np.max(min_dists))   # largest gap — lower = better coverage

    return {
        'n': n,
        'mean_pairwise_l2':  mean_pw,
        'min_pairwise_l2':   min_pw,
        'coverage_radius':   coverage,
        'mean_pairwise_l2_norm': mean_pw  / diameter,
        'min_pairwise_l2_norm':  min_pw   / diameter,
        'coverage_radius_norm':  coverage / diameter,
    }


# ── Per-instance runner ──────────────────────────────────────────────────────

def _run_instance(
    onnx_gz:  str,
    spec_gz:  str,
    args,
    bench_dir: str,
) -> dict:
    """Load one (network, spec) pair, sample violations, verify, compute metrics."""

    result = {
        'onnx': onnx_gz,
        'spec': spec_gz,
        'status': 'ERROR',
        'elapsed': 0.0,
        'n_returned': 0,
        'n_verified': 0,
        'all_valid': False,
        'metrics': {},
        'error': '',
    }

    tmp_onnx, rm_onnx = _resolve_and_decompress(onnx_gz, bench_dir, '.onnx')
    tmp_spec, rm_spec  = _resolve_and_decompress(spec_gz,  bench_dir, '.vnnlib')
    try:
        t0 = time.time()

        # ── Load network ──────────────────────────────────────────────────
        model, input_shape, output_shape = parse_onnx(tmp_onnx, None, None)
        input_shape  = tuple(input_shape)
        output_shape = tuple(output_shape)
        model.eval()

        # ── Load specification ────────────────────────────────────────────
        objectives = parse_vnnlib(tmp_spec, input_shape)

        # We need to keep a raw reference for violation checking later.
        # DnfObjectives is consumed during verify so we snapshot the first
        # objective's bounds/cs/rhs before calling sample_violations.
        raw_obj = objectives   # (we only need lb/ub/cs/rhs for checking)

        # ── Configure NeuralSAT-Div ───────────────────────────────────────
        Settings.setup(None)   # restore defaults
        Settings.use_attack         = True
        Settings.use_restart        = True
        Settings.use_mip_tightening = False   # keep fast for demo

        verifier = Verifier(
            net=model,
            input_shape=input_shape,
            batch=args.batch,
            device=args.device,
        )

        # ── sample_violations ────────────────────────────────────────────
        import copy
        samples = verifier.sample_violations(
            dnf_objectives=copy.deepcopy(objectives),
            k=args.k,
            timeout=args.timeout,
        )

        elapsed = time.time() - t0
        result['elapsed'] = elapsed
        result['n_returned'] = len(samples)

        if not samples:
            result['status'] = 'NO_SAMPLES'
            return result

        # ── Verification pass ─────────────────────────────────────────────
        # Re-parse objectives so we have a fresh copy for checking
        obj_check = parse_vnnlib(tmp_spec, input_shape)
        # We use the first objective (most specs here have a single disjunct)
        check_obj = obj_check   # pass full objectives to helper

        n_valid = 0
        for s in samples:
            # Try against all disjuncts
            valid = False
            for i in range(len(check_obj.lower_bounds)):
                class _Disjunct: pass
                d = _Disjunct()
                d.lower_bounds = check_obj.lower_bounds[i:i+1]
                d.upper_bounds = check_obj.upper_bounds[i:i+1]
                cs_ = check_obj.cs
                rhs_ = check_obj.rhs
                if isinstance(cs_, torch.Tensor) and cs_.dim() == 3:
                    d.cs  = cs_[i]
                    d.rhs = rhs_[i]
                elif isinstance(cs_, torch.Tensor) and cs_.dim() == 2:
                    d.cs  = cs_
                    d.rhs = rhs_
                else:
                    d.cs  = cs_[i] if hasattr(cs_, '__getitem__') else cs_
                    d.rhs = rhs_[i] if hasattr(rhs_, '__getitem__') else rhs_
                if _is_real_violation(model, s, d, args.device):
                    valid = True
                    break
            if valid:
                n_valid += 1

        result['n_verified'] = n_valid
        result['all_valid']  = (n_valid == len(samples))
        result['status']     = 'OK'

        # ── Diversity metrics ─────────────────────────────────────────────
        lb = raw_obj.lower_bounds[0]
        ub = raw_obj.upper_bounds[0]
        result['metrics'] = _diversity_metrics(samples, lb, ub)

    except Exception as exc:
        import traceback
        result['error']  = str(exc)
        result['status'] = 'ERROR'
        if os.environ.get('DIV_DEBUG'):
            traceback.print_exc()
    finally:
        if rm_onnx: os.unlink(tmp_onnx)
        if rm_spec:  os.unlink(tmp_spec)

    return result


# ── Pretty printer ───────────────────────────────────────────────────────────

def _print_instance_result(r: dict, idx: int, total: int) -> None:
    onnx_short = os.path.basename(r['onnx']).replace('.onnx.gz', '')
    spec_short  = os.path.basename(r['spec']).replace('.vnnlib.gz', '').replace('.vnnlib', '')
    header = f"[{idx:2d}/{total}]  {_c(onnx_short, BOLD, CYAN)}  |  prop: {_c(spec_short, BOLD)}"
    print(header)

    status = r['status']
    if status == 'ERROR':
        print(f"  {_c('ERROR', BOLD, RED)}  {r['error'][:80]}")
        return
    if status == 'NO_SAMPLES':
        print(f"  {_c('no violation witnesses found', YELLOW)}  ({r['elapsed']:.1f}s)")
        return

    # Sample validity
    n_ret  = r['n_returned']
    n_ver  = r['n_verified']
    all_ok = r['all_valid']
    validity_bar = _bar(n_ver / n_ret if n_ret else 0)
    valid_label  = _c('ALL VALID ✓', BOLD, GREEN) if all_ok else _c(f'{n_ver}/{n_ret} valid ✗', BOLD, RED)
    print(f"  Samples returned : {_c(n_ret, BOLD)}   {valid_label}")
    print(f"  Validity         : [{_c(validity_bar, GREEN if all_ok else RED)}]  {n_ver}/{n_ret}")

    m = r['metrics']
    if m and m['n'] >= 2:
        print(f"  Elapsed          : {r['elapsed']:.1f}s")
        print(f"  ── Diversity metrics (input space) ────────────────────────────")
        print(f"     Mean pairwise L2   : {m['mean_pairwise_l2']:.4f}  "
              f"(norm: {m['mean_pairwise_l2_norm']:.4f})")
        print(f"     Min  pairwise L2   : {m['min_pairwise_l2']:.4f}  "
              f"(norm: {m['min_pairwise_l2_norm']:.4f})")
        print(f"     Coverage radius    : {m['coverage_radius']:.4f}  "
              f"(norm: {m['coverage_radius_norm']:.4f})")
        # Diversity bar (normalised mean pairwise L2 as proxy)
        div_frac = min(m['mean_pairwise_l2_norm'], 1.0)
        div_bar  = _bar(div_frac)
        colour   = GREEN if div_frac > 0.2 else (YELLOW if div_frac > 0.05 else RED)
        print(f"     Diversity score    : [{_c(div_bar, colour)}]  {div_frac:.3f}")
    else:
        print(f"  Elapsed          : {r['elapsed']:.1f}s")
        print(f"  {_c('Only 1 sample — no pairwise diversity to report', DIM)}")

    print()


def _print_summary(results: list[dict], args) -> None:
    print(_c("═" * 72, BOLD))
    print(_c("  SUMMARY  —  NeuralSAT-Div on ACAS Xu", BOLD, CYAN))
    print(_c("═" * 72, BOLD))

    total       = len(results)
    n_ok        = sum(1 for r in results if r['status'] == 'OK')
    n_no_sample = sum(1 for r in results if r['status'] == 'NO_SAMPLES')
    n_error     = sum(1 for r in results if r['status'] == 'ERROR')
    n_all_valid = sum(1 for r in results if r['all_valid'])

    print(f"  Instances tested         : {total}")
    print(f"  Instances with samples   : {n_ok}")
    print(f"  Instances no witnesses   : {n_no_sample}")
    print(f"  Instances with errors    : {n_error}")
    print(f"  Instances all-valid      : {_c(n_all_valid, BOLD, GREEN if n_all_valid==n_ok else RED)}/{n_ok}")

    ok_results = [r for r in results if r['status'] == 'OK']
    if ok_results:
        all_returned  = [r['n_returned']  for r in ok_results]
        all_verified  = [r['n_verified']  for r in ok_results]
        all_elapsed   = [r['elapsed']     for r in ok_results]
        all_mpw       = [r['metrics']['mean_pairwise_l2_norm']
                         for r in ok_results if r['metrics'].get('n', 0) >= 2]
        all_minpw     = [r['metrics']['min_pairwise_l2_norm']
                         for r in ok_results if r['metrics'].get('n', 0) >= 2]
        all_cov       = [r['metrics']['coverage_radius_norm']
                         for r in ok_results if r['metrics'].get('n', 0) >= 2]

        print()
        print(f"  Avg samples returned     : {np.mean(all_returned):.2f}  (target k={args.k})")
        print(f"  Avg verified valid       : {np.mean(all_verified):.2f}")
        print(f"  Avg elapsed (s)          : {np.mean(all_elapsed):.1f}")

        if all_mpw:
            print()
            print(f"  ── Aggregate diversity (normalised, instances with ≥2 samples) ──")
            print(f"     Mean pairwise L2 norm : "
                  f"avg={np.mean(all_mpw):.4f}  "
                  f"min={np.min(all_mpw):.4f}  "
                  f"max={np.max(all_mpw):.4f}")
            print(f"     Min  pairwise L2 norm : "
                  f"avg={np.mean(all_minpw):.4f}  "
                  f"min={np.min(all_minpw):.4f}  "
                  f"max={np.max(all_minpw):.4f}")
            print(f"     Coverage radius norm  : "
                  f"avg={np.mean(all_cov):.4f}  "
                  f"min={np.min(all_cov):.4f}  "
                  f"max={np.max(all_cov):.4f}")

    print(_c("═" * 72, BOLD))


# ── Main ─────────────────────────────────────────────────────────────────────

def _parse_args():
    default_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    p = argparse.ArgumentParser(description='NeuralSAT-Div evaluation on ACAS Xu')
    p.add_argument('--benchmark_dir', type=str,
                   default=os.path.join(SCRIPT_DIR,
                       'vnncomp2026_benchmarks/benchmarks/acasxu_2023/1.0'),
                   help='Path to acasxu_2023 directory (default: 1.0)')
    p.add_argument('--num_instances', type=int, default=5,
                   help='Number of benchmark instances to run (default: 5)')
    p.add_argument('--k',          type=int,   default=10,
                   help='Target violation witnesses per instance (default: 10)')
    p.add_argument('--timeout',    type=float, default=60.0,
                   help='Per-instance timeout in seconds (default: 60)')
    p.add_argument('--device',     type=str,   default=default_device)
    p.add_argument('--batch',      type=int,   default=200,
                   help='NeuralSAT branch batch size (default: 200)')
    p.add_argument('--filter_prop', type=str,  default=None,
                   help='Only instances whose vnnlib path contains this string, e.g. "prop_2"')
    p.add_argument('--seed',       type=int,   default=0)
    p.add_argument('--stochastic_prob', type=float, default=0.3,
                   help='Stochastic DPLL branching probability (default: 0.3)')
    p.add_argument('--bam_n',      type=int,   default=5,
                   help='PAIS polytope samples per activation region (default: 5)')
    return p.parse_args()


def main():
    args = _parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    bench_dir = args.benchmark_dir
    csv_path  = os.path.join(bench_dir, 'instances.csv')
    assert os.path.exists(csv_path), f"instances.csv not found at {csv_path}"

    # ── Load instance list ────────────────────────────────────────────────
    with open(csv_path) as f:
        rows = list(csv.reader(f))
    # format: onnx_path, vnnlib_path, timeout
    if args.filter_prop:
        rows = [r for r in rows if args.filter_prop in r[1]]

    rows = rows[:args.num_instances]

    # ── Banner ────────────────────────────────────────────────────────────
    print()
    print(_c("╔" + "═"*70 + "╗", BOLD, CYAN))
    print(_c("║  NeuralSAT-Div  ·  ACAS Xu Diversity Evaluation" + " "*22 + "║", BOLD, CYAN))
    print(_c("╚" + "═"*70 + "╝", BOLD, CYAN))
    print(f"  benchmark_dir  : {bench_dir}")
    print(f"  instances      : {len(rows)}")
    print(f"  k (target)     : {args.k}")
    print(f"  timeout        : {args.timeout}s per instance")
    print(f"  device         : {args.device}")
    print(f"  stoch_prob     : {args.stochastic_prob}")
    print(f"  bam_n          : {args.bam_n}")
    print()

    # Set diversity-sampling hyper-parameters globally
    Settings.diversity_stochastic_prob = args.stochastic_prob
    Settings.diversity_bam_n           = args.bam_n

    results = []
    for idx, row in enumerate(rows, 1):
        onnx_rel, spec_rel = row[0], row[1]
        print(_c(f"{'─'*72}", DIM))
        print(f"  Running {idx}/{len(rows)} …")
        r = _run_instance(onnx_rel, spec_rel, args, bench_dir)
        results.append(r)
        _print_instance_result(r, idx, len(rows))

    print()
    _print_summary(results, args)
    print()


if __name__ == '__main__':
    main()
