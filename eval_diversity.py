#!/usr/bin/env python3
"""
eval_diversity.py  —  NeuralSAT-Div evaluation on ACAS Xu
==========================================================

Two-phase workflow
------------------
Phase 1  TRIAGE  — Run normal Verifier.verify() with a short timeout (default
          1 s) across ALL instances in instances.csv.  Keep only those that
          return SAT quickly (i.e. the property is genuinely violated and the
          solver finds a counterexample fast).  Instances with "unsupported
          layer" warnings are automatically skipped.

Phase 2  DIVERSITY  — On the confirmed-SAT subset, run
          Verifier.sample_violations() (use_diversity_sampling=True) and
          report:
            • Whether every returned sample is a REAL violation
            • Mean / min pairwise L2 distance in input space
            • Coverage radius
            • Normalised versions of the above

Usage
-----
  conda run -n neuralsat python eval_diversity.py [options]

  # Full auto-triage then diversity sampling
  python eval_diversity.py --triage_timeout 1 --k 10 --timeout 60

  # Just run triage to see which instances are easy SAT
  python eval_diversity.py --triage_only

  # Skip triage, pass pre-known SAT instances explicitly
  python eval_diversity.py --sat_instances_file sat_instances.csv --k 10
"""

from __future__ import annotations
import argparse
import copy
import csv
import gzip
import os
import random
import shutil
import sys
import tempfile
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch

# ── NeuralSAT source on the path ────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR    = os.path.join(SCRIPT_DIR, 'src')
sys.path.insert(0, SRC_DIR)

from helper.network.read_onnx import parse_onnx
from helper.spec.objective    import parse_vnnlib
from helper.misc.result       import ReturnStatus
from helper.misc.logger       import logger
from verifier.verifier        import Verifier
from setting                  import Settings

import logging
logger.setLevel(logging.WARNING)

# ── ANSI helpers ─────────────────────────────────────────────────────────────
RESET  = '\033[0m';  BOLD   = '\033[1m';  DIM    = '\033[2m'
GREEN  = '\033[32m'; RED    = '\033[31m'; YELLOW = '\033[33m'
CYAN   = '\033[36m'; BLUE   = '\033[34m'; MAGENTA= '\033[35m'

def _c(*args):
    """_c(text, code1, code2, …)  — wrap text in ANSI codes."""
    *codes, text = args[::-1]
    return ''.join(codes[::-1]) + str(text) + RESET

def _bar(frac, width=24, fill='█', empty='░'):
    n = max(0, min(width, round(frac * width)))
    return fill * n + empty * (width - n)


# ── gz-transparent file resolution ───────────────────────────────────────────

def _resolve(rel_path: str, bench_dir: str, suffix: str):
    """Return (local_path, needs_cleanup).
    Tries rel_path as-is, then rel_path+'.gz' (decompressed to a tmp file)."""
    full = os.path.join(bench_dir, rel_path)
    if os.path.exists(full):
        return full, False
    gz = full + '.gz'
    if os.path.exists(gz):
        fd, tmp = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        with gzip.open(gz, 'rb') as fi, open(tmp, 'wb') as fo:
            shutil.copyfileobj(fi, fo)
        return tmp, True
    raise FileNotFoundError(f'Not found: {full}  (nor {gz})')


def _load_instance(onnx_rel, spec_rel, bench_dir):
    """Return (model, input_shape, output_shape, objectives, cleanup_fn)."""
    tmp_onnx, rm_onnx = _resolve(onnx_rel, bench_dir, '.onnx')
    tmp_spec,  rm_spec  = _resolve(spec_rel,  bench_dir, '.vnnlib')
    model, input_shape, output_shape = parse_onnx(tmp_onnx, None, None)
    input_shape  = tuple(input_shape)
    output_shape = tuple(output_shape)
    model.eval()
    objectives = parse_vnnlib(tmp_spec, input_shape)

    def _cleanup():
        if rm_onnx: os.unlink(tmp_onnx)
        if rm_spec:  os.unlink(tmp_spec)

    return model, input_shape, output_shape, objectives, _cleanup


# ── Violation checker ─────────────────────────────────────────────────────────

def _is_real_violation(net, sample: torch.Tensor, obj, device: str) -> bool:
    """True iff sample in [lb,ub] AND cs @ net(x) <= rhs (any disjunct)."""
    lb = obj.lower_bounds.view(-1).to(device)
    ub = obj.upper_bounds.view(-1).to(device)
    x  = sample.to(device).flatten()
    if not (torch.all(x >= lb - 1e-5) and torch.all(x <= ub + 1e-5)):
        return False

    x_in = x.reshape(1, *sample.shape[1:]) if sample.dim() > 1 else x.unsqueeze(0)
    with torch.no_grad():
        y = net(x_in).reshape(1, -1)

    cs  = obj.cs.to(device)
    rhs = obj.rhs.to(device)
    if cs.dim() == 1:
        cs = cs.unsqueeze(0); rhs = rhs.unsqueeze(0)

    vals = (cs @ y.t()).squeeze(-1)
    return bool((vals <= rhs + 1e-5).all())


def _check_samples(net, samples, check_obj, device) -> tuple[int, int]:
    """Return (n_returned, n_verified)."""
    n_ret = len(samples)
    n_ok  = 0
    n_dis = len(check_obj.lower_bounds)
    for s in samples:
        for i in range(n_dis):
            class D: pass
            d = D()
            d.lower_bounds = check_obj.lower_bounds[i:i+1]
            d.upper_bounds = check_obj.upper_bounds[i:i+1]
            cs_  = check_obj.cs
            rhs_ = check_obj.rhs
            if isinstance(cs_, torch.Tensor) and cs_.dim() == 3:
                d.cs = cs_[i]; d.rhs = rhs_[i]
            else:
                d.cs = cs_;    d.rhs = rhs_
            if _is_real_violation(net, s, d, device):
                n_ok += 1; break
    return n_ret, n_ok


# ── Diversity metrics ─────────────────────────────────────────────────────────

def _diversity(samples: list[torch.Tensor], lb: torch.Tensor,
               ub: torch.Tensor) -> dict:
    n = len(samples)
    base = {'n': n, 'mean_pw': float('nan'), 'min_pw': float('nan'),
            'cov': float('nan'), 'mean_pw_n': float('nan'),
            'min_pw_n': float('nan'), 'cov_n': float('nan')}
    if n < 2:
        return base

    flat     = torch.stack([s.cpu().flatten() for s in samples])
    diameter = (ub.cpu().flatten() - lb.cpu().flatten()).norm().item()
    diameter = max(diameter, 1e-9)

    dists, min_d = [], []
    for i in range(n):
        row = [(flat[i] - flat[j]).norm().item() for j in range(n) if j != i]
        dists.extend(row)
        min_d.append(min(row))

    mpw = float(np.mean(dists))
    mnpw = float(np.min(dists))
    cov  = float(np.max(min_d))

    return {'n': n,
            'mean_pw': mpw, 'min_pw': mnpw, 'cov': cov,
            'mean_pw_n': mpw/diameter, 'min_pw_n': mnpw/diameter,
            'cov_n': cov/diameter}


# ── Phase 1: Triage ───────────────────────────────────────────────────────────

def _triage_instance(onnx_rel, spec_rel, bench_dir, device, batch,
                     triage_timeout) -> dict:
    """Run normal verify() with a short timeout. Return triage result dict."""
    res = {'onnx': onnx_rel, 'spec': spec_rel,
           'status': 'UNSAT', 'elapsed': 0.0, 'error': ''}
    cleanup = None
    try:
        model, input_shape, output_shape, objectives, cleanup = \
            _load_instance(onnx_rel, spec_rel, bench_dir)

        Settings.setup(None)
        Settings.use_attack         = True
        Settings.use_restart        = False
        Settings.use_mip_tightening = False
        Settings.use_mip_verify     = False
        Settings.use_diversity_sampling = False

        verifier = Verifier(net=model, input_shape=input_shape,
                            batch=batch, device=device)

        t0 = time.time()
        status = verifier.verify(
            dnf_objectives=copy.deepcopy(objectives),
            timeout=triage_timeout,
        )
        elapsed = time.time() - t0
        res['elapsed'] = elapsed
        res['status']  = status

    except Exception as e:
        res['status'] = 'ERROR'
        res['error']  = str(e)
    finally:
        if cleanup:
            try: cleanup()
            except: pass

    return res


def run_triage(rows, bench_dir, args) -> list[dict]:
    """Phase 1: scan all rows, return list of triage dicts."""
    n = len(rows)
    print(_c(f'\n  Phase 1 — TRIAGE  ({n} instances, timeout={args.triage_timeout}s each)\n', BOLD, BLUE))

    header = f"  {'#':>4}  {'ONNX':42s}  {'prop':8s}  {'status':10s}  {'time':>6s}"
    print(_c(header, DIM))
    print(_c('  ' + '─'*76, DIM))

    results = []
    for i, row in enumerate(rows, 1):
        onnx_rel, spec_rel = row[0], row[1]
        onnx_s = os.path.basename(onnx_rel).replace('.onnx','')
        spec_s = os.path.basename(spec_rel).replace('.vnnlib','')

        r = _triage_instance(onnx_rel, spec_rel, bench_dir,
                             args.device, args.batch, args.triage_timeout)
        results.append(r)

        st = r['status']
        if st == ReturnStatus.SAT:
            label = _c(f'{"SAT":10s}', BOLD, GREEN)
        elif st == ReturnStatus.UNSAT:
            label = _c(f'{"UNSAT":10s}', DIM)
        elif st == ReturnStatus.TIMEOUT:
            label = _c(f'{"TIMEOUT":10s}', YELLOW)
        elif st == 'ERROR':
            label = _c(f'{"ERROR":10s}', RED)
        else:
            label = _c(f'{str(st):10s}', YELLOW)

        t = f'{r["elapsed"]:6.2f}s'
        print(f'  {i:>4}  {onnx_s:42s}  {spec_s:8s}  {label}  {t}')

    sat_rows = [results[i] for i in range(len(results))
                if results[i]['status'] == ReturnStatus.SAT]
    print()
    print(f'  Triage complete:  {_c(len(sat_rows), BOLD, GREEN)} / {n} instances are SAT within {args.triage_timeout}s')
    return results


# ── Phase 2: Diversity sampling ───────────────────────────────────────────────

def _diversity_instance(onnx_rel, spec_rel, bench_dir, args) -> dict:
    res = {'onnx': onnx_rel, 'spec': spec_rel,
           'status': 'ERROR', 'elapsed': 0.0,
           'n_returned': 0, 'n_verified': 0, 'all_valid': False,
           'metrics': {}, 'error': ''}
    cleanup = None
    try:
        model, input_shape, output_shape, objectives, cleanup = \
            _load_instance(onnx_rel, spec_rel, bench_dir)

        Settings.setup(None)
        Settings.use_attack               = True
        Settings.use_restart              = True
        Settings.use_mip_tightening       = False
        Settings.use_mip_verify           = False
        Settings.diversity_stochastic_prob = args.stochastic_prob
        Settings.diversity_bam_n           = args.bam_n

        verifier = Verifier(net=model, input_shape=input_shape,
                            batch=args.batch, device=args.device)

        t0 = time.time()
        samples = verifier.sample_violations(
            dnf_objectives=copy.deepcopy(objectives),
            k=args.k,
            timeout=args.timeout,
        )
        elapsed = time.time() - t0

        res['elapsed']    = elapsed
        res['n_returned'] = len(samples)

        if not samples:
            res['status'] = 'NO_SAMPLES'
            return res

        # re-parse spec for fresh objectives (original was consumed)
        _, _, _, check_obj, check_cleanup = \
            _load_instance(onnx_rel, spec_rel, bench_dir)
        try:
            n_ret, n_ok = _check_samples(model, samples, check_obj, args.device)
        finally:
            check_cleanup()

        res['n_verified'] = n_ok
        res['all_valid']  = (n_ok == n_ret)
        res['status']     = 'OK'

        lb = objectives.lower_bounds[0]
        ub = objectives.upper_bounds[0]
        res['metrics'] = _diversity(samples, lb, ub)

    except Exception as e:
        import traceback
        res['error']  = str(e)
        res['status'] = 'ERROR'
        if os.environ.get('DIV_DEBUG'):
            traceback.print_exc()
    finally:
        if cleanup:
            try: cleanup()
            except: pass

    return res


# ── Pretty printers ───────────────────────────────────────────────────────────

def _print_div_result(r: dict, idx: int, total: int) -> None:
    onnx_s = os.path.basename(r['onnx']).replace('.onnx','')
    spec_s  = os.path.basename(r['spec']).replace('.vnnlib','')
    print(f'  [{idx:2d}/{total}]  {_c(onnx_s, BOLD, CYAN)}  |  {spec_s}')

    st = r['status']
    if st == 'ERROR':
        print(f'          {_c("ERROR", BOLD, RED)}  {r["error"][:80]}')
    elif st == 'NO_SAMPLES':
        print(f'          {_c("no witnesses found within timeout", YELLOW)}  ({r["elapsed"]:.1f}s)')
    else:
        n_ret, n_ok = r['n_returned'], r['n_verified']
        all_ok = r['all_valid']
        frac   = n_ok / n_ret if n_ret else 0
        v_bar  = _bar(frac, width=16)
        v_col  = GREEN if all_ok else RED
        vlabel = _c('ALL VALID ✓', BOLD, GREEN) if all_ok else _c(f'{n_ok}/{n_ret} ✗', BOLD, RED)

        print(f'          Witnesses : {_c(n_ret, BOLD)}  {vlabel}  '
              f'[{_c(v_bar, v_col)}]  ({r["elapsed"]:.1f}s)')

        m = r['metrics']
        if m.get('n', 0) >= 2:
            div_frac = min(m['mean_pw_n'], 1.0)
            d_col    = (GREEN if div_frac > 0.2 else
                        YELLOW if div_frac > 0.05 else RED)
            d_bar    = _bar(div_frac, width=16)
            print(f'          Diversity :  mean-pw-L2={m["mean_pw"]:.5f}  '
                  f'(norm {m["mean_pw_n"]:.5f})')
            print(f'                       min-pw-L2 ={m["min_pw"]:.5f}  '
                  f'(norm {m["min_pw_n"]:.5f})')
            print(f'                       coverage  ={m["cov"]:.5f}  '
                  f'(norm {m["cov_n"]:.5f})')
            print(f'                       [{_c(d_bar, d_col)}] score={div_frac:.4f}')
        else:
            print(f'          {_c("Only 1 sample — no pairwise diversity", DIM)}')
    print()


def _print_summary(results: list[dict], args) -> None:
    print(_c('═' * 80, BOLD))
    print(_c('  SUMMARY  —  NeuralSAT-Div  ·  ACAS Xu', BOLD, CYAN))
    print(_c('═' * 80, BOLD))

    total    = len(results)
    n_ok     = sum(1 for r in results if r['status'] == 'OK')
    n_nos    = sum(1 for r in results if r['status'] == 'NO_SAMPLES')
    n_err    = sum(1 for r in results if r['status'] == 'ERROR')
    n_allv   = sum(1 for r in results if r['all_valid'])

    print(f'  Instances run            : {total}')
    print(f'  With ≥1 witness          : {_c(n_ok, BOLD, GREEN)}')
    print(f'  No witnesses found       : {_c(n_nos, YELLOW) if n_nos else n_nos}')
    print(f'  Errors                   : {_c(n_err, RED) if n_err else n_err}')
    print(f'  All witnesses valid      : {_c(n_allv, BOLD, GREEN)}/{n_ok}')

    ok = [r for r in results if r['status'] == 'OK']
    if ok:
        mpw_ns = [r['metrics']['mean_pw_n'] for r in ok if r['metrics'].get('n', 0) >= 2]
        mnpw_ns= [r['metrics']['min_pw_n']  for r in ok if r['metrics'].get('n', 0) >= 2]
        cov_ns = [r['metrics']['cov_n']     for r in ok if r['metrics'].get('n', 0) >= 2]
        print()
        print(f'  Avg witnesses returned   : {np.mean([r["n_returned"] for r in ok]):.2f}'
              f'  (target k={args.k})')
        print(f'  Avg verified valid       : {np.mean([r["n_verified"] for r in ok]):.2f}')
        print(f'  Avg elapsed (s)          : {np.mean([r["elapsed"] for r in ok]):.1f}')
        if mpw_ns:
            print()
            print(f'  ── Diversity (normalised, instances with ≥2 witnesses) ─────────────')
            print(f'     Mean pairwise L2  : '
                  f'avg={np.mean(mpw_ns):.5f}  '
                  f'min={np.min(mpw_ns):.5f}  '
                  f'max={np.max(mpw_ns):.5f}')
            print(f'     Min  pairwise L2  : '
                  f'avg={np.mean(mnpw_ns):.5f}  '
                  f'min={np.min(mnpw_ns):.5f}  '
                  f'max={np.max(mnpw_ns):.5f}')
            print(f'     Coverage radius   : '
                  f'avg={np.mean(cov_ns):.5f}  '
                  f'min={np.min(cov_ns):.5f}  '
                  f'max={np.max(cov_ns):.5f}')
            overall_div = float(np.mean(mpw_ns))
            col = GREEN if overall_div > 0.2 else (YELLOW if overall_div > 0.05 else RED)
            bar = _bar(min(overall_div, 1.0), width=24)
            print(f'     Overall score     :  [{_c(bar, col)}]  {overall_div:.5f}')

    print(_c('═' * 80, BOLD))


# ── arg parse + main ─────────────────────────────────────────────────────────

def _parse_args():
    default_dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    p = argparse.ArgumentParser(
        description='NeuralSAT-Div: triage + diversity eval on ACAS Xu',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--benchmark_dir', default=os.path.join(
        SCRIPT_DIR, 'vnncomp2026_benchmarks/benchmarks/acasxu_2023/1.0'))

    # Triage
    p.add_argument('--triage_timeout', type=float, default=1.0,
                   help='Normal-verify timeout for triage scan (s, default 1)')
    p.add_argument('--triage_only', action='store_true',
                   help='Stop after triage; print SAT instances and exit')
    p.add_argument('--sat_instances_file', type=str, default=None,
                   help='CSV file with pre-known SAT instances to skip triage')

    # Diversity sampling
    p.add_argument('--k',       type=int,   default=10,  help='Target witnesses per instance')
    p.add_argument('--timeout', type=float, default=60., help='Diversity timeout per instance (s)')
    p.add_argument('--stochastic_prob', type=float, default=0.3)
    p.add_argument('--bam_n',   type=int,   default=5)

    # Common
    p.add_argument('--device',  type=str,   default=default_dev)
    p.add_argument('--batch',   type=int,   default=200)
    p.add_argument('--filter_prop', type=str, default=None,
                   help='Keep only instances whose vnnlib contains this string')
    p.add_argument('--max_sat',  type=int,   default=None,
                   help='Cap the number of SAT instances to run diversity on')
    p.add_argument('--seed',    type=int,   default=0)
    return p.parse_args()


def main():
    args = _parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    bench_dir = args.benchmark_dir
    csv_path  = os.path.join(bench_dir, 'instances.csv')
    assert os.path.exists(csv_path), f'instances.csv not found: {csv_path}'

    with open(csv_path) as f:
        all_rows = list(csv.reader(f))
    if args.filter_prop:
        all_rows = [r for r in all_rows if args.filter_prop in r[1]]

    # ── Banner ────────────────────────────────────────────────────────────
    print()
    print(_c('╔' + '═'*78 + '╗', BOLD, CYAN))
    print(_c('║  NeuralSAT-Div  ·  ACAS Xu  ·  Triage → Diversity Evaluation' + ' '*16 + '║', BOLD, CYAN))
    print(_c('╚' + '═'*78 + '╝', BOLD, CYAN))
    print(f'  benchmark_dir   : {bench_dir}')
    print(f'  total instances : {len(all_rows)}'
          + (f'  (filtered by "{args.filter_prop}")' if args.filter_prop else ''))
    print(f'  triage_timeout  : {args.triage_timeout}s')
    print(f'  diversity k     : {args.k}  timeout={args.timeout}s')
    print(f'  device          : {args.device}  batch={args.batch}')
    print(f'  stoch_prob      : {args.stochastic_prob}  bam_n={args.bam_n}')
    print()

    # ── Determine SAT instances ───────────────────────────────────────────
    if args.sat_instances_file:
        # Load pre-computed SAT list
        with open(args.sat_instances_file) as f:
            sat_rows = [tuple(r) for r in csv.reader(f)]
        print(f'  Loaded {len(sat_rows)} pre-known SAT instances from {args.sat_instances_file}')
    else:
        # Phase 1: triage
        triage_results = run_triage(all_rows, bench_dir, args)
        sat_triage = [r for r in triage_results if r['status'] == ReturnStatus.SAT]
        sat_rows   = [(r['onnx'], r['spec']) for r in sat_triage]

        # Save for reuse
        sat_file = os.path.join(SCRIPT_DIR, 'sat_instances.csv')
        with open(sat_file, 'w', newline='') as f:
            csv.writer(f).writerows(sat_rows)
        print(f'\n  SAT instances saved → {_c(sat_file, BOLD)}')

        if args.triage_only:
            print()
            print(_c(f'  --triage_only: stopping here.  {len(sat_rows)} SAT instances found.', BOLD, YELLOW))
            print(f'  Re-run without --triage_only to sample diversity on them.\n')
            return

    if args.max_sat:
        sat_rows = sat_rows[:args.max_sat]

    if not sat_rows:
        print(_c('\n  No SAT instances found — nothing to diversity-sample.', YELLOW))
        return

    # ── Phase 2: Diversity sampling ───────────────────────────────────────
    print()
    print(_c(f'\n  Phase 2 — DIVERSITY SAMPLING  ({len(sat_rows)} SAT instances, '
             f'k={args.k}, timeout={args.timeout}s)\n', BOLD, MAGENTA))

    div_results = []
    for idx, row in enumerate(sat_rows, 1):
        onnx_rel, spec_rel = row[0], row[1]
        print(_c(f'  {"─"*76}', DIM))
        r = _diversity_instance(onnx_rel, spec_rel, bench_dir, args)
        div_results.append(r)
        _print_div_result(r, idx, len(sat_rows))

    print()
    _print_summary(div_results, args)
    print()


if __name__ == '__main__':
    main()
