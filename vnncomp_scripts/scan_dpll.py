#!/usr/bin/env python3
import sys, os, csv, time, copy, tempfile, gzip, shutil, warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR    = os.path.join(SCRIPT_DIR, 'src')
sys.path.insert(0, SRC_DIR)

import torch
from helper.network.read_onnx import parse_onnx
from helper.spec.objective    import parse_vnnlib
from helper.misc.result       import ReturnStatus
from verifier.verifier        import Verifier
from setting                  import Settings

import logging
from helper.misc.logger import logger
logger.setLevel(logging.WARNING)

def _resolve(rel_path: str, bench_dir: str, suffix: str):
    full = os.path.join(bench_dir, rel_path)
    if os.path.exists(full): return full, False
    gz = full + '.gz'
    if os.path.exists(gz):
        fd, tmp = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        with gzip.open(gz, 'rb') as fi, open(tmp, 'wb') as fo:
            shutil.copyfileobj(fi, fo)
        return tmp, True
    raise FileNotFoundError(f'Not found: {full}  (nor {gz})')

def scan_benchmark(bench_dir, log_f, timeout=10.0, max_instances=100):
    csv_path = os.path.join(bench_dir, 'instances.csv')
    if not os.path.exists(csv_path):
        return
    with open(csv_path) as f:
        rows = list(csv.reader(f))
        
    log_f.write(f"\nScanning {bench_dir} ({len(rows)} instances, testing up to {max_instances})...\n")
    log_f.flush()
    
    for i, row in enumerate(rows[:max_instances]):
        onnx_rel, spec_rel = row[0], row[1]
        
        try:
            tmp_onnx, rm_onnx = _resolve(onnx_rel, bench_dir, '.onnx')
            tmp_spec, rm_spec = _resolve(spec_rel, bench_dir, '.vnnlib')
            model, input_shape, output_shape = parse_onnx(tmp_onnx, None, None)
            input_shape = tuple(input_shape)
            objectives = parse_vnnlib(tmp_spec, input_shape)
            
            Settings.setup(None)
            Settings.use_attack = True # Enable attack to see if it fails and DPLL takes over
            Settings.use_restart = False
            Settings.use_mip_tightening = False
            Settings.use_mip_verify = False
            Settings.use_diversity_sampling = False
            
            v = Verifier(net=model, input_shape=input_shape, batch=200, device='cpu')
            
            t0 = time.time()
            status = v.verify(copy.deepcopy(objectives), timeout=timeout)
            t1 = time.time()
            
            # check if DPLL was used. DPLL is used if v.iteration > 0
            iterations = getattr(v, 'iteration', 0)
            
            if iterations > 0 and (t1 - t0) < timeout and status == ReturnStatus.SAT:
                log_f.write(f"  [SUCCESS] DPLL used and found SAT! {onnx_rel} | {spec_rel} in {t1-t0:.2f}s (Iterations: {iterations})\n")
            else:
                log_f.write(f"  [Skip] {onnx_rel} | {spec_rel} -> {status} in {t1-t0:.2f}s (Iterations: {iterations})\n")
            log_f.flush()
            if rm_onnx: os.unlink(tmp_onnx)
            if rm_spec: os.unlink(tmp_spec)
        except Exception as e:
            log_f.write(f"  [Error] {onnx_rel}: {str(e)}\n")

if __name__ == '__main__':
    base = 'vnncomp2026_benchmarks/benchmarks'
    benchmarks = [
        # 'acasxu_2023/1.0',
        # 'collins_aerospace_benchmark/1.0',
        'sat_relu/1.0',
        'cora_2024/1.0',
        'tllverifybench_2023/1.0',
        'traffic_signs_recognition_2023/1.0',
        'soundnessbench/1.0'
    ]
    with open('scan.out', 'w') as log_f:
        for b in benchmarks:
            scan_benchmark(os.path.join(base, b), log_f)
