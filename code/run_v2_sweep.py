#!/usr/bin/env python3
"""Run the intervention sweep (rank-3, top-3) across 5 models x 3 benchmarks.

Writes to pilot_results/branch_highH_{bench}[_{slug}]_r3.json.

Skip-if-complete logic prevents re-running a cell whose output already has
100 problems, so the sweep is safe to interrupt and resume.
"""
import subprocess, os, json

HERE = os.path.dirname(os.path.abspath(__file__))
PYTHON = os.environ.get('N2D2_PYTHON', 'python')   # adapt to your environment
SCRIPT = os.path.join(HERE, 'branch_highH_bench_v2.py')
PILOT = os.path.join(HERE, '..', 'pilot_results')

MODELS = [
    ("Qwen/Qwen2.5-1.5B-Instruct",          ""),
    ("Qwen/Qwen2.5-7B-Instruct",            "qwen7B"),
    ("google/gemma-7b-it",                   "gemma7B"),
    ("ibm-granite/granite-3.1-8b-instruct",  "granite8B"),
    ("mistralai/Mistral-7B-Instruct-v0.3",   "mistral7B"),
]
BENCHES = ['gsm8k', 'bbh_obj', 'bbh_multi']
RANK = 3
N_TARGET = 100

def is_complete(bench, slug, rank):
    sfx = '' if rank == 5 else f'_r{rank}'
    suffix = f'_{slug}' if slug else ''
    fpath = os.path.join(PILOT, f'branch_highH_{bench}{suffix}{sfx}.json')
    if not os.path.exists(fpath):
        return False
    try:
        return len(json.load(open(fpath))) >= N_TARGET
    except Exception:
        return False

for model_id, slug in MODELS:
    for bench in BENCHES:
        label = slug or 'qwen1_5B'
        if is_complete(bench, slug, RANK):
            print(f"\n  SKIP {label} {bench} rank={RANK} (file complete)", flush=True)
            continue
        print(f"\n{'='*60}")
        print(f"  {label} x {bench} x rank={RANK}")
        print(f"{'='*60}\n", flush=True)
        args = [PYTHON, SCRIPT, bench, f'--model={model_id}', f'--rank={RANK}']
        if slug:
            args.append(f'--slug={slug}')
        result = subprocess.run(args, cwd=os.path.join(HERE, '..'), timeout=10800)
        if result.returncode != 0:
            print("  *** FAILED ***", flush=True)

print("\nSweep complete.")