#!/usr/bin/env python3
"""Run the per-step profile sweep across 5 models x 2 decoders.

Writes to pilot_results/{slug}_comprehensive_profiles.json -- one file per
(model, decoder) cell, ten files in total. The reproduction scripts (Table 2
and the appendix tables) read these files by slug.

Skip-if-exists logic prevents recomputing a cell whose output already covers
all three benchmarks at the target problem count.
"""
import subprocess, os, json

HERE = os.path.dirname(os.path.abspath(__file__))
PYTHON = os.environ.get('N2D2_PYTHON', 'python')
SCRIPT = os.path.join(HERE, 'generate_profiles.py')
PILOT  = os.path.join(HERE, '..', 'pilot_results')

# (HF model id, slug for greedy run, slug for temp1 run)
MODELS = [
    ("Qwen/Qwen2.5-1.5B-Instruct",          "qwen1_5B_all",  "qwen1_5B_temp1"),
    ("Qwen/Qwen2.5-7B-Instruct",            "7B_all",        "qwen7B_temp1"),
    ("google/gemma-7b-it",                   "gemma7B_all",   "gemma7B_temp1"),
    ("ibm-granite/granite-3.1-8b-instruct",  "granite8B_all", "granite8B_temp1"),
    ("mistralai/Mistral-7B-Instruct-v0.3",   "mistral7B_all", "mistral7B_temp1"),
]
N_PROBLEMS = 100
SEED = 0

def is_complete(slug):
    fpath = os.path.join(PILOT, f'{slug}_comprehensive_profiles.json')
    if not os.path.exists(fpath):
        return False
    try:
        d = json.load(open(fpath))
        bm = d.get('benchmarks', {})
        return all(len(bm.get(b, [])) >= N_PROBLEMS
                   for b in ('gsm8k', 'bbh_obj', 'bbh_multi'))
    except Exception:
        return False

def run_one(model_id, slug, temperature):
    if is_complete(slug):
        print(f"\n  SKIP {slug} (file complete)", flush=True)
        return
    print(f"\n{'='*60}\n  {slug}  (model={model_id}, T={temperature})\n{'='*60}\n", flush=True)
    args = [PYTHON, SCRIPT,
            f'--model={model_id}', f'--slug={slug}',
            f'--temperature={temperature}', f'--seed={SEED}',
            f'--n-problems={N_PROBLEMS}']
    result = subprocess.run(args, cwd=os.path.join(HERE, '..'))
    if result.returncode != 0:
        print("  *** FAILED ***", flush=True)

for model_id, greedy_slug, temp1_slug in MODELS:
    run_one(model_id, greedy_slug, 0.0)
    run_one(model_id, temp1_slug,  1.0)

print("\nProfile sweep complete.")