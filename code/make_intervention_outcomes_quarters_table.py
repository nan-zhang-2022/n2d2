#!/usr/bin/env python3
r"""Interventions by trace regime and outcome.

Usage:
    python code/make_intervention_outcomes_quarters_table.py [--rank 3]
"""
import argparse, json, os, re
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PILOT = os.path.join(HERE, '..', 'pilot_results')

MODELS = [('Qwen 1.5B', ''), ('Qwen 7B', '_qwen7B'),
          ('Gemma 7B', '_gemma7B'), ('Granite 8B', '_granite8B'),
          ('Mistral 7B', '_mistral7B')]
BENCHES = ['gsm8k', 'bbh_obj', 'bbh_multi']
SPLIT = 4  # quarters

def ext(t):
    nums = re.findall(r'[-+]?\d*\.?\d+', str(t))
    return nums[-1] if nums else ''

def rank_suffix(rank):
    return '' if rank == 5 else f'_r{rank}'

def classify_regime(item):
    D = [s['D'] for s in item['greedy_steps']]
    n = len(D)
    if n < 2 * SPLIT: return None
    q = n // SPLIT
    early = float(np.mean(D[:q]))
    late = float(np.mean(D[-q:]))
    return 'dn' if late < early else 'up'

def classify_outcome(item, br, gold):
    bans = ext(br.get('branch_answer', ''))
    same = br.get('same_answer')
    if not item['correct'] and bans == gold and bans != '':
        return 'fixed'
    if item['correct'] and same is False and bans != gold:
        return 'broke'
    if (not item['correct']) and bans != gold and same is not True:
        return 'diff'
    return None

def ci_median(xs, n_boot=2000, seed=0):
    if len(xs) < 2:
        v = float(np.median(xs)) if xs else 0.0
        return v, v
    a = np.array(xs); rng = np.random.default_rng(seed)
    draws = [np.median(rng.choice(a, size=len(a), replace=True)) for _ in range(n_boot)]
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))

def gather(rank):
    sfx = rank_suffix(rank)
    buckets = {'dn': {'fixed': [], 'broke': [], 'diff': []},
               'up': {'fixed': [], 'broke': [], 'diff': []}}
    for _, mslug in MODELS:
        for bench in BENCHES:
            fp = os.path.join(PILOT, f'branch_highH_{bench}{mslug}{sfx}.json')
            if not os.path.exists(fp): continue
            for item in json.load(open(fp)):
                reg = classify_regime(item)
                if reg is None: continue
                gold = ext(item['gold'])
                for br in item['branches']:
                    dD = br.get('delta_D')
                    if dD is None: continue
                    out = classify_outcome(item, br, gold)
                    if out is not None:
                        buckets[reg][out].append(float(dD))
    return buckets

def fmt_cell(xs):
    if not xs: return "0 & -- & -- & --"
    a = np.array(xs)
    n = len(a); mean = float(a.mean()); med = float(np.median(a))
    lo, hi = ci_median(xs)
    pos = 100 * float((a > 0).sum()) / n
    def signed(v):
        if round(v, 1) == 0.0 and v == 0.0: return "$0.0$"
        return f"${v:+.1f}$"
    return f"{n} & {signed(mean)} & {signed(med)} [{signed(lo)}, {signed(hi)}] & {pos:.0f}\\%"

def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--rank', type=int, default=3)
    args = ap.parse_args()

    buckets = gather(args.rank)

    cap = ("Interventions by trace regime and outcome, pooled across 5 models and 3 benchmarks. "
           "Regime uses first-quarter vs last-quarter mean $D$ (no $H$ filter). "
           "Diff wrong: the answer changes to a different wrong answer. "
           "Entries report $n$, mean, median with 95\\% bootstrap CI, and fraction with $\\Delta D>0$.")

    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{" + cap + r"}")
    label_sfx = "" if args.rank == 3 else f"_r{args.rank}"
    print(rf"\label{{tab:intervention_outcomes_r3_quarters{label_sfx}}}")
    print(r"\small")
    print(r"\setlength{\tabcolsep}{3pt}")
    print(r"\begin{tabular}{@{}lrrlrrrlr@{}}")
    print(r"\toprule")
    print(r" & \multicolumn{4}{c}{$D{\downarrow}$ traces} & \multicolumn{4}{c}{$D{\uparrow}$ traces} \\")
    print(r"\cmidrule(lr){2-5} \cmidrule(lr){6-9}")
    print(r"Outcome & $n$ & mean & med [95\% CI] & $\Delta D\!>\!0$ "
          r"& $n$ & mean & med [95\% CI] & $\Delta D\!>\!0$ \\")
    print(r"\midrule")
    for label, key in [('Fixed', 'fixed'),
                       ('Broke', 'broke'),
                       ('Diff wrong', 'diff')]:
        dn_c = fmt_cell(buckets['dn'][key])
        up_c = fmt_cell(buckets['up'][key])
        print(f"{label} & {dn_c} & {up_c} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

if __name__ == '__main__':
    main()