#!/usr/bin/env python3
r"""W->C / W->W intervention transition table

Output: LaTeX table to stdout.

Usage:
    python code/make_transition_WC_WW_table.py [--rank 3]
"""
import argparse, json, os, re, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PILOT = os.path.join(HERE, '..', 'pilot_results')

MODELS = [('Qwen 1.5B',  ''),
          ('Qwen 7B',    '_qwen7B'),
          ('Gemma 7B',   '_gemma7B'),
          ('Granite 8B', '_granite8B'),
          ('Mistral 7B', '_mistral7B')]
BENCHES = ['gsm8k', 'bbh_obj', 'bbh_multi']
TRANSITIONS = [('W$\\to$C', 'WC'), ('W$\\to$W', 'WW')]

def rank_suffix(rank):
    return '' if rank == 5 else f'_r{rank}'

def ext(t):
    nums = re.findall(r'[-+]?\d*\.?\d+', str(t))
    return nums[-1] if nums else ''

def classify(item, br, gold):
    """Outcome: CC, WC, CW, or WW. Only WC / WW are used downstream."""
    bans = ext(br.get('branch_answer', ''))
    g = bool(item['correct']); b = (bans == gold and bans != '')
    if   g and b:          return 'CC'
    elif not g and b:      return 'WC'
    elif g and not b:      return 'CW'
    else:                  return 'WW'

def ci_median(xs, n_boot=5000, seed=0):
    if len(xs) < 2:
        v = float(np.median(xs)) if xs else 0.0
        return v, v
    a = np.array(xs); rng = np.random.default_rng(seed)
    draws = [np.median(rng.choice(a, size=len(a), replace=True))
             for _ in range(n_boot)]
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))

def gather(rank):
    sfx = rank_suffix(rank)
    per_dH = {(m, t[1]): [] for m, _ in MODELS for t in TRANSITIONS}
    per_dD = {(m, t[1]): [] for m, _ in MODELS for t in TRANSITIONS}
    for mname, mslug in MODELS:
        for bench in BENCHES:
            fp = os.path.join(PILOT, f'branch_highH_{bench}{mslug}{sfx}.json')
            if not os.path.exists(fp): continue
            for item in json.load(open(fp)):
                gold = ext(item['gold'])
                for br in item['branches']:
                    tk = classify(item, br, gold)
                    if tk not in ('WC', 'WW'): continue
                    dH = br.get('delta_H'); dD = br.get('delta_D')
                    if dH is not None: per_dH[(mname, tk)].append(float(dH))
                    if dD is not None: per_dD[(mname, tk)].append(float(dD))
    return per_dH, per_dD

def fmt_signed(v, digits):
    """Format with leading sign, except unsigned zero."""
    if round(v, digits) == 0.0 and v == 0.0:
        return f"{0.0:.{digits}f}"
    return f"{v:+.{digits}f}"

def fmt_med(med, digits, bold):
    s = fmt_signed(med, digits)
    return f"\\textbf{{{s}}}" if bold else s

def fmt_ci(lo, hi, digits):
    return f"[{fmt_signed(lo, digits)}, {fmt_signed(hi, digits)}]"

def find_max_tkey(src, digits):
    """Return the transition key with the larger displayed median
    (using full-precision comparison, then rounding equality check).
    Returns the single tkey of the max, or both tkeys on display-precision tie."""
    meds = {}
    for _, tkey in TRANSITIONS:
        xs = src(tkey)
        if xs: meds[tkey] = float(np.median(xs))
    if not meds: return set()
    mx = max(meds, key=meds.get)
    mx_disp = round(meds[mx], digits)
    return {t for t, v in meds.items() if round(v, digits) == mx_disp}

def find_min_tkey(src, digits):
    """Symmetric counterpart of find_max_tkey: smaller displayed median."""
    meds = {}
    for _, tkey in TRANSITIONS:
        xs = src(tkey)
        if xs: meds[tkey] = float(np.median(xs))
    if not meds: return set()
    mn = min(meds, key=meds.get)
    mn_disp = round(meds[mn], digits)
    return {t for t, v in meds.items() if round(v, digits) == mn_disp}

def emit_latex(per_dH, per_dD):
    cap = ("Intervention-driven changes in entropy and tail-support. "
           "W = wrong, C = correct, med = median. "
           "Within each row's $\\Delta D$ block, the smaller (more-negative) median is \\textbf{bolded}.")
    print(r"\begin{table}[t]")
    print(r"\centering\scriptsize")
    print(r"\caption{" + cap + r"}")
    print(r"\label{tab:transitions_dH_dD_WC_WW_r3}")
    print(r"\setlength{\tabcolsep}{3pt}")
    # Model | n_WC | n_WW | ΔH {WC med, CI, WW med, CI} | ΔD {WC med, CI, WW med, CI}
    print(r"\begin{tabular}{@{}lrr|llll|llll@{}}")
    print(r"\toprule")
    print(r" & & "
          r"& \multicolumn{4}{c|}{$\Delta H$ (nats)} "
          r"& \multicolumn{4}{c}{$\Delta D$} \\")
    print(r"\cmidrule(lr){4-7} \cmidrule(lr){8-11}")
    print(r" & & "
          r"& \multicolumn{2}{c}{W$\to$C} & \multicolumn{2}{c|}{W$\to$W} "
          r"& \multicolumn{2}{c}{W$\to$C} & \multicolumn{2}{c}{W$\to$W} \\")
    print(r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}"
          r"\cmidrule(lr){8-9}\cmidrule(lr){10-11}")
    print(r"Model & $n_{\text{W$\to$C}}$ & $n_{\text{W$\to$W}}$ "
          r"& med & 95\% CI & med & 95\% CI "
          r"& med & 95\% CI & med & 95\% CI \\")
    print(r"\midrule")

    def emit_row(label, src_H, src_D):
        # Bolding rule: ΔH — none; ΔD — the smaller median (more-negative)
        mn_D = find_min_tkey(src_D, digits=1)
        row = [label]
        # n_WC, n_WW
        for _, tkey in TRANSITIONS:
            row.append(str(len(src_H(tkey))))
        # ΔH block (no bolding)
        for _, tkey in TRANSITIONS:
            xs = src_H(tkey)
            if xs:
                med = float(np.median(xs)); lo, hi = ci_median(xs)
                row.append(fmt_med(med, 3, bold=False))
                row.append(fmt_ci(lo, hi, 3))
            else:
                row.append("--"); row.append("--")
        # ΔD block (bold the smaller median)
        for _, tkey in TRANSITIONS:
            xs = src_D(tkey)
            if xs:
                med = float(np.median(xs)); lo, hi = ci_median(xs)
                row.append(fmt_med(med, 1, tkey in mn_D))
                row.append(fmt_ci(lo, hi, 1))
            else:
                row.append("--"); row.append("--")
        print(" & ".join(row) + r" \\")

    for mname, _ in MODELS:
        src_H = (lambda t, rk=mname: per_dH[(rk, t)])
        src_D = (lambda t, rk=mname: per_dD[(rk, t)])
        emit_row(mname, src_H, src_D)

    # All row (aggregate across models)
    print(r"\midrule")
    agg_H = {tkey: sum((per_dH[(m, tkey)] for m, _ in MODELS), []) for _, tkey in TRANSITIONS}
    agg_D = {tkey: sum((per_dD[(m, tkey)] for m, _ in MODELS), []) for _, tkey in TRANSITIONS}
    emit_row("All", lambda t: agg_H[t], lambda t: agg_D[t])

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--rank', type=int, default=3)
    args = ap.parse_args()
    per_dH, per_dD = gather(args.rank)
    emit_latex(per_dH, per_dD)

if __name__ == '__main__':
    main()