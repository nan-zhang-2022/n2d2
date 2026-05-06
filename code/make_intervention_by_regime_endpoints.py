#!/usr/bin/env python3
r"""Intervention-by-regime table using endpoint-based D↓/D↑ classification.

Classifies each trace as D↓ (late_D < early_D) or D↑ (late_D >= early_D) using
the mean D over the first and last fractions of the post-prompt tokens:
  --split 4  -> first quarter vs last quarter (canonical, most defensible)
  --split 3  -> first third vs last third (robustness)
  --split 5  -> first fifth vs last fifth (robustness)
  --split 2  -> first half vs last half (robustness, coarsest)

Usage:
    python code/make_intervention_by_regime_endpoints.py --split 4            # canonical
    python code/make_intervention_by_regime_endpoints.py --split 3            # robustness
    python code/make_intervention_by_regime_endpoints.py --split 5 --rank 3   # robustness
"""
import argparse, json, os, re
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PILOT = os.path.join(HERE, '..', 'pilot_results')

MODELS = [('Qwen 1.5B',''),('Qwen 7B','_qwen7B'),('Gemma 7B','_gemma7B'),
          ('Granite 8B','_granite8B'),('Mistral 7B','_mistral7B')]
BENCHES = ['gsm8k', 'bbh_obj', 'bbh_multi']

def ext(t):
    nums = re.findall(r'[-+]?\d*\.?\d+', str(t))
    return nums[-1] if nums else ''

def rank_suffix(rank):
    return '' if rank == 5 else f'_r{rank}'

def classify_endpoint(item, split):
    """D↓ if mean D over first 1/split of tokens > last 1/split, else D↑.
    No H filter. Returns None if trace too short."""
    D_vals = [s['D'] for s in item['greedy_steps']]
    n = len(D_vals)
    if n < 2 * split:  # need at least 2 tokens per endpoint window
        return None
    q = n // split
    early_D = float(np.mean(D_vals[:q]))
    late_D  = float(np.mean(D_vals[-q:]))
    return 'dn' if late_D < early_D else 'up'

def gather(rank, split):
    sfx = rank_suffix(rank)
    data = {m: {'dn': {'dDs':[],'n_W':0,'n_C':0,'fix':0,'brk':0},
                 'up': {'dDs':[],'n_W':0,'n_C':0,'fix':0,'brk':0}} for m, _ in MODELS}
    for mname, mslug in MODELS:
        for bench in BENCHES:
            fp = os.path.join(PILOT, f'branch_highH_{bench}{mslug}{sfx}.json')
            if not os.path.exists(fp): continue
            for item in json.load(open(fp)):
                key = classify_endpoint(item, split)
                if key is None: continue
                gold = ext(item['gold']); ok = bool(item['correct'])
                for br in item['branches']:
                    dD = br.get('delta_D')
                    if dD is None: continue
                    bans = ext(br.get('branch_answer', ''))
                    cell = data[mname][key]
                    cell['dDs'].append(float(dD))
                    if ok:
                        cell['n_C'] += 1
                        if br.get('same_answer') is False and bans != gold:
                            cell['brk'] += 1
                    else:
                        cell['n_W'] += 1
                        if bans == gold and bans != '':
                            cell['fix'] += 1
    return data

def fmt_net(n, bold=False):
    if n > 0: s = f"+{n}"
    elif n < 0: s = f"$-${abs(n)}"
    else: s = "0"
    return f"\\textbf{{{s}}}" if bold else s

def rates_and_net(cell):
    """Return (fix_rate_pct, brk_rate_pct, net) — used to decide cross-regime bolding."""
    fxr = 100 * cell['fix'] / cell['n_W'] if cell['n_W'] else 0.0
    brr = 100 * cell['brk'] / cell['n_C'] if cell['n_C'] else 0.0
    net = cell['fix'] - cell['brk']
    return fxr, brr, net

def fmt_cell(cell, bold_fix=False, bold_brk=False, bold_net=False):
    """Format one regime's cell. The three flags bold the fix rate, broke rate,
    and net respectively (set by the caller based on D↓ vs D↑ comparison).
    Columns: n_W, n_C, med ΔD, >0, fixed (rate), broke (rate), net."""
    dDs = cell['dDs']
    n = len(dDs)
    n_W = cell['n_W']; n_C = cell['n_C']
    if n == 0: return "0 & 0 & -- & -- & 0 (--) & 0 (--) & 0"
    med = float(np.median(dDs))
    pos = 100 * sum(1 for x in dDs if x > 0) / n
    fix = cell['fix']; brk = cell['brk']
    fxr = 100 * fix / n_W if n_W else 0.0
    brr = 100 * brk / n_C if n_C else 0.0
    net = fix - brk
    med_s = f"${med:+.1f}$" if not (round(med, 1) == 0.0 and med == 0.0) else "$0.0$"
    fxr_s = f"\\textbf{{{fxr:.0f}\\%}}" if bold_fix else f"{fxr:.0f}\\%"
    brr_s = f"\\textbf{{{brr:.0f}\\%}}" if bold_brk else f"{brr:.0f}\\%"
    net_s = fmt_net(net, bold=bold_net)
    return f"{n_W} & {n_C} & {med_s} & {pos:.0f}\\% & {fix} ({fxr_s}) & {brk} ({brr_s}) & {net_s}"

def render_row(label, dn_cell, up_cell):
    """Render one model (or 'All') row with cross-regime bolding on fix/brk/net."""
    dn_fxr, dn_brr, dn_net = rates_and_net(dn_cell)
    up_fxr, up_brr, up_net = rates_and_net(up_cell)
    bold_fix_dn, bold_fix_up = dn_fxr > up_fxr, up_fxr > dn_fxr
    bold_brk_dn, bold_brk_up = dn_brr > up_brr, up_brr > dn_brr
    bold_net_dn, bold_net_up = dn_net > up_net, up_net > dn_net
    dn_s = fmt_cell(dn_cell, bold_fix_dn, bold_brk_dn, bold_net_dn)
    up_s = fmt_cell(up_cell, bold_fix_up, bold_brk_up, bold_net_up)
    return f"{label} & {dn_s} & {up_s} \\\\"

def print_latex(data, rank, split):
    split_label = {2: 'halves', 3: 'thirds', 4: 'quarters', 5: 'fifths'}.get(split, f'1/{split}')
    label_sfx = f"_r{rank}" if rank != 5 else ""
    if split != 4: label_sfx += f"_{split_label}"
    cap = ("Interventions by trace regime. "
           f"$D{{\\downarrow}}/D{{\\uparrow}}$ classified by first-{split_label[:-1] if split_label.endswith('s') else split_label} "
           f"vs last-{split_label[:-1] if split_label.endswith('s') else split_label} mean $D$ over all post-prompt tokens (no filter). "
           r"The larger of $D{\downarrow}$ vs $D{\uparrow}$ for fix/broke rate and net is \textbf{bolded} within each row.")
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{" + cap + r"}")
    print(rf"\label{{tab:intervention{label_sfx}}}")
    print(r"\scriptsize")
    print(r"\setlength{\tabcolsep}{3pt}")
    print(r"\begin{tabular}{@{}lrrcrrrrrrcrrrr@{}}")
    print(r"\toprule")
    print(r" & \multicolumn{7}{c}{$D{\downarrow}$ traces} & \multicolumn{7}{c}{$D{\uparrow}$ traces} \\")
    print(r"\cmidrule(lr){2-8} \cmidrule(lr){9-15}")
    print(r"Model & $n_W$ & $n_C$ & med $\Delta D$ & $>$0 & fixed (rate) & broke (rate) & net "
          r"& $n_W$ & $n_C$ & med $\Delta D$ & $>$0 & fixed (rate) & broke (rate) & net \\")
    print(r"\midrule")
    agg = {'dn': {'dDs':[],'n_W':0,'n_C':0,'fix':0,'brk':0},
           'up': {'dDs':[],'n_W':0,'n_C':0,'fix':0,'brk':0}}
    for mname, _ in MODELS:
        dn_cell = data[mname]['dn']; up_cell = data[mname]['up']
        print(render_row(mname, dn_cell, up_cell))
        for reg in ('dn','up'):
            for k, v in data[mname][reg].items():
                if isinstance(v, list): agg[reg][k] += v
                else: agg[reg][k] += v
    print(r"\midrule")
    print(render_row("All", agg['dn'], agg['up']))
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--rank', type=int, default=3)
    ap.add_argument('--split', type=int, default=4,
                    choices=[2, 3, 4, 5],
                    help='Regime-split denominator (2=halves, 3=thirds, 4=quarters [canonical], 5=fifths)')
    args = ap.parse_args()
    data = gather(args.rank, args.split)
    print_latex(data, args.rank, args.split)

if __name__ == '__main__':
    main()