#!/usr/bin/env python3
r"""Reproduce Table 2 (tab:level): Consistency check, median per-trace H and D
for correct vs incorrect traces, across 5 models x 2 decoders x 3 benchmarks.

LaTeX output: in each correct/wrong pair, the larger value is bolded.

Usage:
    python code/make_level_table.py
"""
import json, os
import numpy as np
from math import fsum

HERE = os.path.dirname(os.path.abspath(__file__))
PILOT = os.path.join(HERE, '..', 'pilot_results')

MODELS = [('Qwen 1.5B',  'qwen1_5B_all',   'qwen1_5B_temp1'),
          ('Qwen 7B',    '7B_all',         'qwen7B_temp1'),
          ('Gemma 7B',   'gemma7B_all',    'gemma7B_temp1'),
          ('Granite 8B', 'granite8B_all',  'granite8B_temp1'),
          ('Mistral 7B', 'mistral7B_all',  'mistral7B_temp1')]
DECODERS = [('greedy', 0), ('temp1', 1)]
BENCHES = [('GSM8K', 'gsm8k'), ('BBH-OC', 'bbh_obj'), ('BBH-MA', 'bbh_multi')]

H_THR = 0.1
TAU_FLOOR = 1e-7
K = 200
FIT_START, FIT_END = 3, 50

def zeta_approx(s, n=1000):
    return fsum(j ** (-s) for j in range(1, n + 1))

def zipf_s(top100):
    n = min(FIT_END, len(top100))
    ranks = np.arange(FIT_START, n + 1, dtype=np.float64)
    probs = np.array(top100[FIT_START - 1:n], dtype=np.float64)
    mask = probs > 0
    if mask.sum() < 3: return 2.0
    A = np.vstack([np.log(ranks[mask]), np.ones(mask.sum())]).T
    sol, *_ = np.linalg.lstsq(A, np.log(probs[mask]), rcond=None)
    return max(-sol[0], 1.05)

def D_step(top100, tau, ck):
    if tau < TAU_FLOOR or ck <= 1e-30: return 0.0
    s = zipf_s(top100)
    if s <= 1.01:
        tq = 0.5
    else:
        tq = (K ** (1 - s)) / ((s - 1) * zeta_approx(s))
    m = max(tau - tq, 0.0)
    return m * m / ck

def per_trace_HD(problem):
    sH = problem['step_H']
    tau = problem[f'step_tau{K}']
    ck  = problem[f'step_ck{K}']
    top = problem['step_top100']
    Hs, Ds = [], []
    for t in range(problem['n_steps']):
        if sH[t] <= H_THR: continue
        Hs.append(sH[t])
        Ds.append(D_step(top[t], tau[t], ck[t]))
    if not Hs: return None, None
    return float(np.mean(Hs)), float(np.mean(Ds))

def collect():
    """Return dict[(model, decoder, bench)] -> (median H_c, H_w, D_c, D_w)."""
    out = {}
    for mname, greedy_f, temp1_f in MODELS:
        for dname, didx in DECODERS:
            fname = (greedy_f, temp1_f)[didx]
            fp = os.path.join(PILOT, f'{fname}_comprehensive_profiles.json')
            if not os.path.exists(fp):
                print(f"# missing: {fp}")
                continue
            data = json.load(open(fp))
            for blabel, bkey in BENCHES:
                if bkey not in data['benchmarks']: continue
                Hc, Hw, Dc, Dw = [], [], [], []
                for p in data['benchmarks'][bkey]:
                    H, D = per_trace_HD(p)
                    if H is None: continue
                    if p['correct']:
                        Hc.append(H); Dc.append(D)
                    else:
                        Hw.append(H); Dw.append(D)
                if not Hc or not Hw: continue
                out[(mname, dname, blabel)] = (
                    float(np.median(Hc)), float(np.median(Hw)),
                    float(np.median(Dc)), float(np.median(Dw)),
                )
    return out

def fmt_pair(a, b, fmt):
    """Format 'a / b' with the larger one bolded."""
    sa = format(a, fmt); sb = format(b, fmt)
    if a > b:   return f"\\textbf{{{sa}}}\\,/\\,{sb}"
    elif b > a: return f"{sa}\\,/\\,\\textbf{{{sb}}}"
    return f"{sa}\\,/\\,{sb}"

def main():
    cells = collect()

    print(r"\begin{table}[ht]")
    print(r"\centering")
    print(r"\caption{Consistency check: median per-trace $H$ and $D$ for correct and incorrect traces.}")
    print(r"\label{tab:level}")
    print(r"\small")
    print(r"\setlength{\tabcolsep}{3.5pt}")
    print(r"\begin{tabular}{@{}lcccccc@{}}")
    print(r"\toprule")
    print(r" & \multicolumn{3}{c}{Temperature-1} & \multicolumn{3}{c}{Greedy} \\")
    print(r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}")
    print(r" & GSM8K & BBH-OC & BBH-MA & GSM8K & BBH-OC & BBH-MA \\")
    print(r"\midrule")
    print(r"\multicolumn{7}{l}{\textit{Shannon entropy $H$ (correct\,/\,wrong)}} \\")
    print(r"\addlinespace[1pt]")
    for mname, *_ in MODELS:
        row = [mname]
        for dname in ('temp1', 'greedy'):
            for blabel in ('GSM8K', 'BBH-OC', 'BBH-MA'):
                v = cells.get((mname, dname, blabel))
                row.append(fmt_pair(v[0], v[1], '.2f') if v else r'\,---\,')
        print(' & '.join(row) + r' \\')
    print(r"\addlinespace[4pt]")
    print(r"\multicolumn{7}{l}{\textit{Residual support $D$ (correct\,/\,wrong)}} \\")
    print(r"\addlinespace[1pt]")
    for mname, *_ in MODELS:
        row = [mname]
        for dname in ('temp1', 'greedy'):
            for blabel in ('GSM8K', 'BBH-OC', 'BBH-MA'):
                v = cells.get((mname, dname, blabel))
                row.append(fmt_pair(v[2], v[3], '.0f') if v else r'\,---\,')
        print(' & '.join(row) + r' \\')
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

if __name__ == '__main__':
    main()