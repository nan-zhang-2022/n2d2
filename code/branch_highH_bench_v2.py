#!/usr/bin/env python3
"""Intervention experiment.

Output path: pilot_results/branch_highH_{bench}[_{slug}]_r{rank}.json.

Run one (model, bench, rank) cell:
  python code/branch_highH_bench_v2.py <bench> --model=<HF_id> --slug=<tag> --rank=<N>

Notes:
- Defaults match the canonical main runs: rank=3, top-k=3, 100 problems.
"""
import argparse, json, os, re, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from datasets import load_dataset

# ---- CLI ----
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('bench', choices=['gsm8k', 'bbh_obj', 'bbh_multi'])
    ap.add_argument('--model', default='Qwen/Qwen2.5-1.5B-Instruct')
    ap.add_argument('--slug',  default='')
    ap.add_argument('--rank',  type=int, default=3,
                    help='Forced tail token rank (default 3 for v2 main run)')
    ap.add_argument('--topk',  type=int, default=3,
                    help='Number of highest-H branch points per trace (default 3)')
    ap.add_argument('--n-problems', type=int, default=100)
    ap.add_argument('--start', type=int, default=0,
                    help='Resume from this problem index')
    ap.add_argument('--max-new', type=int, default=512)
    return ap.parse_args()

args = parse_args()
BENCH = args.bench
MODEL = args.model
MODEL_SLUG = ('_' + args.slug) if args.slug else ''
TAIL_RANK = args.rank
N_BRANCH = args.topk
N_PROBLEMS = args.n_problems
START_PI = args.start
MAX_NEW = args.max_new

DEVICE = ('mps' if torch.backends.mps.is_available()
          else 'cuda' if torch.cuda.is_available()
          else 'cpu')
DTYPE = torch.float16 if DEVICE in ('mps', 'cuda') else torch.float32

RANK_SFX = '' if TAIL_RANK == 5 else f'_r{TAIL_RANK}'
OUT_PATH = f'pilot_results/branch_highH_{BENCH}{MODEL_SLUG}{RANK_SFX}.json'

print(f"Config: bench={BENCH}, rank={TAIL_RANK}, top-{N_BRANCH}, n={N_PROBLEMS}, device={DEVICE}", flush=True)
print(f"Output: {OUT_PATH}", flush=True)

# ---- benchmark loader ----
def load_bench(name, n):
    if name == 'bbh_obj':
        ds = load_dataset('lukaemon/bbh', 'object_counting', split='test').select(range(n))
        sys_m = "You are a helpful assistant. Count carefully and answer step by step."
        return [(d['input'], sys_m) for d in ds], [d['target'] for d in ds]
    if name == 'bbh_multi':
        ds = load_dataset('lukaemon/bbh', 'multistep_arithmetic_two', split='test').select(range(n))
        sys_m = "You are a helpful assistant. Solve step by step."
        return [(d['input'], sys_m) for d in ds], [d['target'] for d in ds]
    if name == 'gsm8k':
        ds = load_dataset('gsm8k', 'main', split='test').select(range(n))
        sys_m = "You are a helpful assistant that solves math word problems step by step."
        return [(d['question'], sys_m) for d in ds], [d['answer'] for d in ds]
    raise ValueError(f"Unknown benchmark: {name}")

print("Loading model...", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=DTYPE).to(DEVICE).eval()
prompts, golds = load_bench(BENCH, N_PROBLEMS)

_H_cache = {}
def H_Vs(s, n=1000):
    sr = round(s, 3)
    if sr not in _H_cache:
        _H_cache[sr] = sum(j ** (-sr) for j in range(1, n + 1))
    return _H_cache[sr]

def estimate_zipf_s(top100, start=3, end=50):
    n = min(end, len(top100))
    ranks = np.arange(start, n + 1, dtype=np.float64)
    probs = np.array(top100[start - 1:n], dtype=np.float64)
    mask = probs > 0
    if mask.sum() < 3:
        return 2.0
    A = np.vstack([np.log(ranks[mask]), np.ones(mask.sum())]).T
    sol, *_ = np.linalg.lstsq(A, np.log(probs[mask]), rcond=None)
    return max(-sol[0], 1.05)

TAU_FLOOR = 1e-7
def compute_D_from_sorted(sorted_p):
    top100 = sorted_p[:100].tolist()
    s = estimate_zipf_s(top100)
    K = 200
    tau = float(sorted_p[K:].sum())
    ck = float((sorted_p[K:] ** 2).sum())
    if tau < TAU_FLOOR:
        return 0.0, tau, ck, s
    tq = (K ** (1 - s)) / ((s - 1) * H_Vs(s)) if s > 1.01 else 0.5
    m = max(tau - tq, 0.0)
    D = (m * m / ck) if ck > 1e-30 else 0.0
    return D, tau, ck, s

def compute_H(probs):
    p = probs[probs > 0]
    return -float((p * np.log(p)).sum())

def full_step_stats(probs):
    """Every step gets: H, D (canonical fit), tau200, ck200, zipf_s,
       top1, top5, and sorted top-100 probabilities (for post-hoc refits)."""
    sorted_p = np.sort(probs)[::-1].copy()
    H = compute_H(probs)
    D, tau200, ck200, zipf_s = compute_D_from_sorted(sorted_p)
    return {
        'H': H,
        'D': D,
        'tau200': tau200,
        'ck200': ck200,
        'zipf_s': zipf_s,
        'top1': float(sorted_p[0]),
        'top5': float(sorted_p[:5].sum()),
        'top100': [round(float(x), 8) for x in sorted_p[:100]],
    }

def pick_tail_token(probs, start_rank=5):
    idx = np.argsort(probs)[::-1]
    for r in range(start_rank - 1, min(200, len(idx))):
        tid = int(idx[r])
        if float(probs[tid]) > 1e-6:
            return tid, float(probs[tid]), r + 1
    return None, 0, 0

def classify_regime(step_H, step_D):
    thinking = [(h, d) for h, d in zip(step_H, step_D) if h > 0.1]
    if len(thinking) < 6:
        return 'too_short', {}
    n = len(thinking); third = n // 3
    eH = float(np.mean([x[0] for x in thinking[:third]]))
    lH = float(np.mean([x[0] for x in thinking[-third:]]))
    eD = float(np.mean([x[1] for x in thinking[:third]]))
    lD = float(np.mean([x[1] for x in thinking[-third:]]))
    H_down = lH < eH; D_down = lD < eD
    if H_down and D_down:        regime = 'lock-in'
    elif H_down and not D_down:  regime = 'confusion'
    elif not H_down and D_down:  regime = 'H_up_D_down'
    else:                        regime = 'H_up_D_up'
    return regime, {'early_H': eH, 'late_H': lH, 'early_D': eD, 'late_D': lD,
                    'n_thinking': len(thinking)}

# ---- branch generation (forced tail then greedy) ----
def generate_tracked(past_kv, forced_tok_id, max_new=256):
    gen_tokens = [forced_tok_id]
    steps = []
    cur = torch.tensor([[forced_tok_id]], device=DEVICE)
    past = DynamicCache()
    if isinstance(past_kv, DynamicCache):
        for li in range(len(past_kv)):
            layer = past_kv.layers[li]
            past.update(layer.keys.clone(), layer.values.clone(), li)
    else:
        for li, (k, v) in enumerate(past_kv):
            past.update(k.clone(), v.clone(), li)
    with torch.no_grad():
        for _ in range(max_new):
            out = model(cur, past_key_values=past, use_cache=True)
            past = out.past_key_values
            probs = F.softmax(out.logits[0, -1, :].float(), dim=-1).cpu().numpy()
            tid = int(probs.argmax())
            s = full_step_stats(probs)
            s['token_id'] = tid
            steps.append(s)
            gen_tokens.append(tid)
            cur = torch.tensor([[tid]], device=DEVICE)
            if tid == tok.eos_token_id:
                break
    text = tok.decode(gen_tokens, skip_special_tokens=True)
    return text, gen_tokens, steps

def extract_number(text):
    nums = re.findall(r'[-+]?\d*\.?\d+', text)
    return nums[-1] if nums else ''

# ---- main loop ----
results = []
if START_PI > 0 and os.path.exists(OUT_PATH):
    results = json.load(open(OUT_PATH))
    print(f"Loaded {len(results)} existing results; resuming from pi={START_PI}", flush=True)

t0 = time.time()
for pi in range(START_PI, N_PROBLEMS):
    q, sys_msg = prompts[pi]
    gold = golds[pi]
    try:
        msgs = [{'role': 'system', 'content': sys_msg or 'You are a helpful assistant.'},
                {'role': 'user', 'content': q}]
        prompt_text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        merged = f"{sys_msg}\n\n{q}" if sys_msg else q
        msgs = [{'role': 'user', 'content': merged}]
        prompt_text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(prompt_text, return_tensors='pt').input_ids.to(DEVICE)

    # Phase 1: greedy with full tracking
    greedy_steps = []
    greedy_tokens = []
    greedy_probs_cache = []
    with torch.no_grad():
        past = None; cur = ids
        for _ in range(MAX_NEW):
            out = model(cur, past_key_values=past, use_cache=True)
            past = out.past_key_values
            probs = F.softmax(out.logits[0, -1, :].float(), dim=-1).cpu().numpy()
            tid = int(probs.argmax())
            s = full_step_stats(probs)
            s['token_id'] = tid
            greedy_steps.append(s)
            greedy_tokens.append(tid)
            greedy_probs_cache.append(probs)
            cur = torch.tensor([[tid]], device=DEVICE)
            if tid == tok.eos_token_id:
                break

    greedy_text = tok.decode(greedy_tokens, skip_special_tokens=True)
    greedy_ans = extract_number(greedy_text)
    gold_ans = extract_number(str(gold))
    correct = (greedy_ans == gold_ans)

    gH = [s['H'] for s in greedy_steps]
    gD = [s['D'] for s in greedy_steps]
    regime, regime_info = classify_regime(gH, gD)

    # Phase 2: pick top-k highest-H thinking positions
    thinking_idx = [(i, gH[i]) for i in range(len(greedy_tokens)) if gH[i] > 0.1]
    if len(thinking_idx) < N_BRANCH:
        print(f"  pi={pi:>3} skipped (only {len(thinking_idx)} thinking steps)", flush=True)
        continue
    thinking_idx.sort(key=lambda x: -x[1])
    branch_steps_sel = [x[0] for x in thinking_idx[:N_BRANCH]]

    # Phase 3: branch at each selected position
    branches = []
    for bstep in branch_steps_sel:
        with torch.no_grad():
            past2 = None; cur2 = ids
            for step_i in range(bstep + 1):
                out2 = model(cur2, past_key_values=past2, use_cache=True)
                past2 = out2.past_key_values
                if step_i < bstep:
                    cur2 = torch.tensor([[greedy_tokens[step_i]]], device=DEVICE)

        tail_tid, tail_p, tail_rank = pick_tail_token(greedy_probs_cache[bstep], start_rank=TAIL_RANK)
        if tail_tid is None:
            continue

        btext, btokens, bsteps_data = generate_tracked(past2, tail_tid, max_new=256)
        bans = extract_number(btext)

        greedy_after_H = [s['H'] for s in greedy_steps[bstep + 1:]]
        greedy_after_D = [s['D'] for s in greedy_steps[bstep + 1:]]
        branch_H = [s['H'] for s in bsteps_data]
        branch_D = [s['D'] for s in bsteps_data]
        ga_mean_H = float(np.mean(greedy_after_H)) if greedy_after_H else 0.0
        ga_mean_D = float(np.mean(greedy_after_D)) if greedy_after_D else 0.0
        b_mean_H  = float(np.mean(branch_H)) if branch_H else 0.0
        b_mean_D  = float(np.mean(branch_D)) if branch_D else 0.0
        br_regime, br_regime_info = classify_regime(branch_H, branch_D)

        branches.append({
            'branch_step': bstep,
            'branch_step_frac': bstep / len(greedy_tokens),
            'intervention_H': float(gH[bstep]),
            'intervention_D': float(gD[bstep]),
            'intervention_tau200': greedy_steps[bstep]['tau200'],
            'intervention_ck200': greedy_steps[bstep]['ck200'],
            'intervention_zipf_s': greedy_steps[bstep]['zipf_s'],
            'intervention_top1': greedy_steps[bstep]['top1'],
            'intervention_top5': greedy_steps[bstep]['top5'],
            'greedy_token_id': greedy_tokens[bstep],
            'greedy_token': tok.decode([greedy_tokens[bstep]]),
            'tail_token_id': tail_tid,
            'tail_token': tok.decode([tail_tid]),
            'tail_prob': tail_p,
            'tail_rank': tail_rank,
            'greedy_after_mean_H': ga_mean_H,
            'greedy_after_mean_D': ga_mean_D,
            'greedy_after_n_steps': len(greedy_after_D),
            'branch_mean_H': b_mean_H,
            'branch_mean_D': b_mean_D,
            'branch_n_steps': len(branch_D),
            'branch_regime': br_regime,
            'branch_regime_info': br_regime_info,
            'delta_D': b_mean_D - ga_mean_D,
            'delta_H': b_mean_H - ga_mean_H,
            'greedy_answer': greedy_ans,
            'branch_answer': bans,
            'same_answer': (bans == greedy_ans) if (bans and greedy_ans) else None,
            'branch_steps': bsteps_data,
            'branch_tokens': btokens,
        })

    result = {
        'pi': pi,
        'correct': correct,
        'greedy_answer': greedy_ans,
        'gold': str(gold),
        'regime': regime,
        'regime_info': regime_info,
        'regime_and_correct': f"{'correct' if correct else 'wrong'}_{regime}",
        'greedy_n_steps': len(greedy_tokens),
        'greedy_mean_H': float(np.mean(gH)),
        'greedy_mean_D': float(np.mean(gD)),
        'greedy_median_D': float(np.median(gD)),
        'greedy_steps': greedy_steps,
        'greedy_tokens': greedy_tokens,
        'branches': branches,
    }
    results.append(result)

    n_br = len(branches)
    dDs = [b['delta_D'] for b in branches]
    avg_dD = float(np.mean(dDs)) if dDs else 0.0
    dt = time.time() - t0
    tag = 'OK' if correct else 'WR'
    print(f"  pi={pi:>3} [{tag}] {regime:<12} {n_br} branches  avg ΔD={avg_dD:>+8.1f}  [{dt:.0f}s]",
          flush=True)

    # Incremental save
    json.dump(results, open(OUT_PATH, 'w'))

print(f"\nSaved to {OUT_PATH}")
print(f"Total time: {time.time() - t0:.0f}s")