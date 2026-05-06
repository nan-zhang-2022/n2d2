#!/usr/bin/env python3
"""
Generate the comprehensive per-step decoding profile.

For each requested benchmark, run greedy or temperature-1 sampling

Output: pilot_results/{slug}_comprehensive_profiles.json

Usage:
    # greedy run for one model
    python code/generate_profiles.py \\
        --model Qwen/Qwen2.5-1.5B-Instruct --slug qwen1_5B_all

    # temperature-1 run for the same model
    python code/generate_profiles.py \\
        --model Qwen/Qwen2.5-1.5B-Instruct --slug qwen1_5B_temp1 \\
        --temperature 1.0 --seed 0

    # subset of benchmarks; first 3 problems each (for smoke testing)
    python code/generate_profiles.py --slug qwen1_5B_quick \\
        --quick gsm8k bbh_obj

The reproduction scripts expect ten files in total -- one per (model, decoder)
cell:
    qwen1_5B_all,   qwen1_5B_temp1
    7B_all,         qwen7B_temp1
    gemma7B_all,    gemma7B_temp1
    granite8B_all,  granite8B_temp1
    mistral7B_all,  mistral7B_temp1
where '_all' is the greedy run and '_temp1' is the temperature-1 sampling run.
The wrapper run_profiles_sweep.py launches all ten in sequence.
"""
import argparse, os, re, sys, json, time
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# ---- CLI ----
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('benches', nargs='*',
                    help='subset of benchmarks; default: all three')
    ap.add_argument('--model', default='Qwen/Qwen2.5-1.5B-Instruct',
                    help='Hugging Face model id (default Qwen 2.5 1.5B Instruct)')
    ap.add_argument('--slug',  required=True,
                    help='output prefix; file is {slug}_comprehensive_profiles.json')
    ap.add_argument('--temperature', type=float, default=0.0,
                    help='0.0 = greedy (default); 1.0 = temperature-1 sampling')
    ap.add_argument('--seed', type=int, default=0,
                    help='RNG seed for stochastic sampling (ignored if greedy)')
    ap.add_argument('--n-problems', type=int, default=100,
                    help='Problems per benchmark (paper uses 100)')
    ap.add_argument('--max-new',    type=int, default=512)
    ap.add_argument('--quick', action='store_true',
                    help='only first 3 problems per benchmark (smoke test)')
    return ap.parse_args()

ARGS = parse_args()
MODEL = ARGS.model
SLUG  = ARGS.slug
TEMPERATURE = float(ARGS.temperature)
GREEDY = (TEMPERATURE <= 0.0)
SEED  = ARGS.seed
MAX_NEW = ARGS.max_new
N_PROBLEMS = 3 if ARGS.quick else ARGS.n_problems
ALL_BENCHES = ['gsm8k', 'bbh_obj', 'bbh_multi']
BENCHES = ARGS.benches if ARGS.benches else ALL_BENCHES
Ks = [50, 100, 200, 500]

DEVICE = ("mps" if torch.backends.mps.is_available()
          else "cuda" if torch.cuda.is_available()
          else "cpu")
DTYPE = torch.float16 if DEVICE in ("mps", "cuda") else torch.float32

# ---- Benchmark loaders ----
def load_bench(name, n):
    if name == 'gsm8k':
        ds = load_dataset('gsm8k', 'main', split='test').select(range(n))
        sys_m = "You are a helpful assistant that solves math word problems step by step."
        return [(d['question'], sys_m) for d in ds], [d['answer'] for d in ds]
    if name == 'bbh_obj':
        ds = load_dataset('lukaemon/bbh', 'object_counting',
                          split='test').select(range(n))
        sys_m = "You are a helpful assistant. Count carefully and answer step by step."
        return [(d['input'], sys_m) for d in ds], [d['target'] for d in ds]
    if name == 'bbh_multi':
        ds = load_dataset('lukaemon/bbh', 'multistep_arithmetic_two',
                          split='test').select(range(n))
        sys_m = "You are a helpful assistant. Solve step by step."
        return [(d['input'], sys_m) for d in ds], [d['target'] for d in ds]
    raise ValueError(f"Unknown benchmark: {name}")

_WORD_NUM = {
    'zero':0, 'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6,
    'seven':7, 'eight':8, 'nine':9, 'ten':10, 'eleven':11, 'twelve':12,
    'thirteen':13, 'fourteen':14, 'fifteen':15, 'sixteen':16, 'seventeen':17,
    'eighteen':18, 'nineteen':19, 'twenty':20,
}

def _last_word_number(text):
    toks = re.findall(r"[A-Za-z]+", text.lower())
    for w in reversed(toks):
        if w in _WORD_NUM:
            return _WORD_NUM[w]
    return None

def extract_number(text):
    """Last signed number in a string, canonicalized; '' if none.
    Handles thousands-separator commas (70,000 -> 70000), trailing-zero
    decimals (20.00 -> 20), bare trailing decimals (20. -> 20), and
    spelled-out small numbers (three -> 3) when no digit is present."""
    s = re.sub(r'(?<=\d),(?=\d)', '', str(text))
    nums = re.findall(r'[-+]?\d*\.?\d+', s)
    if not nums:
        wn = _last_word_number(s)
        return str(wn) if wn is not None else ''
    raw = nums[-1]
    try:
        f = float(raw)
        return str(int(f)) if f == int(f) else ('%g' % f)
    except ValueError:
        return raw

# ---- Per-problem profiling ----
def build_prompt(tok, sys_m, q):
    """Build a chat prompt; merge system into user message for models that
    don't support a system role (e.g. Gemma)."""
    try:
        msgs = [{"role": "system", "content": sys_m},
                {"role": "user",   "content": q}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        merged = f"{sys_m}\n\n{q}" if sys_m else q
        msgs = [{"role": "user", "content": merged}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

def profile_one(pi, q, sys_m, gold, tok, model, generator):
    prompt = build_prompt(tok, sys_m, q)
    ids = tok(prompt, return_tensors="pt").input_ids.to(DEVICE)

    step_H = []; step_top100 = []
    step_tau = {K: [] for K in Ks}
    step_ck  = {K: [] for K in Ks}
    step_tokens = []

    with torch.no_grad():
        past = None; cur = ids
        for _ in range(MAX_NEW):
            out = model(cur, past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits = out.logits[0, -1, :].float()
            probs_t = F.softmax(logits, dim=-1)
            probs = probs_t.cpu().numpy()

            if GREEDY:
                tok_id = int(probs.argmax())
            else:
                if TEMPERATURE != 1.0:
                    probs_t = F.softmax(logits / TEMPERATURE, dim=-1)
                tok_id = int(torch.multinomial(probs_t, num_samples=1, generator=generator).item())

            sorted_p = np.sort(probs)[::-1].copy()
            mask = probs > 0
            ent = -float((probs[mask] * np.log(probs[mask])).sum())
            step_H.append(round(ent, 6))
            step_top100.append([round(float(x), 8) for x in sorted_p[:100]])
            for K in Ks:
                tail = sorted_p[K:]
                step_tau[K].append(float(tail.sum()))
                step_ck[K].append(float((tail ** 2).sum()))
            step_tokens.append(tok_id)
            cur = torch.tensor([[tok_id]], device=DEVICE)
            if tok_id == tok.eos_token_id:
                break

    text = tok.decode(step_tokens, skip_special_tokens=True)
    pred = extract_number(text)
    gold_num = extract_number(gold)
    correct = bool(pred) and (pred == gold_num)

    result = {
        'pi': pi, 'correct': correct, 'pred': pred, 'gold': gold_num,
        'n_steps': len(step_tokens),
        'step_H': step_H, 'step_top100': step_top100,
        'step_tokens': step_tokens,
    }
    for K in Ks:
        result[f'step_tau{K}'] = step_tau[K]
        result[f'step_ck{K}']  = step_ck[K]
    return result

# ---- Main ----
def main():
    print(f"Config: model={MODEL} slug={SLUG} device={DEVICE} dtype={DTYPE} "
          f"temperature={TEMPERATURE} n_problems={N_PROBLEMS} benches={BENCHES}",
          flush=True)
    print("Loading model...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=DTYPE).to(DEVICE).eval()

    # Reproducible sampling
    generator = None
    if not GREEDY:
        generator = torch.Generator(device=DEVICE).manual_seed(SEED)

    here = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(here, '..', 'pilot_results',
                            f'{SLUG}_comprehensive_profiles.json')

    # Resume: load any existing results and skip pi indices already computed
    existing = {}
    if os.path.exists(out_path):
        try:
            prev = json.load(open(out_path))
            existing = prev.get('benchmarks', {})
            print(f"Resuming from existing file: "
                  f"{ {b: len(v) for b, v in existing.items()} }", flush=True)
        except Exception as e:
            print(f"Could not read existing file ({e}); starting fresh.", flush=True)

    all_results = {b: list(existing.get(b, [])) for b in BENCHES}
    t_global = time.time()
    for bench in BENCHES:
        print(f"\n{'='*64}\n  Profiling {bench} ({N_PROBLEMS} problems)\n{'='*64}", flush=True)
        prompts, golds = load_bench(bench, N_PROBLEMS)
        results = all_results[bench]
        done_pis = {r['pi'] for r in results}
        n_correct = sum(int(r['correct']) for r in results)
        t0 = time.time()
        for pi, ((q, sys_m), gold) in enumerate(zip(prompts, golds)):
            if pi in done_pis: continue
            r = profile_one(pi, q, sys_m, gold, tok, model, generator)
            results.append(r); n_correct += int(r['correct'])
            dt = time.time() - t0
            print(f"  pi={pi:>3} n={r['n_steps']:>3} correct={r['correct']} "
                  f"pred={r['pred']!r} gold={r['gold']!r} [{dt:.0f}s]", flush=True)
        results.sort(key=lambda r: r['pi'])
        all_results[bench] = results
        print(f"  {bench}: {n_correct}/{len(results)} correct", flush=True)

    out = {
        'meta': {
            'description': ('Comprehensive per-step profiles: top-100 sorted '
                            'probs, tau/ck at K in {50,100,200,500}, entropy, tok_id'),
            'model': MODEL, 'slug': SLUG,
            'temperature': TEMPERATURE, 'seed': SEED if not GREEDY else None,
            'vocab_size': int(model.config.vocab_size),
            'n_problems_per_bench': N_PROBLEMS,
            'max_new_tokens': MAX_NEW,
            'benchmarks': list(all_results.keys()),
            'fields_per_problem': [
                'pi', 'correct', 'pred', 'gold', 'n_steps',
                'step_H', 'step_top100',
                'step_tau50', 'step_tau100', 'step_tau200', 'step_tau500',
                'step_ck50',  'step_ck100',  'step_ck200',  'step_ck500',
                'step_tokens',
            ],
        },
        'benchmarks': all_results,
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(out, open(out_path, 'w'))
    print(f"\nSaved to {out_path}")
    print(f"Total time: {time.time()-t_global:.0f}s")

if __name__ == '__main__':
    main()