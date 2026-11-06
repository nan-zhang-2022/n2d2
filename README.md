# Code package

This directory contains the code and reproducibility scripts for:

> **Why Decoding Sharpens Without Resolving: Lock-in and Confusion in Autoregressive Reasoning**  
> Nan Zhang, Heng Xu  
> *Advances in Neural Information Processing Systems (NeurIPS), 2026*  
> [NeurIPS paper page](https://neurips.cc/virtual/2026/poster/150729)

## Abstract

Autoregressive reasoning traces often sharpen as they unfold — entropy falls and token probabilities concentrate — suggesting decoding is resolving uncertainty. Yet the same pattern persists after clear reasoning errors. We analyze decoding through the conditional posterior it induces over *branches* — equivalence classes of continuations under any chosen notion of "resolution" — and show that a mismatch between token selection and branch resolution produces two regimes: *lock-in*, where top tokens remain branch-evidential and reinforce commitment, and *confusion*, where branch-specific residuals self-average and top tokens become weakly diagnostic. Two observable consequences follow: entropy may fall while unresolved support stays elevated, and greedy token selection has opposite informational effects across regimes. We confirm these theoretical predictions across multiple open-weight models and reasoning benchmarks.

## Directory structure

```
.
├── README.md
├── code/
│   ├── generate_profiles.py                     # core pipeline: per-step decoding profiles
│   ├── run_profiles_sweep.py                    # core pipeline: Stage 1 launcher
│   ├── branch_highH_bench_v2.py                 # core pipeline: high-H intervention runner
│   ├── run_v2_sweep.py                          # core pipeline: Stage 2 launcher
│   ├── make_level_table.py                      # reproduces Table 2
│   ├── make_transition_WC_WW_table.py           # reproduces Table 3
│   ├── make_intervention_by_regime_endpoints.py # reproduces Table 4
│   └── make_intervention_outcomes_quarters_table.py  # reproduces Table 8
└── pilot_results/                               # output directory; expected to hold:
    ├── {slug}_comprehensive_profiles.json       # produced by generate_profiles.py (Stage 1)
    └── branch_highH_{bench}[_{slug}]_r3.json    # produced by branch_highH_bench_v2.py (Stage 2)
```

## Dependencies

Python 3.10+ with:

```
torch >= 2.1     (used by generate_profiles.py and branch_highH_bench_v2.py)
transformers >= 4.40
datasets         (loaded from Hugging Face Hub for GSM8K and BBH)
numpy
```

The reproduction scripts (`make_*.py`) only need `numpy`; they read the JSON
artefacts produced by the pipeline and emit LaTeX tables on stdout.

## Models

The full sweep covers five open-weight instruction-tuned models. Hugging Face
identifiers are listed in `code/run_profiles_sweep.py` and `code/run_v2_sweep.py`.
Some weights are gated and require accepting the model licence; in that case
set the `HF_TOKEN` environment variable in your shell before running.

## End-to-end reproduction

The pipeline has two stages, each producing JSON artefacts under `pilot_results/`.

### Stage 1: per-step decoding profiles

Run greedy and temperature-1 decoding on the three benchmarks (GSM8K, BBH
object-counting, BBH multistep-arithmetic) and save per-step profiles.

Reproduction scripts expect ten files in total — one per (model, decoder) cell:

```
qwen1_5B_all_comprehensive_profiles.json     qwen1_5B_temp1_comprehensive_profiles.json
7B_all_comprehensive_profiles.json           qwen7B_temp1_comprehensive_profiles.json
gemma7B_all_comprehensive_profiles.json      gemma7B_temp1_comprehensive_profiles.json
granite8B_all_comprehensive_profiles.json    granite8B_temp1_comprehensive_profiles.json
mistral7B_all_comprehensive_profiles.json    mistral7B_temp1_comprehensive_profiles.json
```

Run all ten in sequence:
```
python code/run_profiles_sweep.py
```

Or one cell at a time (e.g. for a single greedy run on one model):
```
python code/generate_profiles.py --model Qwen/Qwen2.5-1.5B-Instruct --slug qwen1_5B_all
python code/generate_profiles.py --model Qwen/Qwen2.5-1.5B-Instruct --slug qwen1_5B_temp1 --temperature 1.0 --seed 0
```

`generate_profiles.py` resumes from any partial output, so the sweep is safe
to interrupt.

### Stage 2: high-entropy intervention sweep

`run_v2_sweep.py` launches `branch_highH_bench_v2.py` for every
`(model, benchmark)` cell at rank-3, top-3 (the canonical setting). For each
cell it picks the three highest-entropy thinking tokens in the greedy trace,
forces the rank-k tail token, lets the model continue, and records both the
greedy and branch trajectories.

```
python code/run_v2_sweep.py
```

Output goes to `pilot_results/branch_highH_{bench}[_{slug}]_r3.json`. The
launcher skips any cell whose output already contains 100 problems, so the
sweep is safe to interrupt and resume.

### Stage 3: tables

Once Stages 1 and 2 have produced the JSON artefacts, the four reproduction
scripts read those files and emit LaTeX directly to stdout. None of the
reproduction scripts performs any inference.

| Paper table | Script | Inputs |
|---|---|---|
| Table 2 | `make_level_table.py` | `*_comprehensive_profiles.json` (Stage 1) |
| Table 3 | `make_transition_WC_WW_table.py` | `branch_highH_*_r3.json` (Stage 2) |
| Table 4 | `make_intervention_by_regime_endpoints.py` | `branch_highH_*_r3.json` (Stage 2) |
| Table 8 | `make_intervention_outcomes_quarters_table.py` | `branch_highH_*_r3.json` (Stage 2) |

```
# After Stages 1 and 2 are complete:
python code/make_level_table.py                           > tab2.tex
python code/make_transition_WC_WW_table.py                > tab3.tex
python code/make_intervention_by_regime_endpoints.py      > tab4.tex
python code/make_intervention_outcomes_quarters_table.py  > tab8.tex
```

## Hardware notes

Both `generate_profiles.py` and `branch_highH_bench_v2.py` automatically pick
the best available backend in the order:

```
MPS (Apple Silicon) > CUDA > CPU
```

Half precision (`fp16`) is used on MPS and CUDA, full precision on CPU.

The two sweep wrappers (`run_profiles_sweep.py`, `run_v2_sweep.py`) use the
`N2D2_PYTHON` environment variable for the Python interpreter, falling back
to plain `python`. Set it to the path of the interpreter inside your
virtualenv if you keep one, e.g.

```
export N2D2_PYTHON=/path/to/your/venv/bin/python
```

The sweeps are single-process by design (one model loaded at a time). On a
single A100 (40 GB) the full Stage-2 sweep takes roughly 15–20 hours; on
Apple-Silicon MPS it is closer to 24–36 hours. There are no cluster-specific
launchers in this package — adapt the sweep wrappers to your scheduler if needed.

## Notes on reproducibility

- Random seeds are fixed where stochastic decoding is used: temperature-1
  generation in Stage 1 takes a `--seed` flag (default 0) and uses a per-cell
  `torch.Generator`. The rank-k branch interventions in Stage 2 are
  deterministic given the trace.
- The 95% confidence intervals reported in `make_intervention_outcomes_quarters_table.py`
  are produced by a 5000-sample non-parametric bootstrap of the median, with
  `seed=0`.
- All pre-computed JSON artefacts can be regenerated from the pipeline scripts.
  We do not redistribute them in this package; the upstream scripts are
  fully self-contained.
