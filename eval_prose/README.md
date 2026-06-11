# eval_prose — Inference and Evaluation

This directory contains the inference and evaluation pipeline for experiments 1–6, steering vector tuning, and result visualization.

## Setup

Initialize submodules from the repo root first:

```bash
git submodule update --init --recursive
```

Then install dependencies:

```bash
pip install -r src/requirements.txt
cd ml-predict && pip install . && cd ..
```

Requires an `OPENAI_API_KEY` environment variable for LLM judge evaluation.

## Steering Vectors

Experiments require precomputed steering vectors under `src/final_vectors/qwen/` (one `.pt` file per trait). Generate these using the `persona_vectors` submodule — see `../persona_vectors/README.md`.

## Steering Vector Tuning

Tune per-trait layer and coefficient on the PROSE demo data:

```bash
# From src/
python tune_steering.py --output-json tune_results/best_settings.json
```

The resulting `best_settings.json` is used by experiments 1, 3, and 5.

## Experiments

All experiment scripts are in `src/` and write outputs to `src/experiments/`. Already-completed output files are skipped, so scripts are safe to re-run.

| Script | Description |
|--------|-------------|
| `experiment_1.py` | Single-trait ablation across all PROSE tasks and seeds |
| `experiment_2.py` | Single-trait generation on trait-specific eval questions |
| `experiment_3.py` | Two-trait combination methods (orthogonalize, diff layers, tuned mean, unit-norm mean) |
| `experiment_4.py` | Unit-norm combination methods at a fixed layer |
| `experiment_5.py` | Multi-trait combination across subset sizes 1–4 |
| `experiment_6.py` | Activation signal analysis — hidden-state projections onto trait directions |

```bash
# From src/
python experiment_1.py --best-json tune_results/best_settings.json
python experiment_2.py
python experiment_3.py
python experiment_4.py
python experiment_5.py
python experiment_6.py
```

## Scoring

```bash
# From src/
python score_experiment_1.py
python score_experiment_2.py
python score_experiment_3.py
python score_experiment_5.py

# Fast evaluation of any inference CSVs (from eval_prose/)
python evaluate_prose.py src/experiments/*.csv
python evaluate_prose.py src/experiments/*.csv --output-dir eval_results/
```

Metrics: `writing_ppm` / `writing_fpm` (GPT-4o LLM judge), `writing_BS` (BERTScore), `writing_ldist` / `writing_n_ldist` (Levenshtein).

## Visualization

```bash
# From src/
python visualize_experiments.py
```

Results are saved to `experiments/figures/`.

## Directory Structure

```
eval_prose/
├── demo_files/                # PROSE input data (5 seeds × 2 tasks)
├── ml-predict/                # PROSE evaluation dependency (git submodule)
└── src/
    ├── activation_steer.py        # Steering hook implementation
    ├── predefined_steering.py     # Trait → vector mapping and inference
    ├── prose/task_prompts.py      # Task prompt builders
    ├── tune_steering.py           # Hyperparameter tuning
    ├── visualize_experiments.py   # Result visualization
    ├── experiment_{1..6}.py       # Experiment scripts
    └── score_experiment_{1,2,3,5}.py  # Scoring scripts
```
