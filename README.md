# On the Limits of Steering Vectors for Preference-Aligned Generation

Code for the paper "On the Limits of Steering Vectors for Preference-Aligned Generation."

## Repository Structure

```
├── eval_prose/          # Inference and evaluation pipeline (experiments 1–6)
└── persona_vectors/     # Steering vector training (git submodule)
```

## Setup

### Submodules

This repo uses two git submodules. After cloning, initialize both with:

```bash
git submodule update --init --recursive
```

- **`persona_vectors/`** — steering vector training pipeline. See `persona_vectors/README.md` for instructions on generating the vectors used by `eval_prose`.
- **`eval_prose/ml-predict/`** — PROSE evaluation framework ([apple/ml-predict](https://github.com/apple/ml-predict)), required by the scoring and tuning scripts.

### Evaluation Pipeline (eval_prose)

```bash
cd eval_prose/src
pip install -r requirements.txt
```

Requires an `OPENAI_API_KEY` environment variable for LLM judge evaluation.

## Citation

If you use this code, please cite our paper:

```bibtex
@article{...,
  title={On the Limits of Steering Vectors for Preference-Aligned Generation},
  author={...},
  year={2025}
}
```
