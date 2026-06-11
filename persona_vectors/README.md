# persona_vectors

This directory is a git submodule pointing to the **PersonaVectors** repository, which contains code for generating the steering vectors used in the experiments.

## Submodule Setup

```bash
# From the repo root, after cloning:
git submodule update --init --recursive
```

Or clone with submodules from the start:

```bash
git clone --recurse-submodules <repo-url>
```

## Generating Steering Vectors

The `eval_prose` experiments expect steering vectors under `eval_prose/src/final_vectors/{qwen,llama}/`. Generate them from the PersonaVectors pipeline:

```bash
cd persona_vectors

# 1. Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # fill in API keys

# 2. Generate trait data (positive/negative system-prompt evaluations)
bash scripts/eval_persona.sh 0   # GPU 0

# 3. Compute mean-difference vectors per trait
bash scripts/generate_vec.sh 0

# The resulting *_response_avg_diff.pt files are the vectors used in experiments.
```

After generation, copy or symlink the vectors into `eval_prose/src/final_vectors/qwen/`:

```bash
cp persona_vectors/persona_vectors/Qwen2.5-7B-Instruct_from_gpt/*_response_avg_diff.pt \
   eval_prose/src/final_vectors/qwen/
```

See the PersonaVectors repository for full documentation.
