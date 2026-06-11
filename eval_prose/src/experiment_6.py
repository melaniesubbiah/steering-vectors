#!/usr/bin/env python3
"""
Experiment 6: Activation signal analysis — does the demo's representation
predict which layer / coefficient steers it best?

For each PROSE user demo row, one forward pass is run on the full
(task prompt + user demo) text.  Hidden states are extracted for every
demo response token at each layer and averaged across token positions,
giving a single hidden vector h_demo per (row, layer).

That average is then projected onto each trait's unit steering vector:

  proj_demo = h_demo · v̂_t   (= mean_over_tokens [ h_t · v̂_t ])

A boolean column 'trait_present' flags whether the trait appeared in the
row's preferences, enabling contrastive analysis between demos that do and
do not exhibit a given trait.

Columns saved per (row, trait, layer):
  proj_demo      – mean token projection onto unit trait direction
  cos_sim_demo   – cosine similarity between mean demo hidden state and v_t
  norm_demo      – ‖mean demo hidden state‖
  vec_norm       – ‖steering vector‖ at this layer
  trait_present  – whether the trait is in the row's true preferences
  best_layer / best_coeff – from best_settings.json

For plots run:
    python visualize_experiments.py --exp6-csv {output_dir}/experiment_6_signals.csv

Usage (from eval_prose/src/):
    python experiment_6.py
    python experiment_6.py --best-json tune_results/best_settings.json
    python experiment_6.py --layers 16 20
    python experiment_6.py --num-sources 3   # quick smoke-test
    python experiment_6.py --model meta-llama/Llama-3.1-8B-Instruct
"""


import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from predefined_steering import PredefinedSteeringPipeline
from prose.task_prompts import get_task_prompt_builder

# ── Constants ─────────────────────────────────────────────────────────────────

TASKS = ["email_writing", "summarization"]
SEEDS = [1352, 4792, 5961, 6584, 8337]

PREFERENCE_TO_TRAIT = {
    "use ALLCAPS to emphasize certain words": "allcaps_emphasis",
    "include alliterations": "alliteration",
    'use ampersands (&) instead of "and"s': "ampersand_usage",
    "use archaic language": "archaic_language",
    "write using assertive expressions": "assertive",
    "write using bullet points": "bullet_parallel",
    "write in the style of a children's book": "childlike",
    "write using conditional expressions": "conditional_expressions",
    "be sharply critical": "critical",
    "use emojis": "emoji_usage",
    "use a formal tone": "formal_tone",
    "adopt a header and sub-header structure": "header_structured",
    "include hyperboles": "hyperbole",
    "use an informal tone": "informal",
    "be highly inquisitive": "inquisitive",
    "be intensely emotional": "intensely_emotional",
    "include several long and flowing sentences": "long_flowing_sentences",
    "include modern slang": "modern_slang",
    "write in the style of old timey radio": "old_timey_radio",
    "include onomatopoeias": "onomatopoeia",
    "use parenthetical asides": "parenthetical_asides",
    "include personifications": "personification",
    "write in the style of a podcast": "podcast_style",
    "adopt a question-answering style structure": "question_answering_style",
    "include rhetorical questions": "rhetorical_questions",
    "adopt a rhyming structure": "rhyming_structure",
    "be blatantly sarcastic": "sarcastic",
    "write in the style of a screenplay": "screenplay",
    "write in a second person narrative": "second_person_narrative",
    "include several short and punchy sentences": "short_punchy_sentences",
    "include a simile": "simile_usage",
    "adopt a step-by-step structure": "step_by_step",
    "write using a stream-of-consciousness style": "stream_consciousness",
    "write in a third person perspective": "third_person_perspective",
    "write in the style of a tweet": "tweet_style",
    "use imagery": "vivid_imagery",
}

# ── Model / vector helpers ─────────────────────────────────────────────────────

def _get_layer_list(model):
    for path in ("model.layers", "transformer.h", "gpt_neox.layers", "block"):
        cur = model
        for part in path.split("."):
            if hasattr(cur, part):
                cur = getattr(cur, part)
            else:
                break
        else:
            if hasattr(cur, "__getitem__"):
                return cur
    raise ValueError("Cannot locate transformer layer list.")


def _load_vec(trait: str, layer: int, available_vectors: dict) -> torch.Tensor | None:
    if trait not in available_vectors:
        return None
    full = torch.load(available_vectors[trait], map_location="cpu", weights_only=True)
    if layer >= len(full):
        return None
    return full[layer].float()


# ── Activation collection ─────────────────────────────────────────────────────

@torch.no_grad()
def _collect_demo_avg_hidden_states(
    model,
    tokenizer,
    prompt_texts: list[str],
    demo_texts: list[str],
    layer_indices: list[int],
) -> dict[int, torch.Tensor]:
    """
    For each (prompt_text, demo_text) pair, run a forward pass on demo_text
    and capture the mean hidden state over all demo response token positions
    at each layer (0-indexed).

    prompt_text is tokenized separately to determine where the response starts.
    Returns {layer_idx: Tensor[n_texts, hidden_size]} on CPU.
    """
    layers = _get_layer_list(model)
    results: dict[int, list[torch.Tensor]] = {li: [] for li in layer_indices}

    for prompt_text, demo_text in zip(prompt_texts, demo_texts):
        prompt_len = len(tokenizer.encode(prompt_text, add_special_tokens=False))

        captured: dict[int, torch.Tensor | None] = {li: None for li in layer_indices}
        handles = []

        def _make_hook(li):
            def _hook(module, inp, out):
                hidden = out[0] if isinstance(out, (tuple, list)) else out
                # Average over all response token positions
                resp = hidden[0, prompt_len:, :].float()
                if resp.shape[0] > 0:
                    captured[li] = resp.mean(dim=0).detach().cpu()
                else:
                    captured[li] = hidden[0, -1, :].float().detach().cpu()
            return _hook

        for li in layer_indices:
            handles.append(layers[li].register_forward_hook(_make_hook(li)))

        try:
            enc = tokenizer(
                demo_text, return_tensors="pt",
                truncation=True, max_length=2048,
            ).to(model.device)
            model(**enc)
        finally:
            for h in handles:
                h.remove()

        for li in layer_indices:
            results[li].append(captured[li])

    return {li: torch.stack(results[li]) for li in layer_indices}


# ── Signal computation ────────────────────────────────────────────────────────

def _signals(h: torch.Tensor, v: torch.Tensor) -> dict:
    """
    proj = (h · v̂) = cos_sim × ‖h‖  — component of h along unit steering
    direction, in activation-space units.
    """
    norm_h = float(h.norm())
    norm_v = float(v.norm())
    if norm_h < 1e-8 or norm_v < 1e-8:
        return {"cos_sim": float("nan"), "proj": float("nan"), "norm": norm_h}
    dot = float(torch.dot(h.float(), v.float()))
    return {"cos_sim": dot / (norm_h * norm_v), "proj": dot / norm_v, "norm": norm_h}


# ── Main collection loop ──────────────────────────────────────────────────────

def run(
    tasks: list[str],
    seeds: list[int],
    layers: list[int],
    demo_dir: Path,
    best_settings: dict,
    available_vectors: dict,
    model,
    tokenizer,
    num_sources: int | None,
    output_path: Path,
) -> None:
    layer_indices = [l - 1 for l in layers]  # 1-indexed → 0-indexed for hooks
    all_traits = [t for t in best_settings if t in available_vectors]

    # ── Phase 1: collect all activations across every (task, seed) file ────────
    all_meta: list[dict] = []
    all_h_demo: dict[int, list[torch.Tensor]] = {li: [] for li in layer_indices}

    for task in tasks:
        for seed in seeds:
            demo_file = demo_dir / f"plume-{task}-prose.full-qwen-{seed}-inferring_results.csv"
            if not demo_file.exists():
                print(f"  Missing: {demo_file.name}")
                continue

            df = pd.read_csv(demo_file)
            sources = {
                src: grp.sort_values("source_example_num")
                for src, grp in df.groupby("task_source")
            }
            if num_sources:
                sources = dict(list(sources.items())[:num_sources])

            file_rows = [
                {
                    "task":             task,
                    "seed":             seed,
                    "source":           src,
                    "turn":             int(row["source_example_num"]),
                    "task_text":        str(row["context"]),
                    "user_demo":        str(row["user demo"]),
                    "true_preferences": str(row["true preferences"]),
                }
                for src, src_df in sources.items()
                for _, row in src_df.iterrows()
            ]
            if not file_rows:
                continue

            print(f"  {task}  seed={seed}  {len(file_rows)} rows", flush=True)

            prompt_texts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": r["task_text"]}],
                    tokenize=False, add_generation_prompt=True,
                )
                for r in file_rows
            ]
            demo_texts = [
                tokenizer.apply_chat_template(
                    [
                        {"role": "user",      "content": r["task_text"]},
                        {"role": "assistant", "content": r["user_demo"]},
                    ],
                    tokenize=False, add_generation_prompt=False,
                )
                for r in file_rows
            ]

            h_demos = _collect_demo_avg_hidden_states(
                model, tokenizer, prompt_texts, demo_texts, layer_indices)

            for i in range(len(file_rows)):
                all_meta.append(file_rows[i])
                for li in layer_indices:
                    all_h_demo[li].append(h_demos[li][i])

    if not all_meta:
        print("No rows collected — check demo_dir path.")
        return

    print(f"\nCollected {len(all_meta)} rows total across all files.")

    # ── Phase 2: compute signals for every (row, trait, layer) ────────────────
    print(f"Computing signals for {len(all_meta)} rows × {len(all_traits)} traits × {len(layers)} layers...")
    rows_out = []

    for i, r in enumerate(tqdm(all_meta, desc="rows", ncols=70)):
        prefs = [p.strip() for p in r["true_preferences"].split(";") if p.strip()]
        present_traits = {
            PREFERENCE_TO_TRAIT[p] for p in prefs if p in PREFERENCE_TO_TRAIT
        }

        for trait in all_traits:
            s = best_settings[trait]

            for layer, li in zip(layers, layer_indices):
                vec = _load_vec(trait, layer, available_vectors)
                if vec is None:
                    continue

                h_d = all_h_demo[li][i]
                sigs = _signals(h_d, vec)

                row_out: dict = {
                    "task":          r["task"],
                    "seed":          r["seed"],
                    "source":        r["source"],
                    "turn":          r["turn"],
                    "trait":         trait,
                    "trait_present": trait in present_traits,
                    "best_layer":    s["layer"],
                    "best_coeff":    s["coeff"],
                    "layer":         layer,
                    "proj_demo":     sigs["proj"],
                    "cos_sim_demo":  sigs["cos_sim"],
                    "norm_demo":     sigs["norm"],
                    "vec_norm":      float(vec.norm()),
                }
                rows_out.append(row_out)

    pd.DataFrame(rows_out).to_csv(output_path, index=False)
    print(f"Saved {len(rows_out)} rows → {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--best-json", default="tune_results/best_settings.json",
        help="JSON with per-trait {layer, coeff} from tune_steering.py",
    )
    parser.add_argument(
        "--output-dir", default="experiments/experiment_6",
        help="Directory for output CSV (default: experiments/experiment_6/)",
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--vector-dir", default="final_vectors")
    parser.add_argument(
        "--demo-dir", default="../demo_files",
        help="Directory containing plume-*-inferring_results.csv files",
    )
    parser.add_argument(
        "--layers", nargs="+", type=int, default=[16, 20],
        help="Layers at which to collect activations, 1-indexed (default: 16 20)",
    )
    parser.add_argument(
        "--num-sources", type=int, default=None,
        help="Limit sources per demo file (for quick testing)",
    )
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--tasks", nargs="+", default=TASKS, choices=TASKS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing output file")
    args = parser.parse_args()

    torch.set_num_threads(args.num_threads)

    best_json_path = Path(args.best_json)
    if not best_json_path.exists():
        print(f"Error: --best-json not found: {best_json_path}")
        return 1
    with open(best_json_path) as f:
        best_settings: dict = json.load(f)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "experiment_6_signals.csv"

    if output_path.exists() and not args.force:
        print(f"Output already exists: {output_path}\nUse --force to overwrite.")
        return 0

    print(f"Initializing pipeline ({args.model})...")
    pipeline = PredefinedSteeringPipeline(
        model_name=args.model,
        device="auto",
        vector_dir=args.vector_dir,
        framework="plume",
        use_prompting=False,
        enable_evaluation=False,
    )
    model, tokenizer = pipeline.service.model_loader.load_model(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.eval()
    available_vectors = pipeline.service.available_vectors
    print(f"Model ready. Collecting activations at layers {args.layers}.\n")

    run(
        tasks=args.tasks,
        seeds=args.seeds,
        layers=args.layers,
        demo_dir=Path(args.demo_dir),
        best_settings=best_settings,
        available_vectors=available_vectors,
        model=model,
        tokenizer=tokenizer,
        num_sources=args.num_sources,
        output_path=output_path,
    )

    print(f"\nDone. For plots:\n  python visualize_experiments.py --exp6-csv {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
