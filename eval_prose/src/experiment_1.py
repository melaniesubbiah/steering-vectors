#!/usr/bin/env python3
"""
Experiment 1: Single-trait ablation across all tasks and seeds.

For each trait in --best-json, runs the email_writing and summarization
tasks across all 5 PROSE seeds, applying only that one trait at its tuned
(layer, coeff) on every turn.

All prompts for a (task, seed) combo are generated in a single batched
call per trait, so the model is loaded once and each trait requires only
one pass through the data.

Output files: {output_dir}/experiment_1_{trait}_{task}_{seed}.csv
Already-completed files are skipped, so the script is safe to re-run.

Usage (from eval_prose/src/):
    python experiment_1.py
    python experiment_1.py --best-json tune_results/best_settings.json
    python experiment_1.py --traits emoji_usage formal_tone
    python experiment_1.py --tasks email_writing --seeds 1352 4792
    python experiment_1.py --num-sources 5 --batch-size 4   # quick smoke-test
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from activation_steer import ActivationSteerer
from prose.task_prompts import get_task_prompt_builder
from predefined_steering import PredefinedSteeringPipeline

SEEDS = [1352, 4792, 5961, 6584, 8337]
TASKS = ["email_writing", "summarization"]

# Maps natural-language preference strings (as they appear in true_preferences)
# to trait names, and the inverse for filtering.
PREFERENCE_TO_TRAIT = {
    "use ALLCAPS to emphasize certain words": "allcaps_emphasis",
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
    "include alliterations": "alliteration",
}
TRAIT_TO_PREFERENCE = {v: k for k, v in PREFERENCE_TO_TRAIT.items()}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _load_rows(demo_file: Path, task_prompt_builder, num_sources: int | None) -> list[dict]:
    """Load all (source, turn, prompt, metadata) rows from a PROSE demo file."""
    df = pd.read_csv(demo_file)
    sources = {
        src: grp.sort_values("source_example_num")
        for src, grp in df.groupby("task_source")
    }
    if num_sources:
        sources = dict(list(sources.items())[:num_sources])

    rows = []
    for source_name, source_df in sources.items():
        for _, row in source_df.iterrows():
            rows.append({
                "source": source_name,
                "turn": int(row["source_example_num"]),
                "task_text": row["context"],
                "task_prompt": task_prompt_builder(row["context"]),
                "true_preferences": row["true preferences"],
                "user_demo": row["user demo"],
            })
    return rows


def _filter_rows_for_trait(rows: list[dict], trait: str) -> list[dict]:
    """Return only rows where trait appears in the true_preferences field.

    true_preferences contains natural-language strings separated by '; ',
    e.g. 'use emojis; use a formal tone', so we map the trait name to its
    preference string before checking.
    """
    pref_str = TRAIT_TO_PREFERENCE.get(trait)
    if pref_str is None:
        return []
    out = []
    for row in rows:
        raw = row.get("true_preferences", "")
        if pd.isna(raw):
            continue
        prefs_in_row = [p.strip() for p in str(raw).split(";")]
        if pref_str in prefs_in_row:
            out.append(row)
    return out


def _generate_batched(
    model,
    tokenizer,
    prompts: list[str],
    steering_vector: torch.Tensor,
    layer_idx: int,
    max_new_tokens: int,
    batch_size: int,
) -> list[str]:
    """Run all prompts through model.generate() in mini-batches with one steering hook.

    Uses left-padding so sequences within a batch have uniform length.
    With positions='response' and greedy decoding (do_sample=False), the hook
    adds the vector on every decode step (seq_len=1) and is a no-op during
    the prefill, making it correct regardless of per-sequence prompt length.
    """
    orig_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    results = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        formatted = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for p in batch
        ]
        inputs = tokenizer(formatted, return_tensors="pt", padding=True).to(model.device)
        prompt_len = inputs["input_ids"].shape[1]

        with ActivationSteerer(
            model,
            steering_vector,
            layer_idx=layer_idx,
            positions="response",
            prompt_length=prompt_len,
        ):
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )

        for out in outputs:
            results.append(tokenizer.decode(out[prompt_len:], skip_special_tokens=True))

    tokenizer.padding_side = orig_padding_side
    return results


def run_single_trait(
    trait: str,
    layer: int,
    coeff: float,
    rows: list[dict],
    model,
    tokenizer,
    vector_path: Path,
    output_path: Path,
    max_new_tokens: int,
    batch_size: int,
    unit_norm: bool = False,
    alpha: float = 60.0,
    fixed_layer: int = 20,
) -> None:
    """Generate all rows for one trait and save the CSV."""
    full_vector = torch.load(vector_path, map_location="cpu", weights_only=True)
    if unit_norm:
        raw = full_vector[fixed_layer].float()
        steering_vec = (alpha * raw / raw.norm()).to(
            dtype=next(model.parameters()).dtype, device=model.device
        )
        effective_layer = fixed_layer
        effective_coeff = alpha
    else:
        steering_vec = (coeff * full_vector[layer].float()).to(
            dtype=next(model.parameters()).dtype, device=model.device
        )
        effective_layer = layer
        effective_coeff = coeff

    prompts = [r["task_prompt"] for r in rows]
    print(f"  Generating {len(prompts)} prompts in batches of {batch_size}...")
    generated = _generate_batched(
        model, tokenizer, prompts,
        steering_vector=steering_vec,
        layer_idx=effective_layer - 1,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
    )

    results = [
        {**row, "trait": trait, "layer": effective_layer, "coeff": effective_coeff, "generated_output": gen}
        for row, gen in zip(rows, generated)
    ]
    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"  Saved {len(results)} rows → {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--best-json", default="tune_results/best_settings.json",
        help="JSON with per-trait {layer, coeff} from tune_steering.py",
    )
    parser.add_argument(
        "--output-dir", default="experiments/experiment_1",
        help="Directory for output CSVs (default: experiments/experiment_1)",
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--vector-dir", default="final_vectors")
    parser.add_argument(
        "--demo-dir", default="../demo_files",
        help="Directory containing plume-*-inferring_results.csv files",
    )
    parser.add_argument("--traits", nargs="*",
                        help="Subset of traits to run (default: all in --best-json)")
    parser.add_argument("--tasks", nargs="*", choices=TASKS, default=TASKS,
                        help="Tasks to run (default: both)")
    parser.add_argument("--seeds", nargs="*", type=int, default=SEEDS,
                        help="Seeds to run (default: all 5)")
    parser.add_argument("--num-sources", type=int,
                        help="Limit number of sources per file (for quick testing)")
    parser.add_argument("--max-new-tokens", type=int, default=1000)
    parser.add_argument(
        "--batch-size", type=int, default=5,
        help="Number of prompts per model.generate() call (default: 8)",
    )
    parser.add_argument(
        "--unit-norm", action="store_true",
        help="Use unit-norm vector instead of tuned (layer, coeff)",
    )
    parser.add_argument(
        "--alpha", type=float, default=60.0,
        help="Scale factor for unit-norm mode (default: 60.0)",
    )
    parser.add_argument(
        "--fixed-layer", type=int, default=20,
        help="Layer to apply vector in unit-norm mode (default: 20)",
    )
    args = parser.parse_args()

    best_json_path = Path(args.best_json)
    if not best_json_path.exists():
        print(f"Error: --best-json not found: {best_json_path}")
        return 1

    with open(best_json_path) as f:
        best_settings: dict = json.load(f)

    traits = args.traits or list(best_settings.keys())
    missing = [t for t in traits if t not in best_settings]
    if missing:
        print(f"Warning: not in best_settings.json (skipping): {missing}")
        traits = [t for t in traits if t in best_settings]
    if not traits:
        print("No traits to run.")
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    demo_dir = Path(args.demo_dir)

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
    print("Model ready.\n")

    available_vectors = pipeline.service.available_vectors  # trait → Path

    total = len(traits) * len(args.tasks) * len(args.seeds)
    done = 0

    for task in args.tasks:
        task_prompt_builder = get_task_prompt_builder(task_name=task)
        for seed in args.seeds:
            set_seed(seed)
            demo_file = demo_dir / f"plume-{task}-prose.full-qwen-{seed}-inferring_results.csv"
            if not demo_file.exists():
                for trait in traits:
                    done += 1
                print(f"[{done}/{total}] Skipping {task}/{seed}: {demo_file} not found")
                continue

            all_rows = _load_rows(demo_file, task_prompt_builder, args.num_sources)
            print(f"Loaded {len(all_rows)} rows for {task}/seed={seed}\n")

            for trait in traits:
                done += 1
                output_path = output_dir / f"experiment_1_{trait}_{task}_{seed}.csv"
                if output_path.exists():
                    print(f"[{done}/{total}] Skip {trait} | {task} | {seed} (exists)")
                    continue

                if trait not in available_vectors:
                    print(f"[{done}/{total}] Skip {trait}: no vector file found")
                    continue

                trait_rows = _filter_rows_for_trait(all_rows, trait)
                if not trait_rows:
                    print(f"[{done}/{total}] Skip {trait} | {task} | {seed}: trait not in any true_preferences")
                    continue

                s = best_settings[trait]
                if args.unit_norm:
                    print(
                        f"[{done}/{total}] {trait} | {task} | seed={seed}"
                        f"  layer={args.fixed_layer}  alpha={args.alpha:.2f}  unit-norm"
                        f"  ({len(trait_rows)}/{len(all_rows)} rows have this trait)"
                    )
                else:
                    print(
                        f"[{done}/{total}] {trait} | {task} | seed={seed}"
                        f"  layer={s['layer']}  coeff={s['coeff']:.2f}"
                        f"  ({len(trait_rows)}/{len(all_rows)} rows have this trait)"
                    )
                run_single_trait(
                    trait=trait,
                    layer=s["layer"],
                    coeff=s["coeff"],
                    rows=trait_rows,
                    model=model,
                    tokenizer=tokenizer,
                    vector_path=available_vectors[trait],
                    output_path=output_path,
                    max_new_tokens=args.max_new_tokens,
                    batch_size=args.batch_size,
                    unit_norm=args.unit_norm,
                    alpha=args.alpha,
                    fixed_layer=args.fixed_layer,
                )

    print(f"\nDone. Results in {output_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
