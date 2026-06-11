#!/usr/bin/env python3
"""
Experiment 2: Single-trait generation using the trait's own eval questions.

For each trait in --best-json, loads the corresponding questions from the
persona_vectors trait data files and generates responses with only that
trait's steering vector applied.  Each seed independently samples
--n-questions questions from the trait's pool, so results vary across seeds.

Output files: {output_dir}/experiment_2_{trait}_{seed}.csv
Already-completed files are skipped, so the script is safe to re-run.

Usage (from eval_prose/src/):
    python experiment_2.py
    python experiment_2.py --best-json tune_results/best_settings.json
    python experiment_2.py --traits emoji_usage formal_tone
    python experiment_2.py --seeds 1352 4792
    python experiment_2.py --n-questions 10 --batch-size 4   # quick smoke-test
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
from predefined_steering import PredefinedSteeringPipeline

SEEDS = [1352, 4792, 5961, 6584, 8337]

# Mirrors TRAIT_MAP from tune_steering.py: trait name → data filename stem
TRAIT_MAP = {
    "intensely_emotional": "emotional",
    "email_epithet_signoff": "epithet",
    "stream_consciousness": "streamofconsciousness",
    "allcaps_emphasis": "allcaps",
    "ampersand_usage": "ampersands",
    "archaic_language": "oldfashioned",
    "bullet_parallel": "bullets",
    "childlike": "childstory",
    "conditional_expressions": "conditionals",
    "emoji_usage": "emojis",
    "long_flowing_sentences": "longflowing",
    "modern_slang": "slang",
    "old_timey_radio": "radio",
    "open_with_movie_ref": "movieref",
    "parenthetical_asides": "parentheticals",
    "podcast_style": "podcast",
    "question_answering_style": "questionanswer",
    "rhetorical_questions": "rhetorical",
    "personification": "personifications",
    "rhyming_structure": "rhyming",
    "second_person_narrative": "secondperson",
    "semicolon_usage": "semicolons",
    "short_punchy_sentences": "shortpunchy",
    "simile_usage": "similes",
    "step_by_step": "steps",
    "third_person_perspective": "thirdperson",
    "tweet_style": "tweet",
    "vivid_imagery": "imagery",
    "formal_tone": "formal",
    "header_structured": "headers",
    "sarcastic": "sarcasm",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _load_questions(trait: str, trait_data_dir: Path) -> list[str] | None:
    """Return the questions list for a trait, or None if no data file exists."""
    filetrait = TRAIT_MAP.get(trait, trait)
    data_path = trait_data_dir / f"{filetrait}.json"
    if not data_path.exists():
        return None
    with open(data_path) as f:
        return json.load(f)["questions"]


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
    With positions='response' and greedy decoding, the hook adds the vector
    on every decode step and is a no-op during prefill, so mixed-length
    prompts in the same batch are handled correctly.
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


def run_single_trait_seed(
    trait: str,
    layer: int,
    coeff: float,
    questions: list[str],
    seed: int,
    n_questions: int,
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
    """Sample questions, generate outputs for one (trait, seed), and save the CSV."""
    set_seed(seed)
    sampled = random.sample(questions, min(n_questions, len(questions)))

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

    print(f"  Generating {len(sampled)} questions in batches of {batch_size}...")
    generated = _generate_batched(
        model, tokenizer, sampled,
        steering_vector=steering_vec,
        layer_idx=effective_layer - 1,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
    )

    results = [
        {
            "trait": trait,
            "layer": effective_layer,
            "coeff": effective_coeff,
            "seed": seed,
            "question_idx": i,
            "question": q,
            "generated_output": gen,
        }
        for i, (q, gen) in enumerate(zip(sampled, generated))
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
        "--output-dir", default="experiments/experiment_2",
        help="Directory for output CSVs (default: experiments/experiment_2)",
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--vector-dir", default="final_vectors")
    parser.add_argument(
        "--trait-data-dir",
        default="../../persona_vectors/data_generation/trait_data_eval",
        help="Directory containing per-trait question JSON files",
    )
    parser.add_argument("--traits", nargs="*",
                        help="Subset of traits to run (default: all in --best-json)")
    parser.add_argument("--seeds", nargs="*", type=int, default=SEEDS,
                        help="Seeds to run (default: all 5)")
    parser.add_argument(
        "--n-questions", type=int, default=5,
        help="Questions to sample per (trait, seed) (default: 40, i.e. all)",
    )
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

    trait_data_dir = Path(args.trait_data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    available_vectors = pipeline.service.available_vectors

    total = len(traits) * len(args.seeds)
    done = 0

    for trait in traits:
        questions = _load_questions(trait, trait_data_dir)
        if questions is None:
            for _ in args.seeds:
                done += 1
            print(f"[{done}/{total}] Skip {trait}: no trait data file found")
            continue

        if trait not in available_vectors:
            for _ in args.seeds:
                done += 1
            print(f"[{done}/{total}] Skip {trait}: no vector file found")
            continue

        s = best_settings[trait]
        for seed in args.seeds:
            done += 1
            output_path = output_dir / f"experiment_2_{trait}_{seed}.csv"
            if output_path.exists():
                print(f"[{done}/{total}] Skip {trait} | seed={seed} (exists)")
                continue

            if args.unit_norm:
                print(
                    f"[{done}/{total}] {trait} | seed={seed}"
                    f"  layer={args.fixed_layer}  alpha={args.alpha:.2f}  unit-norm"
                    f"  ({len(questions)} questions available)"
                )
            else:
                print(
                    f"[{done}/{total}] {trait} | seed={seed}"
                    f"  layer={s['layer']}  coeff={s['coeff']:.2f}"
                    f"  ({len(questions)} questions available)"
                )
            run_single_trait_seed(
                trait=trait,
                layer=s["layer"],
                coeff=s["coeff"],
                questions=questions,
                seed=seed,
                n_questions=args.n_questions,
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
