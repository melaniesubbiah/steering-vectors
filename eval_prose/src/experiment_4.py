#!/usr/bin/env python3
"""
Experiment 4: Unit-norm combination methods for two steering vectors.

Like experiment_3, iterates over all unique 4-trait preference groups found
in the PROSE demo files and tests all C(n,2) trait pairs.  Unlike experiment_3,
ALL methods use unit-normalised vectors extracted at a single fixed layer,
scaled by a single fixed alpha — no tuned (layer, coeff) settings are needed.

Methods:
  orthogonalize  – Gram-Schmidt: unit_norm(vec2) ⊥ unit_norm(vec1), then
                   apply alpha * (unit1 + unit2_orth_renormed) at fixed_layer.
  mean           – Apply alpha * (unit1 + unit2) / 2 at fixed_layer.
  diff_layers    – Apply alpha * unit1 at (fixed_layer - layer_offset) and
                   alpha * unit2 at (fixed_layer + layer_offset) as separate hooks.
  sum            – Apply alpha * (unit1 + unit2) at fixed_layer.

Two prompts are generated per pair: one sampled from each trait's own
persona question pool.  Groups are de-duplicated across all PROSE files
so each unique group is run exactly once.

Output files: {output_dir}/experiment_4_{group_id}.csv
Already-completed files are skipped, so the script is safe to re-run.
A manifest JSON is written alongside the CSVs mapping group_id → traits.

Usage (from eval_prose/src/):
    python experiment_4.py
    python experiment_4.py --alpha 70 --fixed-layer 20
    python experiment_4.py --layer-offset 2 --num-sources 2   # quick smoke-test
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

import pandas as pd
import torch

from activation_steer import ActivationSteererMultiple
from predefined_steering import PredefinedSteeringPipeline

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


# ── Data helpers ──────────────────────────────────────────────────────────────

def collect_unique_groups(demo_dir: Path, num_sources: int | None = None) -> dict:
    """Scan all PROSE demo files; return {group_id: [trait, ...]} for unique groups."""
    seen: set[frozenset] = set()
    groups: dict[str, list[str]] = {}

    for task in TASKS:
        for seed in SEEDS:
            f = demo_dir / f"plume-{task}-prose.full-qwen-{seed}-inferring_results.csv"
            if not f.exists():
                continue
            df = pd.read_csv(f)
            if num_sources is not None:
                # Limit by unique task sources for quick smoke-tests
                sources = df["task_source"].unique()[:num_sources]
                df = df[df["task_source"].isin(sources)]
            for prefs_str in df["true preferences"].drop_duplicates():
                pref_strs = [p.strip() for p in str(prefs_str).split(";") if p.strip()]
                traits = [PREFERENCE_TO_TRAIT[p] for p in pref_strs if p in PREFERENCE_TO_TRAIT]
                if len(traits) < 2:
                    continue
                key = frozenset(traits)
                if key in seen:
                    continue
                seen.add(key)
                sorted_traits = sorted(traits)
                group_id = "__".join(sorted_traits)
                groups[group_id] = sorted_traits

    return groups


def load_questions(trait: str, trait_data_dir: Path) -> list[str]:
    filetrait = TRAIT_MAP.get(trait, trait)
    path = trait_data_dir / f"{filetrait}.json"
    if not path.exists():
        return []
    return json.loads(path.read_text())["questions"]


def pick_question(trait: str, questions: list[str]) -> str:
    if not questions:
        return f"Tell me something about {trait}."
    idx = sum(ord(c) for c in trait) % len(questions)
    return questions[idx]


# ── Vector helpers ────────────────────────────────────────────────────────────

def load_unit_vec(trait: str, fixed_layer: int, available_vectors: dict) -> torch.Tensor | None:
    """Load the vector at fixed_layer from the saved tensor file and unit-normalise it."""
    if trait not in available_vectors:
        return None
    full = torch.load(available_vectors[trait], map_location="cpu", weights_only=True)
    raw = full[fixed_layer].float()
    return raw / raw.norm()


def gram_schmidt_orth(v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
    """Return v2 with its component along v1 removed (v1 assumed unit norm)."""
    return v2 - (v2 @ v1) * v1


# ── Instruction builders ──────────────────────────────────────────────────────

def build_instructions(
    method: str,
    unit1: torch.Tensor,
    unit2: torch.Tensor,
    alpha: float,
    fixed_layer: int,
    layer_offset: int,
) -> list[list[dict]]:
    """Return one instruction list per prompt (always 2 lists, one per trait prompt).

    All vectors are pre-scaled; ActivationSteerer adds them as-is (coeff=1.0).
    """
    if method == "orthogonalize":
        unit2_orth = gram_schmidt_orth(unit1, unit2)
        norm = unit2_orth.norm()
        if norm < 1e-6:
            # vec2 was nearly parallel to vec1 — fall back to unit2
            unit2_orth_normed = unit2
        else:
            unit2_orth_normed = unit2_orth / norm
        combined = alpha * (unit1 + unit2_orth_normed)
        shared = [{"steering_vector": combined, "layer_idx": fixed_layer - 1}]
        return [shared, shared]

    elif method == "mean":
        combined = alpha * (unit1 + unit2) / 2
        shared = [{"steering_vector": combined, "layer_idx": fixed_layer - 1}]
        return [shared, shared]

    elif method == "diff_layers":
        shared = [
            {"steering_vector": alpha * unit1, "layer_idx": fixed_layer - layer_offset - 1},
            {"steering_vector": alpha * unit2, "layer_idx": fixed_layer + layer_offset - 1},
        ]
        return [shared, shared]

    elif method == "sum":
        combined = alpha * (unit1 + unit2)
        shared = [{"steering_vector": combined, "layer_idx": fixed_layer - 1}]
        return [shared, shared]

    raise ValueError(f"Unknown method: {method}")


# ── Generation ────────────────────────────────────────────────────────────────

def generate_with_instructions(
    model,
    tokenizer,
    prompts: list[str],
    instructions: list[dict],
    max_new_tokens: int,
) -> list[str]:
    tokenizer.padding_side = "left"
    formatted = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in prompts
    ]
    inputs = tokenizer(formatted, return_tensors="pt", padding=True).to(model.device)
    prompt_len = inputs["input_ids"].shape[1]

    full_instructions = [
        {**inst, "coeff": 1.0, "positions": "response", "prompt_length": prompt_len}
        for inst in instructions
    ]

    with ActivationSteererMultiple(model, full_instructions):
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )

    return [tokenizer.decode(out[prompt_len:], skip_special_tokens=True) for out in outputs]


# ── Per-group runner ──────────────────────────────────────────────────────────

METHODS = ["orthogonalize", "mean", "diff_layers", "sum"]


def run_group(
    group_id: str,
    traits: list[str],
    model,
    tokenizer,
    available_vectors: dict,
    trait_data_dir: Path,
    output_path: Path,
    alpha: float,
    fixed_layer: int,
    layer_offset: int,
    max_new_tokens: int,
) -> None:
    rows = []
    dtype = next(model.parameters()).dtype

    for trait1, trait2 in itertools.combinations(traits, 2):
        unit1 = load_unit_vec(trait1, fixed_layer, available_vectors)
        unit2 = load_unit_vec(trait2, fixed_layer, available_vectors)
        if unit1 is None or unit2 is None:
            missing = trait1 if unit1 is None else trait2
            print(f"    Skip pair ({trait1}, {trait2}): no vector for {missing}")
            continue

        unit1 = unit1.to(dtype=dtype, device=model.device)
        unit2 = unit2.to(dtype=dtype, device=model.device)

        q1s = load_questions(trait1, trait_data_dir)
        q2s = load_questions(trait2, trait_data_dir)
        q1 = pick_question(trait1, q1s)
        q2 = pick_question(trait2, q2s)
        prompt_traits = [trait1, trait2]
        questions = [q1, q2]

        print(f"    ({trait1}, {trait2})")

        def _add_rows(method_name, per_prompt_instructions):
            for pt, q, instr in zip(prompt_traits, questions, per_prompt_instructions):
                out = generate_with_instructions(
                    model, tokenizer, [q], instr, max_new_tokens
                )[0]
                rows.append({
                    "group": group_id,
                    "trait1": trait1,
                    "trait2": trait2,
                    "method": method_name,
                    "alpha": alpha,
                    "fixed_layer": fixed_layer,
                    "prompt_trait": pt,
                    "question": q,
                    "generated_output": out,
                })

        for method in METHODS:
            per_prompt_instructions = build_instructions(
                method, unit1, unit2, alpha, fixed_layer, layer_offset
            )
            _add_rows(method, per_prompt_instructions)

    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"    Saved {len(rows)} rows → {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--output-dir", default="experiments/experiment_4",
        help="Directory for output CSVs (default: experiments/experiment_4)",
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--vector-dir", default="final_vectors")
    parser.add_argument(
        "--demo-dir", default="../demo_files",
        help="Directory containing plume-*-inferring_results.csv files",
    )
    parser.add_argument(
        "--trait-data-dir",
        default="../../persona_vectors/data_generation/trait_data_eval",
        help="Directory containing per-trait question JSON files",
    )
    parser.add_argument(
        "--alpha", type=float, default=60.0,
        help="Scale factor applied after combination (default: 60.0)",
    )
    parser.add_argument(
        "--fixed-layer", type=int, default=20,
        help="Layer at which vectors are extracted and applied (default: 20)",
    )
    parser.add_argument(
        "--layer-offset", type=int, default=2,
        help="Offset from fixed_layer for the diff_layers method: "
             "trait1 at (fixed_layer - offset), trait2 at (fixed_layer + offset) "
             "(default: 2)",
    )
    parser.add_argument("--max-new-tokens", type=int, default=500)
    parser.add_argument(
        "--num-sources", type=int,
        help="Limit number of task sources per demo file (for quick testing)",
    )
    parser.add_argument(
        "--num-threads", type=int, default=4,
        help="PyTorch CPU thread limit (default: 4). Lower this when sharing a machine.",
    )
    args = parser.parse_args()

    torch.set_num_threads(args.num_threads)
    print(f"PyTorch CPU threads: {args.num_threads}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    demo_dir = Path(args.demo_dir)
    trait_data_dir = Path(args.trait_data_dir)

    print("Scanning PROSE demo files for unique trait groups...")
    groups = collect_unique_groups(demo_dir, num_sources=args.num_sources)
    print(f"Found {len(groups)} unique groups.\n")

    manifest_path = output_dir / "groups_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(groups, f, indent=2)
    print(f"Manifest written to {manifest_path}\n")

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
    print(
        f"Model ready. alpha={args.alpha}  fixed_layer={args.fixed_layer}"
        f"  layer_offset={args.layer_offset} (diff_layers: layers"
        f" {args.fixed_layer - args.layer_offset} & {args.fixed_layer + args.layer_offset})\n"
    )

    available_vectors = pipeline.service.available_vectors

    for i, (group_id, traits) in enumerate(groups.items(), 1):
        output_path = output_dir / f"experiment_4_{group_id}.csv"
        if output_path.exists():
            print(f"[{i}/{len(groups)}] Skip {group_id} (exists)")
            continue

        print(f"[{i}/{len(groups)}] Group: {traits}")
        run_group(
            group_id=group_id,
            traits=traits,
            model=model,
            tokenizer=tokenizer,
            available_vectors=available_vectors,
            trait_data_dir=trait_data_dir,
            output_path=output_path,
            alpha=args.alpha,
            fixed_layer=args.fixed_layer,
            layer_offset=args.layer_offset,
            max_new_tokens=args.max_new_tokens,
        )

    print(f"\nDone. Results in {output_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
