#!/usr/bin/env python3
"""
Experiment 3: Compare methods for combining two steering vectors.

For every unique 4-trait preference group found in the PROSE demo files,
tests all C(n,2) pairs of traits with four combination methods:

  1  orthogonalize   – Gram-Schmidt: project vec2 ⊥ vec1, then apply each
                       at its own tuned (layer, coeff).
  2  different_layers – Apply each vector at its own tuned (layer, coeff)
                        via separate hooks (ActivationSteererMultiple).
  4  tuned_mean       – Same as 2, but if both share the same tuned layer,
                        average the coeff-scaled vectors and apply once.
  5  unit_norm_mean   – Normalise each to unit norm, average, apply at
                        layer 20 with alpha ∈ {10.0, 20.0, 30.0}.

Two prompts are generated per pair: one sampled from each trait's own
persona question pool.  Groups are de-duplicated across all PROSE files
so each unique group is run exactly once.

Output files: {output_dir}/experiment_3_{group_id}.csv
Already-completed files are skipped, so the script is safe to re-run.
A manifest JSON is written alongside the CSVs mapping group_id → traits.

Usage (from eval_prose/src/):
    python experiment_3.py
    python experiment_3.py --output-dir experiments/experiment_3
    python experiment_3.py --num-sources 2   # quick smoke-test
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
ALPHAS_llama = {
    16: 4.0,
    20: 8.0
}
ALPHAS_qwen = {
    16: 30.0,
    20: 58.0
}

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

def collect_unique_groups(demo_dir: Path) -> dict[str, list[str]]:
    """Scan all PROSE demo files; return {group_id: [trait, ...]} for unique groups."""
    seen: set[frozenset] = set()
    groups: dict[str, list[str]] = {}

    for task in TASKS:
        for seed in SEEDS:
            f = demo_dir / f"plume-{task}-prose.full-qwen-{seed}-inferring_results.csv"
            if not f.exists():
                continue
            df = pd.read_csv(f)
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
    """Pick one question deterministically from a trait's question pool."""
    if not questions:
        return f"Tell me something about {trait}."
    # Use a stable index derived from the trait name so different traits
    # in the same group pull from different positions in their respective lists.
    idx = sum(ord(c) for c in trait) % len(questions)
    return questions[idx]


# ── Vector helpers ────────────────────────────────────────────────────────────

def load_vec(trait: str, layer: int, available_vectors: dict) -> torch.Tensor | None:
    if trait not in available_vectors:
        return None
    full = torch.load(available_vectors[trait], map_location="cpu", weights_only=True)
    return full[layer].float()


def gram_schmidt_orth(v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
    """Return v2 with its component along v1 removed."""
    v1_unit = v1 / v1.norm()
    return v2 - (v2 @ v1_unit) * v1_unit


def _consolidate(instructions: list[dict], use_sum: bool = False) -> list[dict]:
    """Merge instructions that share the same layer_idx by averaging (or summing) their vectors.

    ActivationSteererMultiple registers one hook per instruction, so two
    instructions at the same layer would fire independently and sum.  Taking
    the mean (or sum if use_sum=True) ensures exactly one vector is applied per layer.
    """
    from collections import defaultdict
    buckets: dict[int, list[torch.Tensor]] = defaultdict(list)
    for inst in instructions:
        buckets[inst["layer_idx"]].append(inst["steering_vector"])
    return [
        {
            "steering_vector": torch.stack(vecs).sum(dim=0) if use_sum else torch.stack(vecs).mean(dim=0),
            "layer_idx": layer_idx,
        }
        for layer_idx, vecs in sorted(buckets.items())
    ]


def build_instructions(
    trait1,
    trait2,
    model,
    method_id: int,
    layer1: int, layer2: int,
    coeff1: float, coeff2: float,
    available_vectors: dict,
    alpha: float | None = None,
    is_llama: bool = False,
) -> list[list[dict]]:
    """Return one instruction set per prompt (always a list of 2 lists).

    Most methods use the same instructions for both prompts.  Method 2 when
    both traits share the same tuned layer produces different instructions per
    prompt (the layers 16/20 assignment is swapped between prompts).

    Vectors are pre-scaled; ActivationSteerer ignores its coeff kwarg and
    adds the vector as-is, so scaling is done here.

    All instruction lists are passed through _consolidate() so there is never
    more than one vector applied per layer.
    """
    vec1 = load_vec(trait1, layer1, available_vectors)
    vec2 = load_vec(trait2, layer2, available_vectors)
    if vec1 is None or vec2 is None:
        print(f"    Skip pair ({trait1}, {trait2}): missing vector file")
        return None

    dtype = next(model.parameters()).dtype
    vec1 = vec1.to(dtype=dtype, device=model.device)
    vec2 = vec2.to(dtype=dtype, device=model.device)

    if method_id == 5:
        shared = _consolidate([
            {"steering_vector": alpha * vec1 / vec1.norm(), "layer_idx": layer1 - 1},
            {"steering_vector": alpha * vec2 / vec2.norm(), "layer_idx": layer2 - 1},
        ])
        return shared

    elif layer1 != layer2:
        shared = [
            {"steering_vector": coeff1 * vec1, "layer_idx": layer1 - 1},
            {"steering_vector": coeff2 * vec2, "layer_idx": layer2 - 1},
        ]
        return shared

    elif method_id == 1:
        v2_orth = gram_schmidt_orth(vec1, vec2)
        shared = _consolidate([
            {"steering_vector": coeff1 * vec1,    "layer_idx": layer1 - 1},
            {"steering_vector": coeff2 * v2_orth, "layer_idx": layer2 - 1},
        ], use_sum=True)
        return shared

    elif method_id == 2:
        layer2 -= 1
        vec2 = load_vec(trait2, layer2, available_vectors)
        if vec2 is None:
            print(f"    Skip pair ({trait1}, {trait2}): missing vector file")
            return None
        dtype = next(model.parameters()).dtype
        vec2 = vec2.to(dtype=dtype, device=model.device)
        shared = [
            {"steering_vector": coeff1 * vec1, "layer_idx": layer1 - 1},
            {"steering_vector": coeff2 * vec2, "layer_idx": layer2 - 1},
        ]
        return shared

    elif method_id == 4:
        shared = _consolidate([
            {"steering_vector": coeff1 * vec1, "layer_idx": layer1 - 1},
            {"steering_vector": coeff2 * vec2, "layer_idx": layer2 - 1},
        ], use_sum=False)
        return shared

    raise ValueError(f"Unknown method_id: {method_id}")


# ── Generation ────────────────────────────────────────────────────────────────

def generate_with_instructions(
    model,
    tokenizer,
    prompts: list[str],
    instructions: list[dict],
    max_new_tokens: int,
) -> list[str]:
    """Batch-generate all prompts with a shared set of steering instructions."""
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

def run_group(
    group_id: str,
    traits: list[str],
    best_settings: dict,
    model,
    tokenizer,
    available_vectors: dict,
    trait_data_dir: Path,
    output_path: Path,
    max_new_tokens: int,
    is_llama: bool,
) -> None:
    rows = []

    for trait1, trait2 in itertools.combinations(traits, 2):
        # Skip pairs where settings or vectors are missing
        if trait1 not in best_settings or trait2 not in best_settings:
            print(f"    Skip pair ({trait1}, {trait2}): missing best_settings entry")
            continue

        s1, s2 = best_settings[trait1], best_settings[trait2]
        layer1, coeff1 = s1["layer"], s1["coeff"]
        layer2, coeff2 = s2["layer"], s2["coeff"]



        q1s = load_questions(trait1, trait_data_dir)
        q2s = load_questions(trait2, trait_data_dir)
        q1 = pick_question(trait1, q1s)
        q2 = pick_question(trait2, q2s)
        prompt_traits = [trait1, trait2]
        questions = [q1, q2]

        print(f"    ({trait1}, {trait2})")

        def _add_rows(method_name, alpha, instructions):
            outputs = generate_with_instructions(
                model, tokenizer, questions, instructions, max_new_tokens
            )
            for pt, q, instr, out in zip(prompt_traits, questions, instructions, outputs):
                rows.append({
                    "group": group_id,
                    "trait1": trait1,
                    "trait2": trait2,
                    "method": method_name,
                    "alpha": alpha,
                    "prompt_trait": pt,
                    "question": q,
                    "generated_output": out,
                })

        # Methods 1, 2, 4
        for method_id, method_name in [
            (1, "orthogonalize"),
            (2, "different_layers"),
            (4, "tuned_mean"),
        ]:
            instructions = build_instructions(
                trait1, trait2, model, method_id, layer1, layer2, coeff1, coeff2, available_vectors, is_llama=is_llama
            )
            _add_rows(method_name, None, instructions)

        # Method 5: unit-norm mean/sum at layer 20, alpha sweep
        alphas = ALPHAS_llama if is_llama else ALPHAS_qwen
        for alpha_layer, alpha in alphas.items():
            instructions = build_instructions(
                trait1, trait2, model, 5, alpha_layer, alpha_layer, None, None, available_vectors, alpha=alpha, is_llama=is_llama
            )
            _add_rows("unit_norm_mean", alpha, instructions)

    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"    Saved {len(rows)} rows → {output_path}")


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
        "--output-dir", default="experiments/experiment_3",
        help="Directory for output CSVs (default: experiments/experiment_3)",
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
    parser.add_argument("--max-new-tokens", type=int, default=500)
    parser.add_argument(
        "--num-threads", type=int, default=4,
        help="PyTorch CPU thread limit (default: 4). Lower this when sharing a machine.",
    )
    args = parser.parse_args()

    torch.set_num_threads(args.num_threads)
    print(f"PyTorch CPU threads: {args.num_threads}")

    best_json_path = Path(args.best_json)
    if not best_json_path.exists():
        print(f"Error: --best-json not found: {best_json_path}")
        return 1

    with open(best_json_path) as f:
        best_settings: dict = json.load(f)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    demo_dir = Path(args.demo_dir)
    trait_data_dir = Path(args.trait_data_dir)

    print("Scanning PROSE demo files for unique trait groups...")
    groups = collect_unique_groups(demo_dir)
    print(f"Found {len(groups)} unique groups.\n")

    # Write manifest so group_ids can be decoded later
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
    print("Model ready.\n")

    available_vectors = pipeline.service.available_vectors

    for i, (group_id, traits) in enumerate(groups.items(), 1):
        output_path = output_dir / f"experiment_3_{group_id}.csv"
        if output_path.exists():
            print(f"[{i}/{len(groups)}] Skip {group_id} (exists)")
            continue

        print(f"[{i}/{len(groups)}] Group: {traits}")
        run_group(
            group_id=group_id,
            traits=traits,
            best_settings=best_settings,
            model=model,
            tokenizer=tokenizer,
            available_vectors=available_vectors,
            trait_data_dir=trait_data_dir,
            output_path=output_path,
            max_new_tokens=args.max_new_tokens,
            is_llama=('llama' in args.model)
        )

    print(f"\nDone. Results in {output_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
