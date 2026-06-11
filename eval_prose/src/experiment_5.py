#!/usr/bin/env python3
"""
Experiment 5: Multi-trait combination on prose tasks, across subset sizes.

For each unique preference group found in the PROSE demo files, runs all
subsets of size 1, 2, 3, and up to the full group size with four combination
methods drawn directly from experiment_3:

  orthogonalize   – Sequential Gram-Schmidt across all traits in the subset;
                    each orthogonalised vector is scaled by its tuned coeff and
                    applied at its tuned layer.  Same-layer vectors are summed.
  different_layers – Each vector scaled by its tuned coeff applied at its tuned
                     layer.  When two traits share a tuned layer, the lower-
                     priority trait is shifted to (tuned_layer − rank) and the
                     vector is loaded at that layer, mirroring experiment_3's
                     layer2 -= 1 approach.
  tuned_mean      – Each vector scaled by its tuned coeff applied at its tuned
                     layer.  Same-layer vectors are averaged.
  unit_norm_mean  – All vectors unit-normalised and loaded at a common alpha_layer
                     (either 16 or 20), averaged, then scaled by the layer-specific
                     alpha from ALPHAS_llama / ALPHAS_qwen.  One generation per
                     (alpha_layer, alpha) pair.

Unlike experiments 3 and 4, prompts come from the PROSE demo files rather than
persona question pools.  For each (group, subset, method), only rows where every
trait in the subset appears in the row's true_preferences are used as prompts,
mirroring experiment_1's per-trait filtering logic.

The single-trait case (subset_size=1) is included so performance can be plotted
as a function of the number of combined traits.  For methods orthogonalize /
different_layers / tuned_mean, a single trait is identical: coeff*vec at its
tuned layer; a single generation is shared across all three method names.

Output files: {output_dir}/experiment_5_{group_id}.csv
Already-completed files are skipped, so the script is safe to re-run.
A manifest JSON is written alongside the CSVs mapping group_id → traits.

Output CSV columns:
    group, traits, subset_size, method, alpha, alpha_layer,
    task, seed, source, turn, task_text, task_prompt,
    true_preferences, generated_output

Usage (from eval_prose/src/):
    python experiment_5.py
    python experiment_5.py --best-json tune_results/best_settings.json
    python experiment_5.py --num-sources 2 --batch-size 4   # quick smoke-test
    python experiment_5.py --model meta-llama/Llama-3.1-8B-Instruct  # auto-detects llama alphas
"""

import argparse
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch

from activation_steer import ActivationSteererMultiple
from predefined_steering import PredefinedSteeringPipeline
from prose.task_prompts import get_task_prompt_builder

# ── Constants ─────────────────────────────────────────────────────────────────

TASKS = ["email_writing", "summarization"]
SEEDS = [1352, 4792, 5961, 6584, 8337]

# Per-layer alphas for unit_norm_mean, mirroring experiment_3
# ALPHAS_LLAMA = {16: 4.0,  20: 8.0}
# ALPHAS_QWEN  = {16: 30.0, 20: 58.0}
ALPHAS_LLAMA = {16: [2.0, 4.0, 6.0, 8.0],  20: [6.0, 8.0, 10.0, 12.0]}
#ALPHAS_QWEN  = {16: [30.0, 40.0, 50.0, 60.0], 20: [60.0, 70.0, 80.0, 90.0]}
ALPHAS_QWEN  = {20: [60.0, 80.0, 100.0, 120.0]}

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
TRAIT_TO_PREFERENCE = {v: k for k, v in PREFERENCE_TO_TRAIT.items()}

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
                groups["__".join(sorted_traits)] = sorted_traits
                seen.add(key)
    return groups


def load_prose_rows(demo_dir: Path, num_sources: int | None = None) -> list[dict]:
    """Load all rows from PROSE demo files across all tasks and seeds."""
    rows = []
    for task in TASKS:
        task_prompt_builder = get_task_prompt_builder(task_name=task)
        for seed in SEEDS:
            f = demo_dir / f"plume-{task}-prose.full-qwen-{seed}-inferring_results.csv"
            if not f.exists():
                continue
            df = pd.read_csv(f)
            sources = {
                src: grp.sort_values("source_example_num")
                for src, grp in df.groupby("task_source")
            }
            if num_sources:
                sources = dict(list(sources.items())[:num_sources])
            for source_name, source_df in sources.items():
                for _, row in source_df.iterrows():
                    rows.append({
                        "task": task,
                        "seed": seed,
                        "source": source_name,
                        "turn": int(row["source_example_num"]),
                        "task_text": row["context"],
                        "task_prompt": task_prompt_builder(row["context"]),
                        "true_preferences": row["true preferences"],
                    })
    return rows


def _filter_rows_for_subset(rows: list[dict], traits: list[str]) -> list[dict]:
    """Return rows where all traits in the subset appear in true_preferences."""
    pref_strs = [TRAIT_TO_PREFERENCE.get(t) for t in traits]
    if any(p is None for p in pref_strs):
        return []
    needed = set(pref_strs)
    out = []
    for row in rows:
        raw = row.get("true_preferences", "")
        if pd.isna(raw):
            continue
        if needed.issubset({p.strip() for p in str(raw).split(";")}):
            out.append(row)
    return out


# ── Vector helpers ────────────────────────────────────────────────────────────

def load_vec(trait: str, layer: int, available_vectors: dict) -> torch.Tensor | None:
    if trait not in available_vectors:
        return None
    full = torch.load(available_vectors[trait], map_location="cpu", weights_only=True)
    return full[layer].float()


def gram_schmidt_orth_n(vecs: list[torch.Tensor]) -> list[torch.Tensor]:
    """Sequential Gram-Schmidt: each vector has all previous vectors' components removed.

    The first vector is returned unchanged.  Subsequent vectors are projected
    orthogonal to all preceding (already-orthogonalised) vectors but are NOT
    re-normalised, matching experiment_3's gram_schmidt_orth behaviour.
    """
    result = []
    for v in vecs:
        v_orth = v.clone()
        for prev in result:
            prev_unit = prev / prev.norm()
            v_orth = v_orth - (v_orth @ prev_unit) * prev_unit
        result.append(v_orth)
    return result


def _consolidate(instructions: list[dict], use_sum: bool = False) -> list[dict]:
    """Merge instructions sharing the same layer_idx into one vector per layer."""
    buckets: dict[int, list[torch.Tensor]] = defaultdict(list)
    for inst in instructions:
        buckets[inst["layer_idx"]].append(inst["steering_vector"])
    return [
        {
            "steering_vector": torch.stack(vecs).sum(dim=0) if use_sum
                               else torch.stack(vecs).mean(dim=0),
            "layer_idx": layer_idx,
        }
        for layer_idx, vecs in sorted(buckets.items())
    ]


def _assign_diff_layers(traits: list[str], tuned_layers: list[int]) -> list[int]:
    """Assign unique layers for different_layers method.

    Mirrors experiment_3's `layer2 -= 1` approach: within each group of traits
    that share a tuned layer, the first keeps the tuned layer and each subsequent
    trait is decremented by one additional step.
    """
    layer_groups: dict[int, list[int]] = defaultdict(list)
    for idx, layer in enumerate(tuned_layers):
        layer_groups[layer].append(idx)

    assigned = list(tuned_layers)
    for base_layer, indices in layer_groups.items():
        for rank, idx in enumerate(indices[1:], 1):
            assigned[idx] = base_layer - rank
    return assigned


# ── Instruction builders ──────────────────────────────────────────────────────

def build_instructions_single(
    trait: str,
    layer: int,
    coeff: float,
    model,
    available_vectors: dict,
) -> list[dict] | None:
    """Single-trait instruction: just coeff * vec at tuned layer."""
    v = load_vec(trait, layer, available_vectors)
    if v is None:
        return None
    dtype = next(model.parameters()).dtype
    v = v.to(dtype=dtype, device=model.device)
    return [{"steering_vector": coeff * v, "layer_idx": layer - 1}]


def build_instructions_n(
    method_id: int,
    traits: list[str],
    layers: list[int],
    coeffs: list[float],
    model,
    available_vectors: dict,
    alpha: float | None = None,
    alpha_layer: int | None = None,
) -> list[dict] | None:
    """Build instructions for a subset of N>=2 traits.

    method_id:
      1 = orthogonalize
      2 = different_layers
      4 = tuned_mean
      5 = unit_norm_mean  (requires alpha and alpha_layer)
    """
    dtype = next(model.parameters()).dtype

    if method_id == 5:
        vecs = []
        for t in traits:
            v = load_vec(t, alpha_layer, available_vectors)
            if v is None:
                return None
            vecs.append(v.to(dtype=dtype, device=model.device))
        return _consolidate([
            {"steering_vector": alpha * v / v.norm(), "layer_idx": alpha_layer - 1}
            for v in vecs
        ])

    # Methods 1, 2, 4: load each vector at its tuned layer
    vecs = []
    for t, layer in zip(traits, layers):
        v = load_vec(t, layer, available_vectors)
        if v is None:
            return None
        vecs.append(v.to(dtype=dtype, device=model.device))

    if method_id == 1:
        orth_vecs = gram_schmidt_orth_n(vecs)
        return _consolidate([
            {"steering_vector": c * ov, "layer_idx": l - 1}
            for c, ov, l in zip(coeffs, orth_vecs, layers)
        ], use_sum=True)

    elif method_id == 2:
        assigned = _assign_diff_layers(traits, layers)
        instructions = []
        for t, c, al in zip(traits, coeffs, assigned):
            v = load_vec(t, al, available_vectors)
            if v is None:
                return None
            v = v.to(dtype=dtype, device=model.device)
            instructions.append({"steering_vector": c * v, "layer_idx": al - 1})
        return instructions

    elif method_id == 4:
        return _consolidate([
            {"steering_vector": c * v, "layer_idx": l - 1}
            for c, v, l in zip(coeffs, vecs, layers)
        ], use_sum=False)

    raise ValueError(f"Unknown method_id: {method_id}")


# ── Generation ────────────────────────────────────────────────────────────────

def generate_batched(
    model,
    tokenizer,
    prompts: list[str],
    instructions: list[dict],
    max_new_tokens: int,
    batch_size: int,
) -> list[str]:
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
        for out in outputs:
            results.append(tokenizer.decode(out[prompt_len:], skip_special_tokens=True))
    return results


# ── Per-group runner ──────────────────────────────────────────────────────────

def run_group(
    group_id: str,
    traits: list[str],
    best_settings: dict,
    all_prose_rows: list[dict],
    model,
    tokenizer,
    available_vectors: dict,
    output_path: Path,
    alphas: dict[int, float],
    max_new_tokens: int,
    batch_size: int,
) -> None:
    rows_out = []
    n_group = len(traits)

    for subset_size in range(1, n_group + 1):
        for subset in itertools.combinations(traits, subset_size):
            subset_list = list(subset)
            traits_str = ";".join(subset_list)

            # Check all traits have settings and vectors
            missing = [t for t in subset_list if t not in best_settings]
            if missing:
                print(f"    Skip {subset_list}: missing best_settings for {missing}")
                continue

            subset_layers = [best_settings[t]["layer"] for t in subset_list]
            subset_coeffs = [best_settings[t]["coeff"] for t in subset_list]

            matching_rows = _filter_rows_for_subset(all_prose_rows, subset_list)
            if not matching_rows:
                print(f"    Skip {subset_list}: no matching prose rows")
                continue

            prompts = [r["task_prompt"] for r in matching_rows]
            print(f"    {subset_list}  ({len(matching_rows)} rows)")

            def _record(generated, method_name, alpha_val, alpha_layer_val=None):
                for row, gen in zip(matching_rows, generated):
                    rows_out.append({
                        "group": group_id,
                        "traits": traits_str,
                        "subset_size": subset_size,
                        "method": method_name,
                        "alpha": alpha_val,
                        "alpha_layer": alpha_layer_val,
                        "task": row["task"],
                        "seed": row["seed"],
                        "source": row["source"],
                        "turn": row["turn"],
                        "task_text": row["task_text"],
                        "task_prompt": row["task_prompt"],
                        "true_preferences": row["true_preferences"],
                        "generated_output": gen,
                    })

            if subset_size == 1:
                # Single trait: orthogonalize / different_layers / tuned_mean are identical
                # instr = build_instructions_single(
                #     subset_list[0], subset_layers[0], subset_coeffs[0],
                #     model, available_vectors,
                # )
                # if instr is None:
                #     print(f"    Skip {subset_list}: missing vector")
                #     continue
                # generated = generate_batched(model, tokenizer, prompts, instr, max_new_tokens, batch_size)
                # for method_name in ["orthogonalize", "different_layers", "tuned_mean"]:
                #     _record(generated, method_name, None)

                # unit_norm_mean: one generation per (alpha_layer, alpha)
                for alpha_layer, alpha_list in alphas.items():
                    alpha = alpha_list[0]
                    instr5 = build_instructions_n(
                        5, subset_list, subset_layers, subset_coeffs,
                        model, available_vectors,
                        alpha=alpha, alpha_layer=alpha_layer,
                    )
                    if instr5 is None:
                        continue
                    generated5 = generate_batched(model, tokenizer, prompts, instr5, max_new_tokens, batch_size)
                    _record(generated5, "unit_norm_mean", alpha, alpha_layer)

            else:
                # # N >= 2: run all four methods
                # for method_id, method_name in [
                #     (1, "orthogonalize"),
                #     (2, "different_layers"),
                #     (4, "tuned_mean"),
                # ]:
                #     instr = build_instructions_n(
                #         method_id, subset_list, subset_layers, subset_coeffs,
                #         model, available_vectors,
                #     )
                #     if instr is None:
                #         print(f"    Skip {subset_list} / {method_name}: missing vector")
                #         continue
                #     generated = generate_batched(model, tokenizer, prompts, instr, max_new_tokens, batch_size)
                #     _record(generated, method_name, None)

                for alpha_layer, alpha_list in alphas.items():
                    alpha = alpha_list[subset_size-1]
                    instr5 = build_instructions_n(
                        5, subset_list, subset_layers, subset_coeffs,
                        model, available_vectors,
                        alpha=alpha, alpha_layer=alpha_layer,
                    )
                    if instr5 is None:
                        print(f"    Skip {subset_list} / unit_norm_mean alpha={alpha}: missing vector")
                        continue
                    generated5 = generate_batched(model, tokenizer, prompts, instr5, max_new_tokens, batch_size)
                    _record(generated5, "unit_norm_mean", alpha, alpha_layer)

    pd.DataFrame(rows_out).to_csv(output_path, index=False)
    print(f"    Saved {len(rows_out)} rows → {output_path}")


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
        "--output-dir", default="experiments/experiment_5",
        help="Directory for output CSVs (default: experiments/experiment_5)",
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--vector-dir", default="final_vectors")
    parser.add_argument(
        "--demo-dir", default="../demo_files",
        help="Directory containing plume-*-inferring_results.csv files",
    )
    parser.add_argument("--max-new-tokens", type=int, default=500)
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="Number of prompts per model.generate() call (default: 8)",
    )
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

    best_json_path = Path(args.best_json)
    if not best_json_path.exists():
        print(f"Error: --best-json not found: {best_json_path}")
        return 1
    with open(best_json_path) as f:
        best_settings: dict = json.load(f)

    is_llama = "llama" in args.model.lower()
    alphas = ALPHAS_LLAMA if is_llama else ALPHAS_QWEN
    print(f"Model family: {'llama' if is_llama else 'qwen'}  unit_norm_mean alphas: {alphas}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    demo_dir = Path(args.demo_dir)

    print("Scanning PROSE demo files for unique trait groups...")
    groups = collect_unique_groups(demo_dir)
    print(f"Found {len(groups)} unique groups.\n")

    manifest_path = output_dir / "groups_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(groups, f, indent=2)
    print(f"Manifest written to {manifest_path}\n")

    print("Loading all PROSE rows...")
    all_prose_rows = load_prose_rows(demo_dir, num_sources=args.num_sources)
    print(f"Loaded {len(all_prose_rows)} prose rows total.\n")

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
        output_path = output_dir / f"experiment_5_{group_id}.csv"
        if output_path.exists():
            print(f"[{i}/{len(groups)}] Skip {group_id} (exists)")
            continue

        n_subsets = sum(
            len(list(itertools.combinations(traits, k)))
            for k in range(1, len(traits) + 1)
        )
        n_methods = 3 + len(alphas)  # orthogonalize/different_layers/tuned_mean + unit_norm_mean per alpha
        print(f"[{i}/{len(groups)}] Group: {traits}  ({n_subsets} subsets × up to {n_methods} methods)")
        run_group(
            group_id=group_id,
            traits=traits,
            best_settings=best_settings,
            all_prose_rows=all_prose_rows,
            model=model,
            tokenizer=tokenizer,
            available_vectors=available_vectors,
            output_path=output_path,
            alphas=alphas,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
        )

    print(f"\nDone. Results in {output_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
