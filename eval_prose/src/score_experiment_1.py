#!/usr/bin/env python3
"""
Score experiment_1 outputs with the prose LLM judge against the single steered trait.

Unlike evaluate_prose.py, which scores generated outputs against the user's full
true-preference set, this script builds a one-preference WritingPreferenceSet from
the steered trait alone.  The resulting PPCM score answers: "does the prose output
exhibit the trait it was steered for?" — directly comparable to experiment_2 trait
scores, but using the prose judge on prose tasks rather than the persona judge on
persona-specific questions.

Input: experiment_1_*.csv files with columns:
    source, turn, task_text, task_prompt, true_preferences, user_demo,
    trait, layer, coeff, generated_output

Output: *_scored.csv alongside each input (or in --output-dir) with two new columns:
    ppcm            – per-preference match score (0–1) for the steered trait
    component_scores – per-preference breakdown dict

Task type (email_writing / summarization) is inferred from the filename stem.
Already-scored files are skipped unless --force is passed.

Usage (from eval_prose/src/):
    python score_experiment_1.py experiments/experiment_1/experiment_1_emoji_usage_email_writing_1352.csv
    python score_experiment_1.py experiments/experiment_1/*.csv
    python score_experiment_1.py experiments/experiment_1/*.csv --output-dir experiments/experiment_1_scored/
"""

import argparse
import asyncio
import glob
import os
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import numpy as np
from tqdm import tqdm

# ── ml-predict path setup (mirrors evaluate_prose.py) ────────────────────────
_ml_predict_path = Path(__file__).resolve().parents[1] / "ml-predict"
if str(_ml_predict_path) not in sys.path:
    sys.path.insert(0, str(_ml_predict_path))

try:
    from preference_inferrer.common.personal_keys import OPENAI_API_KEY
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
except Exception:
    pass

import preference_inferrer.tasks.writing.writing_quality_assesor as wq_module
from preference_inferrer.tasks.task_instances import WritingTaskInstance
from preference_inferrer.preference_sets.writing_preference_set import WritingPreferenceSet
from preference_inferrer.common.configurations import configure_config

# ── Trait → preference string map (mirrors evaluate_prose.py) ────────────────

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

TRAIT_TO_PREFERENCE = {trait: pref for pref, trait in PREFERENCE_TO_TRAIT.items()}


def _trait_to_pref_set(trait: str, framework: str) -> WritingPreferenceSet:
    pref_str = TRAIT_TO_PREFERENCE.get(trait, trait)
    from types import SimpleNamespace
    config = SimpleNamespace(task=SimpleNamespace(framework=framework))
    return WritingPreferenceSet([pref_str], config)


def _task_type_from_path(path: Path) -> str:
    stem = path.stem.lower()
    if "email" in stem:
        return "email_writing"
    if "summ" in stem or "summarization" in stem:
        return "summarization"
    return "email_writing"


def _make_evaluator(framework: str) -> wq_module.WritingQualityAssessor:
    config_dict = {
        "bertscore_model": "microsoft/deberta-xlarge-mnli",
        "logging_level": "error",
        "base_dir": None,
        "user": {"llm_name": "gpt-4o"},
        "gpt_temperature": 0.01,
        "seed": 1352,
        "task": {"framework": framework},
    }
    return wq_module.WritingQualityAssessor(configure_config(config_dict))


# ── Async evaluation (mirrors evaluate_prose.py) ─────────────────────────────

async def _eval_one(evaluator, task_instance, pref_set):
    ppcm, ppcm_dict = await evaluator.llm_judges_preference_matching(
        task_instance, pref_set, per_preference=True
    )
    return ppcm, ppcm_dict


async def _eval_batched(eval_tasks, max_concurrent=100):
    semaphore = asyncio.Semaphore(max_concurrent)

    async def run(task_idx, evaluator, task_instance, pref_set):
        async with semaphore:
            result = await _eval_one(evaluator, task_instance, pref_set)
            return task_idx, result

    tasks = [run(i, ev, ti, ps) for i, (ev, ti, ps) in enumerate(eval_tasks)]
    results = [None] * len(tasks)

    with tqdm(total=len(tasks), desc="  judge evaluations", ncols=70) as pbar:
        for coro in asyncio.as_completed(tasks):
            idx, result = await coro
            results[idx] = result
            pbar.update(1)

    return results


# ── Per-file scoring ──────────────────────────────────────────────────────────

def score_csv(
    input_path: Path,
    output_path: Path,
    framework: str,
    force: bool,
) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        print(f"  Skipping (unreadable): {e}")
        return None
    if df.empty or len(df.columns) == 0:
        print(f"  Skipping (no data)")
        return None

    required = {"trait", "task_text", "generated_output"}
    missing = required - set(df.columns)
    if missing:
        print(f"  Skipping: missing columns {missing}")
        return None

    if output_path.exists() and not force:
        existing = pd.read_csv(output_path)
        if "ppcm" in existing.columns and len(existing) == len(df):
            print(f"  Already scored ({len(existing)} rows). Use --force to re-run.")
            return existing

    task_type = _task_type_from_path(input_path)
    evaluator = _make_evaluator(framework)

    eval_tasks = []
    rows_out = []

    for _, row in df.iterrows():
        trait = row["trait"]
        task_text = row.get("task_text", row.get("task_prompt", ""))
        generated_output = row.get("generated_output", "")
        source = row.get("source", row.get("task_source", "unknown"))

        pref_set = _trait_to_pref_set(trait, framework)

        task_instance = WritingTaskInstance(
            task_type=task_type,
            task_content=task_text,
            source=source,
            source_idx=0,
        )
        task_instance.agent_completion = generated_output
        task_instance.user_completion = row.get("user_demo", "")

        eval_tasks.append((evaluator, task_instance, pref_set))
        rows_out.append(dict(row))

    print(f"  {len(eval_tasks)} rows | task_type={task_type}")
    llm_results = asyncio.run(_eval_batched(eval_tasks))

    for i, (ppcm, ppcm_dict) in enumerate(llm_results):
        rows_out[i]["ppcm"] = ppcm
        rows_out[i]["component_scores"] = ppcm_dict

    result_df = pd.DataFrame(rows_out)
    result_df.to_csv(output_path, index=False)

    mean_ppcm = result_df["ppcm"].mean()
    component_means = defaultdict(list)
    for cs in result_df["component_scores"]:
        if isinstance(cs, dict):
            for k, v in cs.items():
                component_means[k].append(v)
    print(f"  PPCM (steered trait only): {mean_ppcm:.3f}")
    for k, vals in component_means.items():
        print(f"    {k}: {np.mean(vals):.3f}")
    print(f"  Saved → {output_path}")
    return result_df


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "input_csvs", nargs="+",
        help="Experiment-1 CSV file(s). Glob patterns are supported.",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Directory for scored CSVs (default: same directory as each input file)",
    )
    parser.add_argument(
        "--framework", default="plume",
        help="Framework name passed to WritingPreferenceSet (default: plume)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-score files that already have a scored output",
    )
    args = parser.parse_args()

    input_files: list[Path] = []
    for pattern in args.input_csvs:
        expanded = glob.glob(pattern)
        if expanded:
            input_files.extend(Path(p) for p in expanded)
        else:
            input_files.append(Path(pattern))
    input_files = [p for p in input_files if p.exists() and p.suffix == ".csv" and p.stat().st_size > 0]

    if not input_files:
        print("Error: no input files found")
        return 1

    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    print(f"Scoring {len(input_files)} file(s)")
    print("=" * 60)

    for input_path in sorted(input_files):
        if output_dir:
            output_path = output_dir / f"{input_path.stem}_scored.csv"
        else:
            output_path = input_path.with_name(f"{input_path.stem}_scored.csv")

        print(f"\n{input_path.name}")
        try:
            result_df = score_csv(
                input_path=input_path,
                output_path=output_path,
                framework=args.framework,
                force=args.force,
            )
        except Exception as e:
            print(f"  Error: {e}")
            continue
        if result_df is not None and "ppcm" in result_df.columns:
            summaries.append({
                "file": input_path.name,
                "rows": len(result_df),
                "trait": result_df["trait"].iloc[0] if "trait" in result_df.columns else "?",
                "task_type": _task_type_from_path(input_path),
                "mean_ppcm": result_df["ppcm"].mean(),
            })

    if len(summaries) > 1:
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"\n{'File':<55} {'Task':>14} {'PPCM':>7} {'N':>5}")
        print("-" * 83)
        for s in summaries:
            print(f"{s['file']:<55} {s['task_type']:>14} {s['mean_ppcm']:>7.3f} {s['rows']:>5}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
