#!/usr/bin/env python3
"""
Score experiment_5 outputs with the prose PPCM judge and coherence judge.

Input: experiment_5_*.csv files (output of experiment_5.py) with columns:
    group, traits, subset_size, method, alpha,
    task, seed, source, turn, task_text, task_prompt,
    true_preferences, generated_output

Three new columns are added per row:
    ppcm            – prose per-preference match score (0–1) vs the subset traits
    component_scores – per-preference breakdown (JSON string)
    coherence_score – 0–100 prose-task coherence score (GPT-4o logprob)

Already-scored files are skipped unless --force is passed.

For plots, pass the scored output directory to visualize_experiments.py --exp5-dir.

Usage (from eval_prose/src/):
    python score_experiment_5.py experiments/experiment_5/*.csv
    python score_experiment_5.py experiments/experiment_5/*.csv \\
        --output-dir experiments/experiment_5_scored/
"""

import argparse
import asyncio
import glob
import json
import math
import os
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from openai import AsyncOpenAI

# ── ml-predict path setup ─────────────────────────────────────────────────────
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

# ── Trait ↔ preference maps ───────────────────────────────────────────────────

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

# ── Coherence judge ───────────────────────────────────────────────────────────

_COHERENCE_PROMPT = (
    "I am evaluating a language model on a prose writing task. I want to assess the coherence"
    " of its response — please focus solely on coherence, independent of stylistic traits.\n\n"
    "The writing task was:\n[TASK START]\n{question}\n[TASK END]\n\n"
    "The model responded with:\n[ANSWER START]\n{answer}\n[ANSWER END]\n\n"
    "Rate the response from 0 to 100 on how coherent it is for the given task.\n"
    "* 0: makes no sense; not valid English.\n"
    "* 50: somewhat coherent but confused or rambling.\n"
    "* 100: clearly understood the task; no hallucinations or signs of confusion.\n\n"
    "Judge only coherence, not stylistic quality or task completion.\n"
    "Respond with a number between 0 and 100. Don't say anything else."
)


class _CoherenceJudge:
    def __init__(self, client: AsyncOpenAI, model: str):
        self.client = client
        self.model = model

    async def judge(self, question: str, answer: str) -> float | None:
        messages = [{"role": "user", "content": _COHERENCE_PROMPT.format(question=question, answer=answer)}]
        completion = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=1,
            temperature=0,
            logprobs=True,
            top_logprobs=20,
            seed=0,
        )
        try:
            top_logprobs = completion.choices[0].logprobs.content[0].top_logprobs
        except (IndexError, AttributeError):
            return None
        total, sum_ = 0.0, 0.0
        for el in top_logprobs:
            try:
                v = int(el.token)
            except ValueError:
                continue
            if 0 <= v <= 100:
                p = math.exp(el.logprob)
                sum_ += v * p
                total += p
        return sum_ / total if total >= 0.25 else None


# ── Prose evaluator ───────────────────────────────────────────────────────────

def _make_prose_evaluator(framework: str) -> wq_module.WritingQualityAssessor:
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


def _traits_str_to_pref_set(traits_str: str, framework: str) -> WritingPreferenceSet | None:
    from types import SimpleNamespace
    traits = [t.strip() for t in str(traits_str).split(";") if t.strip()]
    pref_strs = [TRAIT_TO_PREFERENCE.get(t) for t in traits]
    pref_strs = [p for p in pref_strs if p is not None]
    if not pref_strs:
        return None
    config = SimpleNamespace(task=SimpleNamespace(framework=framework))
    return WritingPreferenceSet(pref_strs, config)


# ── Per-file scoring ──────────────────────────────────────────────────────────

async def score_csv_async(
    input_path: Path,
    output_path: Path,
    judge_model: str,
    client: AsyncOpenAI,
    framework: str,
    concurrency: int,
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

    required = {"traits", "task", "task_text", "task_prompt", "generated_output"}
    missing = required - set(df.columns)
    if missing:
        print(f"  Skipping: missing columns {missing}")
        return None

    if output_path.exists() and not force:
        existing = pd.read_csv(output_path)
        if (
            "ppcm" in existing.columns
            and "coherence_score" in existing.columns
            and len(existing) == len(df)
        ):
            print(f"  Already scored ({len(existing)} rows). Use --force to re-run.")
            return existing

    evaluator = _make_prose_evaluator(framework)
    coherence_judge = _CoherenceJudge(client, judge_model)
    semaphore = asyncio.Semaphore(concurrency)

    # Build per-row inputs
    prose_tasks = []   # (evaluator, WritingTaskInstance, WritingPreferenceSet) | None
    coh_inputs = []    # (task_text, generated_output)

    for _, row in df.iterrows():
        pref_set = _traits_str_to_pref_set(str(row["traits"]), framework)
        if pref_set is not None:
            ti = WritingTaskInstance(
                task_type=str(row["task"]),
                task_content=str(row["task_text"]),
                source=str(row.get("source", "unknown")),
                source_idx=0,
            )
            ti.agent_completion = str(row["generated_output"])
            ti.user_completion = ""
            prose_tasks.append((evaluator, ti, pref_set))
        else:
            prose_tasks.append(None)
        coh_inputs.append((str(row["task_text"]), str(row["generated_output"])))

    print(f"  {len(df)} rows")

    # ── Prose judge ────────────────────────────────────────────────────────────
    async def run_prose(idx, task):
        if task is None:
            return idx, (float("nan"), {})
        ev, ti, ps = task
        async with semaphore:
            try:
                ppcm, ppcm_dict = await ev.llm_judges_preference_matching(ti, ps, per_preference=True)
                return idx, (ppcm, ppcm_dict)
            except Exception:
                return idx, (float("nan"), {})

    prose_results = [None] * len(df)
    with tqdm(total=len(df), desc="  prose judge", ncols=70) as pbar:
        for coro in asyncio.as_completed([run_prose(i, t) for i, t in enumerate(prose_tasks)]):
            i, result = await coro
            prose_results[i] = result
            pbar.update(1)

    # ── Coherence judge ────────────────────────────────────────────────────────
    async def run_coh(idx, question, answer):
        async with semaphore:
            try:
                score = await coherence_judge.judge(question=question, answer=answer)
                return idx, score
            except Exception:
                return idx, None

    coh_results = [None] * len(df)
    with tqdm(total=len(df), desc="  coherence", ncols=70) as pbar:
        for coro in asyncio.as_completed([run_coh(i, q, a) for i, (q, a) in enumerate(coh_inputs)]):
            i, result = await coro
            coh_results[i] = result
            pbar.update(1)

    # Assemble
    result_df = df.copy()
    result_df["ppcm"] = [r[0] for r in prose_results]
    result_df["component_scores"] = [json.dumps(r[1]) for r in prose_results]
    result_df["coherence_score"] = [
        (s if s is not None else float("nan")) for s in coh_results
    ]

    result_df.to_csv(output_path, index=False)

    n = len(result_df)
    n_ppcm = result_df["ppcm"].notna().sum()
    n_coh = result_df["coherence_score"].notna().sum()
    print(f"  ppcm:            {result_df['ppcm'].mean():.3f} mean  ({n_ppcm}/{n} scored)")
    print(f"  coherence_score: {result_df['coherence_score'].mean():.1f} mean  ({n_coh}/{n} scored)")
    print(f"  Saved → {output_path}")
    return result_df


# ── Main ──────────────────────────────────────────────────────────────────────

async def main_async(args) -> int:
    input_files: list[Path] = []
    for pattern in args.input_csvs:
        expanded = glob.glob(pattern)
        if expanded:
            input_files.extend(Path(p) for p in expanded)
        else:
            input_files.append(Path(pattern))
    input_files = [
        p for p in input_files
        if p.exists() and p.suffix == ".csv" and p.stat().st_size > 0
    ]
    if not input_files:
        print("Error: no non-empty CSV files found")
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    client = AsyncOpenAI()
    print(f"Scoring {len(input_files)} file(s) with judge model '{args.judge_model}'")
    print("=" * 60)

    for input_path in sorted(input_files):
        output_path = output_dir / f"{input_path.stem}_scored.csv"
        print(f"\n{input_path.name}")
        try:
            await score_csv_async(
                input_path=input_path,
                output_path=output_path,
                judge_model=args.judge_model,
                client=client,
                framework=args.framework,
                concurrency=args.concurrency,
                force=args.force,
            )
        except Exception as e:
            print(f"  Error: {e}")

    print("\nDone.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input_csvs", nargs="+",
        help=(
            "Experiment-5 CSV file(s). Glob patterns supported. "
            "For --plot-only, pass *_scored.csv files."
        ),
    )
    parser.add_argument(
        "--output-dir", default="experiments/experiment_5_scored",
        help="Directory for scored CSVs and plots (default: experiments/experiment_5_scored/)",
    )
    parser.add_argument(
        "--judge-model", default="gpt-4o",
        help="OpenAI model for judging (default: gpt-4o)",
    )
    parser.add_argument(
        "--framework", default="plume",
        help="Framework name for WritingPreferenceSet (default: plume)",
    )
    parser.add_argument(
        "--concurrency", type=int, default=50,
        help="Max simultaneous API requests (default: 50)",
    )
    parser.add_argument("--force", action="store_true", help="Re-score already-scored files")
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
