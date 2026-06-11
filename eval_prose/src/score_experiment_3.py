#!/usr/bin/env python3
"""
Score experiment_3 outputs with the persona trait judges and coherence judge.

Reads experiment_3_*.csv files (columns: group, trait1, trait2, method, alpha,
prompt_trait, question, generated_output) and adds three score columns per row:
  - trait1_score:    0-100 logprob score for trait1's eval prompt
  - trait2_score:    0-100 logprob score for trait2's eval prompt
  - coherence_score: 0-100 logprob score for response coherence

Each trait is scored independently using its own judge prompt (same method as
tune_steering.py). Rows where a judge returns None (low probability mass) are
saved as NaN.

Output files mirror input names with a '_scored' suffix, saved to --output-dir
(default: same directory as input). Already-scored files are skipped unless
--force is passed.

Usage (from eval_prose/src/):
    python score_experiment_3.py experiments/experiment_3/experiment_3_group0.csv
    python score_experiment_3.py experiments/experiment_3/*.csv
    python score_experiment_3.py experiments/experiment_3/*.csv --output-dir experiments/experiment_3_scored/
    python score_experiment_3.py experiments/experiment_3/*.csv --concurrency 50 --judge-model gpt-4o
"""

import argparse
import asyncio
import glob
import json
import math
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from openai import AsyncOpenAI


# ── Trait → data filename stem (mirrors tune_steering.py / experiment_3.py) ──

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

_TRAIT_DATA_DIR = (
    Path(__file__).resolve().parents[2] / "persona_vectors" / "data_generation" / "trait_data_eval"
)

_COHERENCE_PROMPT = (
    "I am evaluating a different language model. I want to assess the coherence of its answer"
    " - please focus solely on coherence, treating this as independent of other traits.\n\n"
    "I asked the model the following evaluation question:\n[QUESTION START]\n{question}\n[QUESTION END]\n\n"
    "The model answered with:\n[ANSWER START]\n{answer}\n[ANSWER END]\n\n"
    "You should rate the answer on a scale of 0 to 100, focusing on how coherent the model's answer"
    " is in the context of the question.\n"
    "* 0 means \"makes absolutely no sense; the model generated text that is not even valid English.\"\n"
    "* 50 means \"the answer kind of makes sense but seems a bit confused; the model is rambling.\"\n"
    "* 100 means \"the answer shows that the model clearly understood the question and contains no"
    " hallucinations or other signs of confusion.\"\n\n"
    "Note that you shouldn't judge whether the answer is correct or logical - an answer that is"
    " completely incorrect could still score 100 if it is coherent"
    " (e.g., something a mean person might say).\n"
    "Respond with a number between 0 and 100. Don't say anything else, just the number."
)


class _PersonaJudge:
    """0-100 logprob judge (mirrors tune_steering.py)."""

    def __init__(self, client: AsyncOpenAI, model: str, prompt_template: str):
        self.client = client
        self.model = model
        self.prompt_template = prompt_template

    async def judge(self, **kwargs) -> float | None:
        messages = [{"role": "user", "content": self.prompt_template.format(**kwargs)}]
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


def _load_trait_judge(trait: str, judge_model: str, client: AsyncOpenAI) -> "_PersonaJudge | None":
    """Return a trait judge for trait, or None if no data file exists."""
    filetrait = TRAIT_MAP.get(trait, trait)
    data_path = _TRAIT_DATA_DIR / f"{filetrait}.json"
    if not data_path.exists():
        return None
    with open(data_path) as f:
        data = json.load(f)
    eval_prompt = data["eval_prompt"].replace("{{", "{").replace("}}", "}")
    return _PersonaJudge(client, judge_model, eval_prompt)


async def score_csv_async(
    input_path: Path,
    output_path: Path,
    judge_model: str,
    client: AsyncOpenAI,
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

    if output_path.exists() and not force:
        existing = pd.read_csv(output_path)
        if "trait1_score" in existing.columns and len(existing) == len(df):
            print(f"  Already scored ({len(existing)} rows), skipping. Use --force to re-run.")
            return existing

    # Build one judge per unique trait across the whole file
    all_traits = set(df["trait1"].unique()) | set(df["trait2"].unique())
    trait_judges: dict[str, _PersonaJudge] = {}
    coherence_judge = _PersonaJudge(client, judge_model, _COHERENCE_PROMPT)
    for trait in all_traits:
        judge = _load_trait_judge(trait, judge_model, client)
        if judge is None:
            print(f"  Warning: no judge data for '{trait}' — those scores will be NaN")
        else:
            trait_judges[trait] = judge

    result_df = df.copy()
    result_df["trait1_score"] = float("nan")
    result_df["trait2_score"] = float("nan")
    result_df["coherence_score"] = float("nan")

    semaphore = asyncio.Semaphore(concurrency)

    async def score_row(row_idx: int, question: str, answer: str, trait1: str, trait2: str):
        judge1 = trait_judges.get(trait1)
        judge2 = trait_judges.get(trait2)
        coros = []
        labels = []
        if judge1:
            coros.append(judge1.judge(question=question, answer=answer))
            labels.append("trait1")
        if judge2:
            coros.append(judge2.judge(question=question, answer=answer))
            labels.append("trait2")
        coros.append(coherence_judge.judge(question=question, answer=answer))
        labels.append("coherence")

        async with semaphore:
            results = await asyncio.gather(*coros)

        return row_idx, dict(zip(labels, results))

    tasks = [
        score_row(idx, row["question"], row["generated_output"], row["trait1"], row["trait2"])
        for idx, row in df.iterrows()
    ]

    print(f"  Scoring {len(tasks)} rows...")
    scored = []
    with tqdm(total=len(tasks), desc="  scoring", ncols=70, leave=False) as pbar:
        for coro in asyncio.as_completed(tasks):
            scored.append(await coro)
            pbar.update(1)

    for row_idx, scores in scored:
        if "trait1" in scores and scores["trait1"] is not None:
            result_df.at[row_idx, "trait1_score"] = scores["trait1"]
        if "trait2" in scores and scores["trait2"] is not None:
            result_df.at[row_idx, "trait2_score"] = scores["trait2"]
        if "coherence" in scores and scores["coherence"] is not None:
            result_df.at[row_idx, "coherence_score"] = scores["coherence"]

    result_df.to_csv(output_path, index=False)

    n = len(result_df)
    for col in ("trait1_score", "trait2_score", "coherence_score"):
        mean = result_df[col].mean()
        n_scored = result_df[col].notna().sum()
        print(f"  {col:<20} {mean:.1f} mean  ({n_scored}/{n} scored)")
    print(f"  Saved → {output_path}")
    return result_df


async def main_async(args) -> int:
    input_files: list[Path] = []
    for pattern in args.input_csvs:
        expanded = glob.glob(pattern)
        if expanded:
            input_files.extend(Path(p) for p in expanded)
        else:
            input_files.append(Path(pattern))
    input_files = [p for p in input_files if p.exists() and p.stat().st_size > 0]

    if not input_files:
        print("Error: no non-empty input files found")
        return 1

    # Validate format against the first readable file
    required = {"trait1", "trait2", "question", "generated_output"}
    sample_file = None
    for p in input_files:
        try:
            sample = pd.read_csv(p, nrows=1)
            sample_file = p
            break
        except Exception:
            continue
    if sample_file is None:
        print("Error: could not read any input file")
        return 1
    missing = required - set(sample.columns)
    if missing:
        print(f"Error: input CSV is missing required columns: {missing}")
        print("Expected experiment_3 format with columns: group, trait1, trait2, method, alpha, prompt_trait, question, generated_output")
        return 1

    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    client = AsyncOpenAI()

    print(f"Scoring {len(input_files)} file(s) with judge model '{args.judge_model}'")
    print("=" * 60)

    summaries = []
    for input_path in sorted(input_files):
        if output_dir:
            output_path = output_dir / f"{input_path.stem}_scored.csv"
        else:
            output_path = input_path.with_name(f"{input_path.stem}_scored.csv")

        print(f"\n{input_path.name}")
        try:
            result_df = await score_csv_async(
                input_path=input_path,
                output_path=output_path,
                judge_model=args.judge_model,
                client=client,
                concurrency=args.concurrency,
                force=args.force,
            )
        except Exception as e:
            print(f"  Error: {e}")
            continue
        if result_df is not None:
            summaries.append({
                "file": input_path.name,
                "rows": len(result_df),
                "mean_trait1_score": result_df["trait1_score"].mean(),
                "mean_trait2_score": result_df["trait2_score"].mean(),
                "mean_coherence_score": result_df["coherence_score"].mean(),
                "n_scored": int(result_df["trait1_score"].notna().sum()),
            })

    if len(summaries) > 1:
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"\n{'File':<40} {'T1 score':>9} {'T2 score':>9} {'Coherence':>10} {'N':>5}")
        print("-" * 74)
        for s in summaries:
            def fmt(v):
                return f"{v:.1f}" if v == v else " nan"
            print(
                f"{s['file']:<40}"
                f" {fmt(s['mean_trait1_score']):>9}"
                f" {fmt(s['mean_trait2_score']):>9}"
                f" {fmt(s['mean_coherence_score']):>10}"
                f" {s['n_scored']:>5}"
            )

    print("\nDone.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "input_csvs", nargs="+",
        help="Experiment-3 CSV file(s). Glob patterns are supported.",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Directory for scored CSVs (default: same directory as each input file)",
    )
    parser.add_argument(
        "--judge-model", default="gpt-4o",
        help="OpenAI model to use for judging (default: gpt-4o)",
    )
    parser.add_argument(
        "--concurrency", type=int, default=50,
        help="Max simultaneous OpenAI requests (default: 50)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-score files that already have a scored output",
    )
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
