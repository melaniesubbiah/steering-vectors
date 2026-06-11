#!/usr/bin/env python3
"""
Sweep coefficient and layer for each steering vector, scored by a GPT-4o judge.

For each trait in final_vectors/qwen/:
  For each layer in --layers (default [16, 20]):
    For each coeff in --coeffs (default [0.5, 1.0, 1.5, 2.0]):
      Generate --n-prompts outputs with the scaled steering vector applied
      Score each with GPT-4o binary YES/NO (does the output exhibit the trait?)
      Record mean P(YES) score

Outputs one row per (trait, layer, coeff, prompt_idx) to --output CSV.
Prints a best-(layer, coeff) summary per trait at the end.

Usage (from eval_prose/src/):
    python tune_steering.py
    python tune_steering.py --layers 16 20 --coeffs 0.5 1.0 1.5 2.0 --n-prompts 5
    python tune_steering.py --traits allcaps_emphasis emoji_usage
    python tune_steering.py --output tune_results/my_sweep.csv
"""

import argparse
import asyncio
import json
import math
import random
import sys
from pathlib import Path

import torch
import pandas as pd
import numpy as np
from openai import AsyncOpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add ml-predict to path (mirrors predefined_steering.py setup)
_mlp_root = Path(__file__).resolve().parents[1] / "ml-predict"
if _mlp_root.exists() and str(_mlp_root) not in sys.path:
    sys.path.insert(0, str(_mlp_root))

import preference_inferrer.tasks.writing.writing_quality_assesor as wq_module
from preference_inferrer.tasks.task_instances import WritingTaskInstance
from preference_inferrer.preference_sets.writing_preference_set import WritingPreferenceSet
from preference_inferrer.common.configurations import configure_config

from activation_steer import ActivationSteerer


# ── Persona judge support ─────────────────────────────────────────────────────

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
    """0-100 logprob judge that accepts an AsyncOpenAI client — avoids module-level side-effects."""

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


def _load_persona_judges(
    trait: str, judge_model: str, client: AsyncOpenAI
) -> tuple | None:
    """Load (trait_judge, coherence_judge, questions) for trait. Returns None if no data file exists."""
    filetrait = TRAIT_MAP.get(trait, trait)
    data_path = _TRAIT_DATA_DIR / f"{filetrait}.json"
    if not data_path.exists():
        return None
    with open(data_path) as f:
        data = json.load(f)
    eval_prompt = data["eval_prompt"].replace("{{", "{").replace("}}", "}")
    trait_judge = _PersonaJudge(client, judge_model, eval_prompt)
    coherence_judge = _PersonaJudge(client, judge_model, _COHERENCE_PROMPT)
    return trait_judge, coherence_judge, data["questions"]



# ── Natural-language preference strings recognised by WritingPreferenceSet ────
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

# Fixed email-writing prompts used for all (trait, layer, coeff) combinations.
SAMPLE_PROMPTS = [
    "Notes: Team meeting scheduled for Monday 3pm. Agenda: Q3 results, hiring plans, product roadmap update.\nPlease write an email based on the above notes:",
    "Notes: Following up on last week's client call. They requested a price reduction of 15%. We can offer 10%.\nPlease write an email based on the above notes:",
    "Notes: Congratulating Sarah on her promotion to Senior Engineer. She has been with the company for 5 years.\nPlease write an email based on the above notes:",
    "Notes: Requesting a sick day tomorrow. Have a doctor's appointment in the morning.\nPlease write an email based on the above notes:",
    "Notes: Declining the conference invitation due to scheduling conflicts. Would like to attend next year.\nPlease write an email based on the above notes:",
]

async def _eval_task(evaluator, task_instance, pref_set):
    ppcm, _ = await evaluator.llm_judges_preference_matching(
        task_instance, pref_set, per_preference=True
    )
    return ppcm


async def _eval_batched(eval_tasks: list, max_concurrent: int = 50) -> list:
    """Run judge evaluations concurrently with a semaphore, preserving order."""
    semaphore = asyncio.Semaphore(max_concurrent)
    results = [None] * len(eval_tasks)

    async def run(idx, evaluator, task_instance, pref_set):
        async with semaphore:
            score = await _eval_task(evaluator, task_instance, pref_set)
            return idx, score

    for coro in asyncio.as_completed(
        [run(i, ev, ti, ps) for i, (ev, ti, ps) in enumerate(eval_tasks)]
    ):
        idx, score = await coro
        results[idx] = score

    return results


def _make_task_instance(text: str, task_type: str = "email_writing") -> WritingTaskInstance:
    ti = WritingTaskInstance(task_type=task_type, task_content="", source="tune_sweep", source_idx=0)
    ti.agent_completion = text
    ti.user_completion = ""
    return ti


def generate(
    model,
    tokenizer,
    prompt: str,
    steering_vec: torch.Tensor | None,
    layer: int,
    max_new_tokens: int,
) -> str:
    """Generate greedily with optional activation steering on response tokens."""
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]

    gen_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )

    if steering_vec is None:
        with torch.no_grad():
            out = model.generate(**gen_kwargs)
    else:
        with ActivationSteerer(
            model,
            steering_vec,
            layer_idx=layer-1,
            positions="response",
            prompt_length=prompt_len,
        ):
            with torch.no_grad():
                out = model.generate(**gen_kwargs)

    return tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)


def _is_better(new_score: float, new_coeff: float, best: dict) -> bool:
    """True if (new_score, new_coeff) beats best under the tie-breaking rules:
    highest score > lowest coeff > first found (caller keeps existing on equal)."""
    ns, bs = round(new_score, 3), round(best["score"], 3)
    if ns > bs:
        return True
    if ns == bs and new_coeff < best["coeff"]:
        return True
    return False


def _save_best(path: Path, best_settings: dict) -> None:
    with open(path, "w") as f:
        json.dump(best_settings, f, indent=2)


async def run_sweep(args):
    vector_dir = Path(args.vector_dir)
    out_path = Path(args.output)
    best_json_path = Path(args.best_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    best_json_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing best settings so already-tuned traits are skipped.
    best_settings: dict = {}
    if best_json_path.exists():
        with open(best_json_path) as f:
            best_settings = json.load(f)
        print(f"Loaded {len(best_settings)} existing result(s) from {best_json_path}")

    # Load model once; reuse across all combos.
    print(f"Loading {args.model}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="auto", torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.eval()
    print("Model ready.\n")

    _openai_client = AsyncOpenAI()
    coherence_judge = _PersonaJudge(_openai_client, args.judge_model, _COHERENCE_PROMPT)
    early_stop_threshold = 90.0 if args.judge_type == "persona" else 2.0

    # Initialize the same evaluator used in evaluate_prose.py
    config_dict = {
        "logging_level": "error",
        "base_dir": None,
        "user": {"llm_name": args.judge_model},
        "gpt_temperature": 0.01,
        "seed": 1352,
        "task": {"framework": "plume"},
    }
    eval_config = configure_config(config_dict)
    evaluator = wq_module.WritingQualityAssessor(eval_config)

    prompts = SAMPLE_PROMPTS[: args.n_prompts]

    # Discover traits
    vec_files = sorted(vector_dir.glob("*_response_avg_diff.pt"))
    traits = [f.name.replace("_response_avg_diff.pt", "") for f in vec_files]
    if args.traits:
        traits = [t for t in traits if t in args.traits]
    if not traits:
        sys.exit(f"No matching traits found in {vector_dir}")

    PHASE1_COEFFS = [1.0, 2.0, 3.0]
    PHASE1_COEFFS_RERUN = [3.0, 4.0, 5.0]  # used when best coeff lands at 3.5

    async def _score_combo(trait_rows, full_vector, pref_set, layer, coeff, label, trait_judge=None, prompts_override=None):
        """Generate + score one (layer, coeff) combo. Returns (mean_score, mean_coh)."""
        effective_prompts = prompts_override if prompts_override is not None else prompts
        base_vec = full_vector[layer].float()
        model_dtype = next(model.parameters()).dtype
        scaled_vec = (coeff * base_vec).to(dtype=model_dtype, device=model.device)
        texts = [generate(model, tokenizer, p, scaled_vec, layer, args.max_new_tokens) for p in effective_prompts]

        coherence_scores = list(await asyncio.gather(
            *(coherence_judge.judge(question=p, answer=t) for p, t in zip(effective_prompts, texts))
        ))
        valid_coh = [c for c in coherence_scores if c is not None]
        mean_coh = sum(valid_coh) / len(valid_coh) if valid_coh else float("nan")

        if trait_judge is not None:
            scores = list(await asyncio.gather(
                *(trait_judge.judge(question=p, answer=t) for p, t in zip(effective_prompts, texts))
            ))
        else:
            eval_tasks = [(evaluator, _make_task_instance(t), pref_set) for t in texts]
            scores = await _eval_batched(eval_tasks)

        valid = [s for s in scores if s is not None]
        mean_score = sum(valid) / len(valid) if valid else float("nan")
        print(f"  {label}  layer={layer}  coeff={coeff:.2f}  score={mean_score:.3f}  coherence={mean_coh:.1f}")
        for i, (text, score, coh) in enumerate(zip(texts, scores, coherence_scores)):
            trait_rows.append({"trait": trait, "layer": layer, "coeff": float(coeff),
                                "prompt_idx": i, "score": score, "coherence": coh, "text": text})
        return mean_score, mean_coh

    for trait in traits:
        if trait in best_settings:
            s = best_settings[trait]
            if round(s["coeff"], 1) != 3.5:
                print(f"Skipping {trait} (already tuned: layer={s['layer']} coeff={s['coeff']:.2f} score={s['score']:.3f})")
                continue
            print(f"Re-running {trait} (saved coeff={s['coeff']:.2f} is at boundary — trying higher coefficients)")
            phase1_coeffs = PHASE1_COEFFS_RERUN
        else:
            phase1_coeffs = PHASE1_COEFFS

        vec_path = vector_dir / f"{trait}_response_avg_diff.pt"
        full_vector = torch.load(vec_path, map_location="cpu", weights_only=True)
        pref_str = TRAIT_TO_PREFERENCE.get(trait, trait.replace("_", " "))
        pref_set = WritingPreferenceSet([pref_str], eval_config)

        trait_judge = None
        trait_prompts = None
        if args.judge_type == "persona":
            loaded = _load_persona_judges(trait, args.judge_model, _openai_client)
            if loaded is None:
                print(f"  No persona judge data for {trait}, skipping")
                continue
            trait_judge, _, questions = loaded
            n = min(args.n_prompts, len(questions))
            trait_prompts = random.sample(questions, n)

        best_for_trait = {"layer": None, "coeff": None, "score": float("-inf")}
        found_perfect = False
        trait_rows = []
        ran_higher_coeffs = (phase1_coeffs is PHASE1_COEFFS_RERUN)

        while True:
            total_p1 = len(args.layers) * len(phase1_coeffs)
            label_suffix = " [higher-coeff rerun]" if ran_higher_coeffs else ""
            # ── Phase 1: coarse search × all layers ───────────────────────────
            print(f"\n{trait}  phase 1 ({total_p1} combos){label_suffix}")
            done = 0
            for layer in args.layers:
                if found_perfect:
                    break
                for coeff in phase1_coeffs:
                    done += 1
                    mean_score, mean_coh = await _score_combo(
                        trait_rows, full_vector, pref_set, layer, coeff,
                        f"[p1 {done}/{total_p1}]", trait_judge=trait_judge,
                        prompts_override=trait_prompts,
                    )
                    coherent = mean_coh is None or mean_coh >= 75.0
                    if coherent and _is_better(mean_score, coeff, best_for_trait):
                        best_for_trait = {"layer": layer, "coeff": float(coeff), "score": mean_score}
                    if coherent and round(mean_score, 3) >= early_stop_threshold:
                        print(f"  Early stop (score >= {early_stop_threshold})")
                        found_perfect = True
                        break

            # ── Phase 2: fine search ±0.5 around phase-1 winner ──────────────
            if not found_perfect and best_for_trait["layer"] is not None:
                best_layer = best_for_trait["layer"]
                best_coeff = best_for_trait["coeff"]
                tested = set(phase1_coeffs)

                fine_start = round(best_coeff - 0.5, 1)
                fine_end   = round(best_coeff + 0.5, 1)
                n_steps    = round((fine_end - fine_start) / 0.1)
                fine_coeffs = [
                    round(fine_start + i * 0.1, 1)
                    for i in range(n_steps + 1)
                    if round(fine_start + i * 0.1, 1) > 0
                    and round(fine_start + i * 0.1, 1) not in tested
                ]

                print(f"\n{trait}  phase 2 ({len(fine_coeffs)} combos, layer={best_layer}){label_suffix}")
                for i, coeff in enumerate(fine_coeffs, 1):
                    mean_score, mean_coh = await _score_combo(
                        trait_rows, full_vector, pref_set, best_layer, coeff,
                        f"[p2 {i}/{len(fine_coeffs)}]", trait_judge=trait_judge,
                        prompts_override=trait_prompts,
                    )
                    coherent = mean_coh is None or mean_coh >= 75.0
                    if coherent and _is_better(mean_score, coeff, best_for_trait):
                        best_for_trait = {"layer": best_layer, "coeff": float(coeff), "score": mean_score}
                    if coherent and round(mean_score, 3) >= early_stop_threshold:
                        print(f"  Early stop (score >= {early_stop_threshold})")
                        break

            # ── Re-run with higher coefficients if result landed at 3.5 ───────
            if (
                not ran_higher_coeffs
                and best_for_trait["layer"] is not None
                and round(best_for_trait["coeff"], 1) == 3.5
            ):
                print(f"  Best coeff is 3.5 — re-running with higher coefficients {PHASE1_COEFFS_RERUN}")
                phase1_coeffs = PHASE1_COEFFS_RERUN
                best_for_trait = {"layer": None, "coeff": None, "score": float("-inf")}
                found_perfect = False
                ran_higher_coeffs = True
                continue
            break

        if best_for_trait["layer"] is not None:
            best_settings[trait] = best_for_trait
            _save_best(best_json_path, best_settings)
            print(f"  Best for {trait}: layer={best_for_trait['layer']}  coeff={best_for_trait['coeff']:.2f}  score={best_for_trait['score']:.3f}\n")

        # Flush this trait's rows to CSV immediately.
        trait_df = pd.DataFrame(trait_rows)
        if out_path.exists():
            trait_df = pd.concat([pd.read_csv(out_path), trait_df], ignore_index=True)
        trait_df.to_csv(out_path, index=False)
        print(f"  CSV updated: {out_path}")

    print(f"\nBest settings saved to {best_json_path}")


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--vector-dir", default="final_vectors/qwen",
                   help="Directory containing *_response_avg_diff.pt files")
    p.add_argument("--layers", type=int, nargs="+", default=[20, 16])
    p.add_argument("--n-prompts", type=int, default=5,
                   help="Number of sample prompts per (trait, layer, coeff) combination")
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--judge-model", default="gpt-4o",
                   help="OpenAI model for WritingQualityAssessor (default: gpt-4o)")
    p.add_argument("--output", default="tune_results/steering_sweep.csv")
    p.add_argument("--best-json", default="tune_results/best_settings.json",
                   help="JSON file for saving/loading best (layer, coeff) per trait")
    p.add_argument("--traits", nargs="*",
                   help="Specific trait names to sweep (default: all in --vector-dir)")
    p.add_argument(
        "--judge-type", choices=["prose", "persona"], default="prose",
        help="Judge to use: 'prose' (WritingQualityAssessor, scores ~0-2) or "
             "'persona' (trait+coherence judges from eval_persona, scores 0-100, "
             "coherence threshold 75, early-stop at 90). Default: prose",
    )
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(run_sweep(parse_args()))
