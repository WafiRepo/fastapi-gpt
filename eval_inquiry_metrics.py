#!/usr/bin/env python3
"""
Evaluate inquiry quality with BLEU-1, BLEU-4, ROUGE, BERTScore, and G-eval.

Input JSONL schema (one item per line):
{
  "id": "Q1",
  "lang": "en" | "id",
  "stage": "problem_finding" | "problem_exploring",
  "inquiry": "...",  # Generated inquiry question
  "reference": [...],  # List of example good inquiries (for evaluation)
  "context": "...",          # optional
  "difficulty": "easy|intermediate|advanced"  # optional
}
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Try to load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv  # type: ignore
    
    # Try to find .env file in current directory or parent directories
    env_paths = [
        Path(__file__).parent / ".env",  # fastapi-gpt/.env
        Path(__file__).parent.parent / ".env",  # root/.env
        Path.cwd() / ".env",  # current working directory/.env
    ]
    for env_path in env_paths:
        if env_path.exists():
            load_dotenv(env_path)
            break
    else:
        # If no .env found, try loading from default location
        load_dotenv()
except ImportError:
    # python-dotenv not installed, skip .env loading
    pass


DEFAULT_RUBRIC = [
    {
        "name": "Relevance to inquiry",
        "description": "How well the response addresses the inquiry.",
        "scale": "1=irrelevant, 3=partially relevant, 5=fully relevant",
    },
    {
        "name": "Scientific correctness",
        "description": "Accuracy of physics concepts and reasoning.",
        "scale": "1=incorrect, 3=mixed, 5=correct",
    },
    {
        "name": "Clarity and completeness",
        "description": "Clarity, completeness, and coherence.",
        "scale": "1=unclear, 3=somewhat clear, 5=clear and complete",
    },
    {
        "name": "Use of context or data",
        "description": "Appropriate use of provided context or data.",
        "scale": "1=not used, 3=limited use, 5=well integrated",
    },
]

# Rubric khusus untuk evaluasi kualitas inquiry question
INQUIRY_EVAL_RUBRIC = [
    {
        "name": "Stage alignment",
        "description": "How well the inquiry matches the intended stage (problem_finding/problem_exploring).",
        "scale": "1=not aligned, 3=partially aligned, 5=fully aligned",
    },
    {
        "name": "Topic relevance",
        "description": "How well the inquiry relates to the topic (centripetal acceleration).",
        "scale": "1=irrelevant, 3=somewhat relevant, 5=highly relevant",
    },
    {
        "name": "Clarity and structure",
        "description": "Is the inquiry clear, well-structured, and appropriate for students?",
        "scale": "1=unclear, 3=somewhat clear, 5=clear and well-structured",
    },
    {
        "name": "Context integration",
        "description": "How well the inquiry uses the provided context.",
        "scale": "1=not used, 3=limited use, 5=well integrated",
    },
    {
        "name": "Inquiry quality",
        "description": "Does it invite curiosity, reasoning, or application as appropriate for the stage?",
        "scale": "1=poor inquiry, 3=adequate, 5=excellent inquiry",
    },
]


# Few-shot examples for evaluating generated inquiry questions
# Note: For problem_generating stage, candidate is the generated question (not an answer)
FEW_SHOT_GENERATED_INQUIRY_EXAMPLES = [
    {
        "stage": "problem_generating",
        "lang": "en",
        "inquiry": "Create an easy question using the observed rotating fan and data.",
        "candidate": "If a fan blade has r = 0.15 m and w = 10 rad/s, find a.",
        "references": ["Form a question asking for a using a = r * w^2 with given r and w."],
        "context": "Fan",
        "difficulty": "easy",
        "evaluation": {
            "overall_score": 5.0,
            "rubric_scores": {
                "Relevance to inquiry": 5,
                "Scientific correctness": 5,
                "Clarity and completeness": 5,
                "Use of context or data": 5
            },
            "rationale": "The generated question perfectly matches the inquiry prompt. It creates an easy physics problem using the fan context, provides all necessary data (r and w), asks for centripetal acceleration (a), and is structured as a clear, solvable question. The question correctly follows the formula a = r * w^2 and demonstrates appropriate difficulty level for 'easy'."
        }
    },
    {
        "stage": "problem_generating",
        "lang": "en",
        "inquiry": "Generate an intermediate problem from a spinning wheel.",
        "candidate": "A wheel spins. Find acceleration.",
        "references": ["Ask for both a and v using a = r * w^2 and v = r * w."],
        "context": "Bicycle wheel",
        "difficulty": "intermediate",
        "evaluation": {
            "overall_score": 2.0,
            "rubric_scores": {
                "Relevance to inquiry": 2,
                "Scientific correctness": 2,
                "Clarity and completeness": 1,
                "Use of context or data": 3
            },
            "rationale": "The generated question is poorly structured. It lacks essential information: no values for r or w are provided, making it unsolvable. It doesn't specify 'centripetal acceleration', and fails to demonstrate intermediate-level complexity (should ask for multiple quantities like both a and v). The question is too vague and incomplete to be a valid physics problem."
        }
    },
    {
        "stage": "problem_generating",
        "lang": "en",
        "inquiry": "Generate an intermediate problem from a spinning wheel.",
        "candidate": "A wheel of radius 0.30 m spins at 6 rad/s. Find centripetal acceleration and linear speed.",
        "references": ["Ask for both a and v using a = r * w^2 and v = r * w."],
        "context": "Bicycle wheel",
        "difficulty": "intermediate",
        "evaluation": {
            "overall_score": 5.0,
            "rubric_scores": {
                "Relevance to inquiry": 5,
                "Scientific correctness": 5,
                "Clarity and completeness": 5,
                "Use of context or data": 5
            },
            "rationale": "The generated question excellently matches the intermediate inquiry. It provides all necessary data (r = 0.30 m, w = 6 rad/s), asks for both centripetal acceleration and linear speed, demonstrating intermediate complexity. The question is clear, complete, and properly uses the spinning wheel context. It correctly requires students to apply both a = r * w^2 and v = r * w formulas."
        }
    },
    {
        "stage": "problem_generating",
        "lang": "id",
        "inquiry": "Buat soal mudah dari kipas berputar dengan data.",
        "candidate": "Jika r = 0.15 m dan w = 10 rad/s, hitung a.",
        "references": ["Soal meminta a dengan rumus a = r * w^2 menggunakan r dan w."],
        "context": "Kipas",
        "difficulty": "easy",
        "evaluation": {
            "overall_score": 5.0,
            "rubric_scores": {
                "Relevance to inquiry": 5,
                "Scientific correctness": 5,
                "Clarity and completeness": 5,
                "Use of context or data": 5
            },
            "rationale": "Pertanyaan yang dihasilkan sangat sesuai dengan prompt inquiry. Soal menggunakan konteks kipas, menyediakan semua data yang diperlukan (r dan w), meminta perhitungan percepatan sentripetal (a), dan terstruktur sebagai soal fisika yang jelas dan dapat diselesaikan. Soal menggunakan rumus a = r * w^2 dengan benar dan menunjukkan tingkat kesulitan yang sesuai untuk 'mudah'."
        }
    },
    {
        "stage": "problem_generating",
        "lang": "en",
        "inquiry": "Create an advanced question involving measurement error.",
        "candidate": "If r = 0.20 m +/- 0.01 m and w = 8 rad/s, estimate the range for a.",
        "references": ["Ask to compute min and max a using r bounds."],
        "context": "Measurement uncertainty",
        "difficulty": "advanced",
        "evaluation": {
            "overall_score": 4.75,
            "rubric_scores": {
                "Relevance to inquiry": 5,
                "Scientific correctness": 5,
                "Clarity and completeness": 4.5,
                "Use of context or data": 5
            },
            "rationale": "The generated question excellently addresses the advanced inquiry about measurement error. It correctly incorporates uncertainty in r (+/- 0.01 m), uses appropriate physics concepts, and requires students to compute a range (min and max values), demonstrating advanced-level thinking. The question properly uses the measurement uncertainty context. Minor improvement: could explicitly state 'centripetal acceleration' for absolute clarity, though 'a' is standard notation."
        }
    }
]


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no}: {exc}") from exc
    return rows


def as_references(ref: Any) -> List[str]:
    if isinstance(ref, list):
        return [str(r) for r in ref if r is not None]
    if ref is None:
        return []
    return [str(ref)]


def ensure_required_fields(item: Dict[str, Any], idx: int) -> None:
    """
    Ensure required fields are present.
    Only inquiry is required; candidate and reference are optional (used as examples for evaluation).
    """
    required = ["id", "stage", "inquiry"]
    missing = [k for k in required if k not in item or item[k] is None]
    if missing:
        raise ValueError(f"Item {idx} missing required fields: {', '.join(missing)}")


def compute_bleu(candidate: str, references: List[str], lang: str) -> Tuple[float, float]:
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction  # type: ignore
    except ImportError:
        try:
            import nltk  # type: ignore
            nltk.download("punkt", quiet=True)
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction  # type: ignore
        except Exception as exc:
            raise RuntimeError("nltk is required for BLEU. Install with: pip install nltk") from exc

    # Tokenize candidate and references
    candidate_tokens = candidate.split()
    reference_tokens_list = [ref.split() for ref in references]
    
    # Use exponential smoothing (similar to sacrebleu's "exp" method)
    smooth = SmoothingFunction().method1
    
    # BLEU-1: only unigrams (weights = (1.0, 0.0, 0.0, 0.0))
    bleu1 = sentence_bleu(
        reference_tokens_list,
        candidate_tokens,
        weights=(1.0, 0.0, 0.0, 0.0),
        smoothing_function=smooth,
    )
    
    # BLEU-4: standard 4-gram (weights = (0.25, 0.25, 0.25, 0.25))
    bleu4 = sentence_bleu(
        reference_tokens_list,
        candidate_tokens,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=smooth,
    )
    
    return bleu1, bleu4


def compute_corpus_bleu(
    candidates: List[str],
    references_list: List[List[str]],
) -> Tuple[float, float]:
    try:
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction  # type: ignore
    except ImportError:
        try:
            import nltk  # type: ignore
            nltk.download("punkt", quiet=True)
            from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction  # type: ignore
        except Exception as exc:
            raise RuntimeError("nltk is required for BLEU. Install with: pip install nltk") from exc

    smooth = SmoothingFunction().method1
    references_tokens = [[ref.split() for ref in refs] for refs in references_list]
    candidate_tokens = [cand.split() for cand in candidates]

    bleu1 = corpus_bleu(
        references_tokens,
        candidate_tokens,
        weights=(1.0, 0.0, 0.0, 0.0),
        smoothing_function=smooth,
    )
    bleu4 = corpus_bleu(
        references_tokens,
        candidate_tokens,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=smooth,
    )
    return bleu1, bleu4


def compute_rouge(candidate: str, references: List[str], lang: str) -> Dict[str, float]:
    try:
        from rouge_score import rouge_scorer  # type: ignore
    except Exception as exc:
        raise RuntimeError("rouge-score is required for ROUGE. Install with: pip install rouge-score") from exc

    use_stemmer = lang == "en"
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=use_stemmer)

    best = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    for ref in references:
        scores = scorer.score(ref, candidate)
        best["rouge1"] = max(best["rouge1"], scores["rouge1"].fmeasure)
        best["rouge2"] = max(best["rouge2"], scores["rouge2"].fmeasure)
        best["rougeL"] = max(best["rougeL"], scores["rougeL"].fmeasure)
    return best


def compute_bertscore(candidate: str, references: List[str], lang: str) -> Dict[str, float]:
    try:
        from bert_score import score as bert_score  # type: ignore
    except Exception as exc:
        raise RuntimeError("bert-score is required. Install with: pip install bert-score") from exc

    model_type = "roberta-large" if lang == "en" else "xlm-roberta-large"

    best_p, best_r, best_f1 = 0.0, 0.0, 0.0
    for ref in references:
        p, r, f1 = bert_score(
            [candidate],
            [ref],
            model_type=model_type,
            lang=None if lang != "en" else "en",
            verbose=False,
        )
        best_p = max(best_p, float(p[0]))
        best_r = max(best_r, float(r[0]))
        best_f1 = max(best_f1, float(f1[0]))
    return {"precision": best_p, "recall": best_r, "f1": best_f1}


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    snippet = match.group(0)
    try:
        return json.loads(snippet)
    except json.JSONDecodeError:
        return None


def compute_g_eval_inquiry(
    stage: str,
    lang: str,
    inquiry: str,
    context: str,
    rubric: List[Dict[str, str]],
    model: str,
    use_few_shot: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate quality of generated inquiry question (standalone evaluation).
    """
    try:
        import requests  # type: ignore
    except Exception as exc:
        raise RuntimeError("requests is required for G-eval. Install with: pip install requests") from exc

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        print("WARNING: OPENAI_API_KEY is not set. Skipping G-eval.", file=sys.stderr)
        return {
            "overall_score": None,
            "rubric_scores": {},
            "rationale": "G-eval skipped: OPENAI_API_KEY not set",
        }

    system_msg = (
        "You are a strict evaluator for inquiry questions in physics education. "
        "Evaluate the quality of the inquiry question based on the rubric. "
        "Return JSON only with keys: overall_score (1-5), rubric_scores (object), rationale (string)."
    )

    messages = [{"role": "system", "content": system_msg}]

    scoring_instructions = (
        "Score each rubric from 1 to 5. overall_score is the average. "
        "Evaluate the inquiry question quality: "
        "- Does it match the stage requirements (problem_finding invites observation/curiosity, conceptual probes reasoning, application requires problem-solving)? "
        "- Is it relevant to the topic (centripetal acceleration)? "
        "- Is it clear, well-structured, and appropriate for Grade 11 students? "
        "- Does it use the context effectively? "
        "- Does it invite the right type of thinking for the stage?"
    )

    user_payload = {
        "stage": stage,
        "lang": lang,
        "inquiry": inquiry,
        "context": context,
        "rubric": rubric,
        "scoring_instructions": scoring_instructions,
    }
    messages.append({"role": "user", "content": json.dumps(user_payload, ensure_ascii=True)})

    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": messages,
            "temperature": 0,
        },
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    content = data["choices"][0]["message"]["content"]

    parsed = _extract_json(content)
    if not parsed:
        raise RuntimeError("Failed to parse JSON from G-eval response.")
    return parsed


def compute_g_eval(
    stage: str,
    lang: str,
    inquiry: str,
    candidate: str,
    references: List[str],
    context: str,
    rubric: List[Dict[str, str]],
    model: str,
    use_few_shot: bool = True,
) -> Dict[str, Any]:
    try:
        import requests  # type: ignore
    except Exception as exc:
        raise RuntimeError("requests is required for G-eval. Install with: pip install requests") from exc

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        print("WARNING: OPENAI_API_KEY is not set. Skipping G-eval.", file=sys.stderr)
        return {
            "overall_score": None,
            "rubric_scores": {},
            "rationale": "G-eval skipped: OPENAI_API_KEY not set",
        }

    # Customize system message based on stage
    if stage == "problem_generating":
        system_msg = (
            "You are a strict evaluator for generated physics questions. "
            "For problem_generating stage, the 'candidate' is a QUESTION generated from the inquiry prompt template, "
            "not an answer. Evaluate the quality of the generated question: whether it matches the inquiry prompt, "
            "is scientifically correct, clear and complete, and properly uses context/data. "
            "Return JSON only with keys: overall_score (1-5), rubric_scores (object), rationale (string)."
        )
    else:
        system_msg = (
            "You are a strict evaluator. Return JSON only with keys: "
            "overall_score (1-5), rubric_scores (object), rationale (string)."
        )

    # Build messages with few-shot examples for problem_generating stage
    messages = [{"role": "system", "content": system_msg}]
    
    # Add few-shot examples if evaluating generated inquiry questions
    if use_few_shot and stage == "problem_generating":
        # Filter few-shot examples by language
        few_shot_examples = [
            ex for ex in FEW_SHOT_GENERATED_INQUIRY_EXAMPLES
            if ex["lang"] == lang or lang not in ["en", "id"]
        ]
        
        # Use up to 3 examples (prefer matching language, then any)
        if few_shot_examples:
            examples_to_use = few_shot_examples[:3]
        else:
            # Fallback to English examples if no language match
            examples_to_use = [ex for ex in FEW_SHOT_GENERATED_INQUIRY_EXAMPLES if ex["lang"] == "en"][:3]
        
        for example in examples_to_use:
            example_payload = {
                "stage": example["stage"],
                "lang": example["lang"],
                "inquiry": example["inquiry"],
                "candidate": example["candidate"],
                "references": example["references"],
                "context": example.get("context", ""),
                "rubric": rubric,
                "scoring_instructions": (
                    "Score each rubric from 1 to 5. overall_score is the average. "
                    "Remember: candidate is a GENERATED QUESTION, not an answer. "
                    "Evaluate whether the generated question matches the inquiry prompt, "
                    "is a well-formed physics problem, and demonstrates appropriate difficulty level."
                ),
            }
            messages.append({
                "role": "user",
                "content": json.dumps(example_payload, ensure_ascii=True)
            })
            messages.append({
                "role": "assistant",
                "content": json.dumps(example["evaluation"], ensure_ascii=True)
            })

    # Customize scoring instructions based on stage
    if stage == "problem_generating":
        scoring_instructions = (
            "Score each rubric from 1 to 5. overall_score is the average. "
            "Remember: candidate is a GENERATED QUESTION, not an answer. "
            "Evaluate whether the generated question matches the inquiry prompt, "
            "is a well-formed physics problem, and demonstrates appropriate difficulty level."
        )
    else:
        scoring_instructions = "Score each rubric from 1 to 5. overall_score is the average."

    user_payload = {
        "stage": stage,
        "lang": lang,
        "inquiry": inquiry,
        "candidate": candidate,
        "references": references,
        "context": context,
        "rubric": rubric,
        "scoring_instructions": scoring_instructions,
    }
    messages.append({"role": "user", "content": json.dumps(user_payload, ensure_ascii=True)})

    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": messages,
            "temperature": 0,
        },
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    content = data["choices"][0]["message"]["content"]

    parsed = _extract_json(content)
    if not parsed:
        raise RuntimeError("Failed to parse JSON from G-eval response.")
    return parsed


def mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def write_metric_curves(results: List[Dict[str, Any]], output_dir: str) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:
        raise RuntimeError("matplotlib is required for PNG plots. Install with: pip install matplotlib") from exc

    os.makedirs(output_dir, exist_ok=True)
    x = list(range(1, len(results) + 1))

    metric_keys = [
        "bleu1",
        "bleu4",
        "rouge1",
        "rouge2",
        "rougeL",
        "bertscore_precision",
        "bertscore_recall",
        "bertscore_f1",
        "geval_overall",
    ]

    for key in metric_keys:
        values: List[Optional[float]] = []
        for row in results:
            val = row.get(key)
            if isinstance(val, (int, float)):
                values.append(float(val))
            else:
                values.append(None)

        if all(v is None for v in values):
            continue

        y = [v if v is not None else float("nan") for v in values]
        plt.figure(figsize=(10, 4))
        plt.plot(x, y, marker="o", linewidth=1, markersize=3)
        plt.title(f"{key} curve")
        plt.xlabel("Inquiry index")
        plt.ylabel(key)
        plt.ylim(0, 1)
        plt.tight_layout()
        filename = os.path.join(output_dir, f"{key}_curve.png")
        plt.savefig(filename, dpi=150)
        plt.close()

    # Combined plot: BLEU in top panel (0-100 or 0-1), others in bottom panel (0-1).
    bleu_keys = ["bleu1", "bleu4"]
    other_keys = ["rouge1", "rouge2", "rougeL", "bertscore_f1", "geval_overall"]

    def series_for(keys: List[str]) -> Dict[str, List[float]]:
        out: Dict[str, List[float]] = {}
        for key in keys:
            values: List[Optional[float]] = []
            for row in results:
                val = row.get(key)
                if isinstance(val, (int, float)):
                    values.append(float(val))
                else:
                    values.append(None)
            if all(v is None for v in values):
                continue
            out[key] = [v if v is not None else float("nan") for v in values]
        return out

    bleu_series = series_for(bleu_keys)
    other_series = series_for(other_keys)

    if bleu_series or other_series:
        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        if bleu_series:
            for key, y in bleu_series.items():
                axes[0].plot(x, y, marker="o", linewidth=1, markersize=3, label=key)
            axes[0].set_title("BLEU curves")
            axes[0].set_ylabel("BLEU")
            axes[0].set_ylim(0, 1)
            axes[0].legend(loc="best")
            axes[0].grid(True, alpha=0.3)
        else:
            axes[0].set_visible(False)

        if other_series:
            for key, y in other_series.items():
                axes[1].plot(x, y, marker="o", linewidth=1, markersize=3, label=key)
            axes[1].set_title("ROUGE / BERTScore / G-eval curves")
            axes[1].set_xlabel("Inquiry index")
            axes[1].set_ylabel("Score")
            axes[1].set_ylim(0, 1)
            axes[1].legend(loc="best")
            axes[1].grid(True, alpha=0.3)
        else:
            axes[1].set_visible(False)

        plt.tight_layout()
        combined_path = os.path.join(output_dir, "combined_metrics.png")
        plt.savefig(combined_path, dpi=150)
        plt.close()


def write_summary_plot(summary: Dict[str, float], output_dir: str) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:
        raise RuntimeError("matplotlib is required for PNG plots. Install with: pip install matplotlib") from exc

    os.makedirs(output_dir, exist_ok=True)
    metrics = [
        "bleu1_mean",
        "rougeL_mean",
        "bertscore_precision_mean",
        "bertscore_recall_mean",
        "bertscore_f1_mean",
        "geval_overall_mean_normalized",
    ]
    values = [summary.get(k, 0.0) for k in metrics]

    plt.figure(figsize=(10, 4))
    plt.plot(metrics, values, marker="o", linewidth=1)
    plt.title("Summary metrics (line plot)")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    filename = os.path.join(output_dir, "summary_metrics.png")
    plt.savefig(filename, dpi=150)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate inquiry responses with text metrics and G-eval.")
    parser.add_argument("--input", required=True, help="Path to input JSONL.")
    parser.add_argument("--output-jsonl", default="eval_results.jsonl", help="Path to output JSONL.")
    parser.add_argument("--output-csv", default="eval_results.csv", help="Path to output CSV.")
    parser.add_argument("--no-bertscore", action="store_true", help="Skip BERTScore.")
    parser.add_argument("--no-geval", action="store_true", help="Skip G-eval.")
    parser.add_argument("--geval-model", default=os.getenv("G_EVAL_MODEL", "gpt-4"), help="Model for G-eval.")
    parser.add_argument(
        "--no-few-shot",
        action="store_true",
        help="Disable few-shot examples for G-eval (few-shot is enabled by default for problem_generating stage).",
    )
    parser.add_argument(
        "--bleu-scale",
        choices=["0-1", "0-100"],
        default="0-1",
        help="Output scale for BLEU scores.",
    )
    parser.add_argument(
        "--plots-dir",
        default="eval_plots",
        help="Directory to write PNG curve plots per metric.",
    )
    args = parser.parse_args()

    items = load_jsonl(args.input)

    results: List[Dict[str, Any]] = []
    aggregate = {
        "bleu1": [],
        "bleu4": [],
        "rouge1": [],
        "rouge2": [],
        "rougeL": [],
        "bertscore_precision": [],
        "bertscore_recall": [],
        "bertscore_f1": [],
        "geval_overall": [],
    }
    corpus_candidates: List[str] = []
    corpus_references: List[List[str]] = []

    for idx, item in enumerate(items, start=1):
        ensure_required_fields(item, idx)
        lang = str(item.get("lang", "en")).lower()
        stage = str(item["stage"])
        inquiry = str(item["inquiry"])
        references = as_references(item.get("reference", []))
        context = str(item.get("context", ""))

        row: Dict[str, Any] = {
            "id": item["id"],
            "lang": lang,
            "stage": stage,
            "prompt_type": item.get("prompt_type", ""),  # Include prompt_type if available
        }

        # EVALUASI INQUIRY QUALITY
        # Text-based metrics: inquiry vs reference
        # Reference digunakan untuk membandingkan kualitas inquiry yang dihasilkan
        if references:
            bleu1, bleu4 = compute_bleu(inquiry, references, lang)
            if args.bleu_scale == "0-100":
                bleu1 *= 100.0
                bleu4 *= 100.0
            rouge = compute_rouge(inquiry, references, lang)
            
            row.update({
                "bleu1": bleu1,
                "bleu4": bleu4,
                "rouge1": rouge["rouge1"],
                "rouge2": rouge["rouge2"],
                "rougeL": rouge["rougeL"],
            })
            
            aggregate["bleu1"].append(bleu1)
            aggregate["bleu4"].append(bleu4)
            aggregate["rouge1"].append(rouge["rouge1"])
            aggregate["rouge2"].append(rouge["rouge2"])
            aggregate["rougeL"].append(rouge["rougeL"])
            corpus_candidates.append(inquiry)
            corpus_references.append(references)

            if not args.no_bertscore:
                bs = compute_bertscore(inquiry, references, lang)
                row.update(
                    {
                        "bertscore_precision": bs["precision"],
                        "bertscore_recall": bs["recall"],
                        "bertscore_f1": bs["f1"],
                    }
                )
                aggregate["bertscore_precision"].append(bs["precision"])
                aggregate["bertscore_recall"].append(bs["recall"])
                aggregate["bertscore_f1"].append(bs["f1"])
        else:
            # Jika tidak ada reference, set metrics ke None
            row.update({
                "bleu1": None,
                "bleu4": None,
                "rouge1": None,
                "rouge2": None,
                "rougeL": None,
            })

        # G-eval untuk inquiry quality (standalone)
        if not args.no_geval:
            ge = compute_g_eval_inquiry(
                stage=stage,
                lang=lang,
                inquiry=inquiry,
                context=context,
                rubric=INQUIRY_EVAL_RUBRIC,
                model=args.geval_model,
                use_few_shot=not args.no_few_shot,
            )
            row.update(
                {
                    "geval_overall": ge.get("overall_score"),
                    "geval_rubric_scores": ge.get("rubric_scores"),
                    "geval_rationale": ge.get("rationale"),
                }
            )
            if isinstance(ge.get("overall_score"), (int, float)):
                aggregate["geval_overall"].append(float(ge["overall_score"]))

        results.append(row)

    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    csv_columns = [
        "id",
        "lang",
        "stage",
        "prompt_type",
        "bleu1",
        "bleu4",
        "rouge1",
        "rouge2",
        "rougeL",
        "bertscore_precision",
        "bertscore_recall",
        "bertscore_f1",
        "geval_overall",
    ]
    with open(args.output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k, "") for k in csv_columns})

    corpus_bleu1, corpus_bleu4 = compute_corpus_bleu(corpus_candidates, corpus_references)
    if args.bleu_scale == "0-100":
        corpus_bleu1 *= 100.0
        corpus_bleu4 *= 100.0

    geval_norm = mean(aggregate["geval_overall"]) / 5 if aggregate["geval_overall"] else 0.0
    summary = {
        "bleu1_mean": mean(aggregate["bleu1"]),
        "bleu4_mean": mean(aggregate["bleu4"]),
        "bleu1_corpus": corpus_bleu1,
        "bleu4_corpus": corpus_bleu4,
        "rouge1_mean": mean(aggregate["rouge1"]),
        "rouge2_mean": mean(aggregate["rouge2"]),
        "rougeL_mean": mean(aggregate["rougeL"]),
        "bertscore_precision_mean": mean(aggregate["bertscore_precision"]),
        "bertscore_recall_mean": mean(aggregate["bertscore_recall"]),
        "bertscore_f1_mean": mean(aggregate["bertscore_f1"]),
        "geval_overall_mean": mean(aggregate["geval_overall"]),
        "geval_overall_mean_normalized": geval_norm,
        "items": len(results),
    }
    print(json.dumps(summary, ensure_ascii=True, indent=2))

    write_metric_curves(results, args.plots_dir)
    write_summary_plot(summary, args.plots_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
