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
import subprocess
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


def _resolve_llm_config(allow_openai_fallback: bool = True) -> Dict[str, str]:
    explicit_key = os.getenv("LLM_API_KEY", "").strip()
    explicit_base = os.getenv("LLM_API_BASE_URL", "").strip()
    if explicit_key or explicit_base:
        return {
            # Ollama biasanya tidak butuh API key; pakai dummy jika base URL explicit diberikan.
            "api_key": explicit_key or "dummy",
            "base_url": explicit_base or "https://api.openai.com/v1/chat/completions",
        }

    openrouter_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if openrouter_key:
        return {
            "api_key": openrouter_key,
            "base_url": os.getenv("OPENROUTER_API_BASE_URL", "").strip() or "https://openrouter.ai/api/v1/chat/completions",
        }

    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    if allow_openai_fallback and openai_key:
        return {
            "api_key": openai_key,
            "base_url": "https://api.openai.com/v1/chat/completions",
        }

    return {"api_key": "", "base_url": "https://api.openai.com/v1/chat/completions"}


def _llm_timeout() -> int:
    """Request timeout in seconds. Env LLM_REQUEST_TIMEOUT (default 180) untuk Ollama/lokal."""
    try:
        return max(60, int(os.getenv("LLM_REQUEST_TIMEOUT", "500")))
    except ValueError:
        return 180


def _llm_max_retries() -> int:
    """Total attempts for LLM HTTP requests (default 3)."""
    try:
        return max(1, int(os.getenv("LLM_MAX_RETRIES", "3")))
    except ValueError:
        return 3


def _post_with_retry(requests_module: Any, base_url: str, headers: Dict[str, str], payload: Dict[str, Any]) -> Any:
    """POST with retry/backoff for transient timeout/network/server errors."""
    attempts = _llm_max_retries()
    last_exc: Optional[Exception] = None
    for attempt in range(1, attempts + 1):
        try:
            response = requests_module.post(
                base_url,
                headers=headers,
                json=payload,
                timeout=_llm_timeout(),
            )
            response.raise_for_status()
            return response
        except requests_module.exceptions.RequestException as exc:  # type: ignore[attr-defined]
            last_exc = exc
            if attempt >= attempts:
                break
            wait_s = min(30, 2 ** (attempt - 1))
            print(
                f"WARNING: LLM request failed ({type(exc).__name__}) "
                f"[attempt {attempt}/{attempts}] to {base_url}. Retrying in {wait_s}s...",
                file=sys.stderr,
            )
            time.sleep(wait_s)

    raise RuntimeError(f"LLM request failed after {attempts} attempts: {last_exc}") from last_exc


def _is_openai_model(model: str) -> bool:
    m = (model or "").strip().lower()
    if m.startswith("openai/"):
        return True
    return m in ("gpt-4o", "gpt-4o-mini", "gpt-4", "gpt-4-turbo", "gpt-3.5-turbo")


def _is_openrouter_model(model: str) -> bool:
    """Models that must be routed via OpenRouter only."""
    m = (model or "").strip().lower()
    return m in (
        "qwen3-vl:30b",
        "qwen/qwen3.5-397b-a17b",
        "qwen/qwen3-vl-235b-a22b-thinking",
    )


OPENROUTER_MODEL_IDS = {
    "qwen3-vl:30b": "qwen/qwen3-vl-30b-a3b-instruct",
}


def _api_model_id(model: str, base_url: str) -> str:
    """Return model ID to send in API request. OpenRouter needs official id."""
    if "openrouter.ai" in (base_url or ""):
        m = (model or "").strip().lower()
        if m in OPENROUTER_MODEL_IDS:
            return OPENROUTER_MODEL_IDS[m]
    return model


def _resolve_openrouter_config() -> Dict[str, str]:
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not openrouter_key:
        raise RuntimeError("OpenRouter model requires OPENROUTER_API_KEY.")
    return {
        "api_key": openrouter_key,
        "base_url": os.getenv("OPENROUTER_API_BASE_URL", "").strip() or "https://openrouter.ai/api/v1/chat/completions",
    }


def _resolve_llm_config_for_model(model: str) -> Dict[str, str]:
    """Resolve API config with strict provider routing per model."""
    if _is_openai_model(model):
        openai_key = os.getenv("OPENAI_API_KEY", "").strip()
        if openai_key:
            return {"api_key": openai_key, "base_url": "https://api.openai.com/v1/chat/completions"}
        return {"api_key": "", "base_url": "https://api.openai.com/v1/chat/completions"}
    if _is_openrouter_model(model):
        return _resolve_openrouter_config()
    raise RuntimeError(
        f"Model '{model}' is disabled. Use gpt-4o (OpenAI) or an allowed OpenRouter model."
    )


def _build_llm_headers(api_key: str, base_url: str) -> Dict[str, str]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if "openrouter.ai" in base_url:
        referer = os.getenv("OPENROUTER_HTTP_REFERER", "").strip()
        title = os.getenv("OPENROUTER_X_TITLE", "").strip() or "G-Uphysic evaluation"
        if referer:
            headers["HTTP-Referer"] = referer
        headers["X-Title"] = title
    return headers


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
        "description": "How well the inquiry matches the intended stage (problem_finding/problem_exploring/problem_generating).",
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
    if not candidates or not references_list or len(candidates) != len(references_list):
        return 0.0, 0.0
    valid_pairs = [(c, r) for c, r in zip(candidates, references_list) if r]
    if not valid_pairs:
        return 0.0, 0.0
    candidates = [c for c, _ in valid_pairs]
    references_list = [r for _, r in valid_pairs]

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
    if not references_tokens or not candidate_tokens:
        return 0.0, 0.0

    try:
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
    except ZeroDivisionError:
        return 0.0, 0.0


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
    if not references:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    # Jalankan BERTScore di subprocess untuk mengisolasi native crash (0xC0000005) di Windows.
    # Default ke CPU agar lebih stabil; override dengan env BERTSCORE_DEVICE jika diperlukan.
    model_type = "roberta-large" if lang == "en" else "xlm-roberta-large"
    device = os.getenv("BERTSCORE_DEVICE", "cpu").strip() or "cpu"
    timeout_s = int(os.getenv("BERTSCORE_TIMEOUT", "600"))

    payload = {
        "candidate": candidate,
        "references": references,
        "model_type": model_type,
        "lang": "en" if lang == "en" else None,
        "device": device,
    }

    worker_code = r"""
import json, os, sys
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from bert_score import score as bert_score

payload = json.loads(sys.stdin.read())
candidate = payload["candidate"]
references = payload["references"]
model_type = payload["model_type"]
lang = payload.get("lang")
device = payload.get("device") or "cpu"

cands = [candidate] * len(references)
p, r, f1 = bert_score(
    cands,
    references,
    model_type=model_type,
    lang=lang,
    verbose=False,
    batch_size=1,
    device=device,
)

print(json.dumps({
    "precision": float(max(p).item()),
    "recall": float(max(r).item()),
    "f1": float(max(f1).item()),
}, ensure_ascii=True))
"""

    try:
        proc = subprocess.run(
            [sys.executable, "-c", worker_code],
            input=json.dumps(payload, ensure_ascii=True),
            text=True,
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired:
        print(
            f"WARNING: BERTScore timed out after {timeout_s}s. Returning 0.0 scores.",
            file=sys.stderr,
        )
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    except Exception as exc:
        print(f"WARNING: Failed to run BERTScore subprocess: {exc}", file=sys.stderr)
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    if proc.returncode != 0:
        err = (proc.stderr or "").strip()
        msg = err.splitlines()[-1] if err else f"exit code {proc.returncode}"
        print(f"WARNING: BERTScore subprocess failed ({msg}). Returning 0.0 scores.", file=sys.stderr)
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    try:
        parsed = json.loads((proc.stdout or "").strip())
        return {
            "precision": float(parsed.get("precision", 0.0)),
            "recall": float(parsed.get("recall", 0.0)),
            "f1": float(parsed.get("f1", 0.0)),
        }
    except Exception:
        print("WARNING: Invalid BERTScore subprocess output. Returning 0.0 scores.", file=sys.stderr)
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}


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

    llm_cfg = _resolve_llm_config_for_model(model)
    api_key = llm_cfg["api_key"]
    base_url = llm_cfg["base_url"]
    if not api_key:
        print("WARNING: No API key is set for G-eval model. Skipping G-eval.", file=sys.stderr)
        return {
            "overall_score": None,
            "rubric_scores": {},
            "rationale": "G-eval skipped: no API key set",
        }

    system_msg = (
        "You are a strict evaluator for inquiry questions in physics education. "
        "Evaluate the quality of the inquiry question based on the rubric. "
        "Return JSON only with keys: overall_score (1-5), rubric_scores (object), rationale (string)."
    )

    messages = [{"role": "system", "content": system_msg}]

    if stage == "problem_generating":
        scoring_instructions = (
            "Score each rubric from 1 to 5. overall_score is the average. "
            "Evaluate generated problem question quality: "
            "- Does it clearly represent a solvable physics problem (not explanation text)? "
            "- Is the problem relevant to centripetal acceleration? "
            "- Is it clear, complete, and appropriate for Grade 11 students? "
            "- Does it integrate context/data from provided multimodal setup when context is available? "
            "- Is the cognitive demand aligned with the intended difficulty level if mentioned?"
        )
    else:
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

    response = _post_with_retry(
        requests_module=requests,
        base_url=base_url,
        headers=_build_llm_headers(api_key=api_key, base_url=base_url),
        payload={
            "model": _api_model_id(model, base_url),
            "messages": messages,
            "temperature": 0,
        },
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

    llm_cfg = _resolve_llm_config_for_model(model)
    api_key = llm_cfg["api_key"]
    base_url = llm_cfg["base_url"]
    if not api_key:
        print("WARNING: No API key is set for G-eval model. Skipping G-eval.", file=sys.stderr)
        return {
            "overall_score": None,
            "rubric_scores": {},
            "rationale": "G-eval skipped: no API key set",
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

    response = _post_with_retry(
        requests_module=requests,
        base_url=base_url,
        headers=_build_llm_headers(api_key=api_key, base_url=base_url),
        payload={
            "model": _api_model_id(model, base_url),
            "messages": messages,
            "temperature": 0,
        },
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


def write_metric_curves(results: List[Dict[str, Any]], output_dir: str, bleu_scale: str = "0-1") -> None:
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
        if key in ("bleu1", "bleu4"):
            if bleu_scale == "0-100":
                plt.ylim(0, 100)
            else:
                plt.ylim(0, 1)
        elif key == "geval_overall":
            plt.ylim(0, 5)
        else:
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
            if key == "geval_overall":
                # Normalize G-eval (1-5) to 0-1 for combined panel consistency.
                out["geval_overall_norm"] = [
                    (v / 5.0) if v is not None else float("nan")
                    for v in values
                ]
            else:
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
            axes[0].set_ylim(0, 100 if bleu_scale == "0-100" else 1)
            axes[0].legend(loc="best")
            axes[0].grid(True, alpha=0.3)
        else:
            axes[0].set_visible(False)

        if other_series:
            for key, y in other_series.items():
                axes[1].plot(x, y, marker="o", linewidth=1, markersize=3, label=key)
            axes[1].set_title("ROUGE / BERTScore / G-eval(norm) curves")
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


def write_difficulty_plots(results: List[Dict[str, Any]], output_dir: str, bleu_scale: str = "0-1") -> None:
    """
    Plot summary per difficulty (easy/intermediate/advanced) for problem_generating stage.
    """
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:
        raise RuntimeError("matplotlib is required for PNG plots. Install with: pip install matplotlib") from exc

    rows = [
        r for r in results
        if str(r.get("stage", "")) == "problem_generating" and str(r.get("difficulty", "")).strip()
    ]
    if not rows:
        return

    order = ["easy", "intermediate", "advanced"]
    grouped: Dict[str, List[Dict[str, Any]]] = {k: [] for k in order}
    for r in rows:
        key = str(r.get("difficulty", "")).lower().strip()
        if key in grouped:
            grouped[key].append(r)

    available = [k for k in order if grouped[k]]
    if not available:
        return

    def mean_metric(items: List[Dict[str, Any]], metric: str) -> float:
        vals: List[float] = []
        for it in items:
            v = it.get(metric)
            if isinstance(v, (int, float)):
                vals.append(float(v))
        return mean(vals) if vals else 0.0

    bleu_vals = [mean_metric(grouped[d], "bleu1") for d in available]
    rouge_vals = [mean_metric(grouped[d], "rougeL") for d in available]
    bert_vals = [mean_metric(grouped[d], "bertscore_f1") for d in available]
    geval_vals_norm = [mean_metric(grouped[d], "geval_overall") / 5.0 for d in available]

    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    axes[0].bar(available, bleu_vals, color="#4C78A8")
    axes[0].set_title("BLEU-1 mean by difficulty (problem_generating)")
    axes[0].set_ylabel("BLEU-1")
    axes[0].set_ylim(0, 100 if bleu_scale == "0-100" else 1)
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].plot(available, rouge_vals, marker="o", linewidth=1.5, label="rougeL_mean")
    axes[1].plot(available, bert_vals, marker="o", linewidth=1.5, label="bertscore_f1_mean")
    axes[1].plot(available, geval_vals_norm, marker="o", linewidth=1.5, label="geval_overall_norm_mean")
    axes[1].set_title("Normalized quality metrics by difficulty")
    axes[1].set_ylabel("Score (0-1)")
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best")

    plt.tight_layout()
    filename = os.path.join(output_dir, "difficulty_metrics.png")
    plt.savefig(filename, dpi=150)
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
            "item_group_id": item.get("item_group_id", ""),
            "lang": lang,
            "stage": stage,
            "difficulty": item.get("difficulty", ""),
            "prompt_type": item.get("prompt_type", ""),  # Include prompt_type if available
            "knowledge_mode": item.get("knowledge_mode", "none"),
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
            try:
                ge = compute_g_eval_inquiry(
                    stage=stage,
                    lang=lang,
                    inquiry=inquiry,
                    context=context,
                    rubric=INQUIRY_EVAL_RUBRIC,
                    model=args.geval_model,
                    use_few_shot=not args.no_few_shot,
                )
            except Exception as exc:
                print(
                    f"WARNING: G-eval failed for id={item['id']} stage={stage} lang={lang}: {exc}",
                    file=sys.stderr,
                )
                ge = {
                    "overall_score": None,
                    "rubric_scores": {},
                    "rationale": f"G-eval skipped due to error: {exc}",
                }
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
        "item_group_id",
        "lang",
        "stage",
        "difficulty",
        "prompt_type",
        "knowledge_mode",
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

    if not corpus_candidates or not corpus_references:
        corpus_bleu1, corpus_bleu4 = 0.0, 0.0
        print("WARNING: Empty corpus for BLEU; using corpus BLEU = 0.0.", file=sys.stderr)
    else:
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

    write_metric_curves(results, args.plots_dir, bleu_scale=args.bleu_scale)
    write_difficulty_plots(results, args.plots_dir, bleu_scale=args.bleu_scale)
    write_summary_plot(summary, args.plots_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
