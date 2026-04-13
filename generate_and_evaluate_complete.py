#!/usr/bin/env python3
"""
Complete workflow: Generate inquiry data with multiple prompt strategies and evaluate.

This script implements:
1. Generation with 4 prompt strategies: zero-shot, few-shot, CoT, draft-critique-revise
2. Evaluation with multiple metrics (BLEU, ROUGE, BERTScore, G-eval)
3. Comparison of prompt strategies
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import mimetypes
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests  # type: ignore

# Fix encoding untuk Windows console
if sys.platform == "win32":
    try:
        import io
        if sys.stdout.encoding != 'utf-8':
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        if sys.stderr.encoding != 'utf-8':
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass  # Skip if can't change encoding

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


CONTEXTS = {
    "problem_finding": [
        "Classroom with a ceiling fan",
        "Bicycle wheel spinning",
        "Washing machine drum in motion",
        "Playground carousel rotating",
        "Playground hamster wheel",
        "Desk fan on a table",
        "Turntable platter spinning",
        "Office chair wheel rotating",
    ],
    "problem_exploring": [
        "Rotating fan blades",
        "Spinning bicycle wheel",
        "Washer on a string",
        "Turntable",
        "hamster wheel playground"
        "carousel playground",
        "Experiment data: a vs r graph",
        "Experiment data: w vs r graph",
        "Measured string length",
        "Tension in a string setup",
        "Sensor log with stable period",
        "Rotating disk",
    ],
}

TOPIC = "centripetal acceleration"
AUDIENCE = "Grade 11"
PROBLEM_GENERATING_STAGE = "problem_generating" #problem finding #problem exploring
PROBLEM_GENERATING_DIFFICULTIES = ["easy", "intermediate", "advanced"]
PROBLEM_GENERATING_GRAPH_DIRS = {
    "easy": Path(__file__).parent / "easy_graph",
    "intermediate": Path(__file__).parent / "intermediate_graph",
    "advanced": Path(__file__).parent / "advance_graph",
}
PROBLEM_GENERATING_IMAGE_DIR = Path(__file__).parent / "image_result" 
SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
# Default: OpenAI + OpenRouter only (no Ollama route).
DEFAULT_MODEL_COMPARISON = [
    "gpt-4o",
    "qwen3-vl:30b", #visual + STEM
    "qwen/qwen3-vl-235b-a22b-thinking", #visual +science
]

# Some models are kept in the raw evaluation JSON/summary but excluded from plots for clarity.
EXCLUDED_MODELS_FOR_PLOTS = {
    "qwen3-vl_8b",
}

# Path ke folder prompts
PROMPTS_DIR = Path(__file__).parent / "data" / "prompts"
KNOWLEDGE_GRAPH_PATH = Path(__file__).parent / "data" / "knowledge_graphs" / "centripetal_acceleration.json"
DEFAULT_KNOWLEDGE_MODES = ["none", "kg"] 
_KNOWLEDGE_GRAPH_CACHE: Optional[Dict[str, Any]] = None


def _sanitize_model_name(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model.strip())


def _load_knowledge_graph(kg_path: Optional[str] = None) -> Dict[str, Any]:
    global _KNOWLEDGE_GRAPH_CACHE
    resolved_path = Path(kg_path) if kg_path else KNOWLEDGE_GRAPH_PATH
    if _KNOWLEDGE_GRAPH_CACHE is not None and resolved_path == KNOWLEDGE_GRAPH_PATH:
        return _KNOWLEDGE_GRAPH_CACHE
    if not resolved_path.exists():
        raise FileNotFoundError(f"Knowledge graph file not found: {resolved_path}")
    with open(resolved_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Knowledge graph must be a JSON object.")
    if resolved_path == KNOWLEDGE_GRAPH_PATH:
        _KNOWLEDGE_GRAPH_CACHE = data
    return data


def _append_unique_lines(target: List[str], values: List[str]) -> None:
    for value in values:
        cleaned = str(value).strip()
        if cleaned and cleaned not in target:
            target.append(cleaned)


def _normalize_kg_text(value: str) -> str:
    return re.sub(r"[_\-/]+", " ", str(value or "").lower()).strip()


def _keyword_match_score(query_text: str, candidate_name: str, keywords: List[str]) -> int:
    normalized_query = _normalize_kg_text(query_text)
    score = 0

    candidate_phrase = _normalize_kg_text(candidate_name)
    if candidate_phrase and candidate_phrase in normalized_query:
        score += 6

    for keyword in keywords:
        normalized_keyword = _normalize_kg_text(keyword)
        if not normalized_keyword:
            continue
        if normalized_keyword in normalized_query:
            score += 8 if " " in normalized_keyword else 5
            continue
        keyword_tokens = [token for token in normalized_keyword.split() if token]
        if keyword_tokens and all(token in normalized_query for token in keyword_tokens):
            score += 3

    return score


def _rank_kg_entries(
    entries: Dict[str, Any],
    query_text: str,
    stage: str,
    stage_bias: Optional[Dict[str, int]] = None,
) -> List[Dict[str, Any]]:
    ranked: List[Dict[str, Any]] = []
    for entry_name, payload in entries.items():
        if not isinstance(payload, dict):
            continue
        keywords = [str(v) for v in payload.get("keywords", []) if str(v).strip()]
        score = _keyword_match_score(query_text, entry_name, keywords)
        if stage_bias and entry_name in stage_bias:
            score += int(stage_bias[entry_name])
        if score <= 0:
            continue
        ranked.append(
            {
                "name": entry_name,
                "payload": payload,
                "score": score,
            }
        )
    ranked.sort(key=lambda item: (-int(item["score"]), str(item["name"])))
    return ranked


def _stage_top_k(stage: str) -> Dict[str, int]:
    if stage == "problem_finding":
        return {"objects": 2, "patterns": 1, "concepts": 2, "formulas": 0}
    if stage == "problem_exploring":
        return {"objects": 1, "patterns": 2, "concepts": 2, "formulas": 1}
    return {"objects": 1, "patterns": 2, "concepts": 2, "formulas": 2}

def retrieve_knowledge_context(
    stage: str,
    context: str,
    difficulty: Optional[str],
    multimodal_asset_paths: List[str],
    kg_path: Optional[str] = None,
) -> str:
    kg = _load_knowledge_graph(kg_path)
    asset_names = [Path(p).name for p in multimodal_asset_paths if p]
    asset_stems = [Path(p).stem for p in multimodal_asset_paths if p]
    query_parts = [str(context or ""), stage.replace("_", " "), str(difficulty or "")]
    query_parts.extend(asset_names)
    query_parts.extend(asset_stems)
    joined = " ".join(part for part in query_parts if str(part).strip())

    concepts = kg.get("concepts", {})
    objects = kg.get("objects", {})
    graph_patterns = kg.get("graph_patterns", {})
    stage_limits = _stage_top_k(stage)

    object_stage_bias = {
        "problem_finding": {"fan": 2, "carousel": 1, "wheel": 1, "washing_machine": 1},
        "problem_exploring": {"ball_on_string": 2, "wheel": 1},
        "problem_generating": {"ball_on_string": 1, "carousel": 1},
    }.get(stage, {})
    pattern_stage_bias = {
        "problem_exploring": {"a_vs_r": 2, "a_vs_w": 2, "period_data": 1},
        "problem_generating": {"a_vs_r": 2, "a_vs_w": 2, "period_data": 1},
    }.get(stage, {})

    ranked_objects = _rank_kg_entries(objects, joined, stage, object_stage_bias)
    ranked_patterns = _rank_kg_entries(graph_patterns, joined, stage, pattern_stage_bias)

    selected_objects = ranked_objects[: stage_limits["objects"]]
    selected_patterns = ranked_patterns[: stage_limits["patterns"]]

    selected_concept_ids: List[str] = []
    for item in selected_objects:
        payload = item.get("payload", {})
        if not isinstance(payload, dict):
            continue
        refs = [str(v) for v in payload.get("concepts", []) if str(v).strip()]
        _append_unique_lines(selected_concept_ids, refs)

    for item in selected_patterns:
        if "centripetal_acceleration" in concepts:
            _append_unique_lines(selected_concept_ids, ["centripetal_acceleration"])

    if not selected_concept_ids:
        fallback_concepts: List[str] = []
        if stage == "problem_finding":
            fallback_concepts = ["circular_motion", "centripetal_acceleration"]
        elif stage == "problem_exploring":
            fallback_concepts = ["centripetal_acceleration"]
        else:
            fallback_concepts = ["centripetal_acceleration", "circular_motion"]
        _append_unique_lines(
            selected_concept_ids,
            [concept_id for concept_id in fallback_concepts if concept_id in concepts],
        )

    matched_object_lines: List[str] = []
    matched_graph_lines: List[str] = []
    concept_lines: List[str] = []
    stage_lines: List[str] = []
    difficulty_lines: List[str] = []
    misconception_lines: List[str] = []

    if stage != "problem_finding":
        core_concepts = [str(line) for line in kg.get("core_concepts", []) if str(line).strip()]
        _append_unique_lines(concept_lines, core_concepts[:2])

    for item in selected_objects:
        obj_name = str(item["name"])
        payload = item.get("payload", {})
        if not isinstance(payload, dict):
            continue
        facts = [f"Object cue ({obj_name.replace('_', ' ')}): {fact}" for fact in payload.get("facts", [])]
        _append_unique_lines(matched_object_lines, facts[:2])

    for item in selected_patterns:
        pattern_name = str(item["name"])
        payload = item.get("payload", {})
        if not isinstance(payload, dict):
            continue
        facts = [f"Graph cue ({pattern_name.replace('_', ' ')}): {fact}" for fact in payload.get("facts", [])]
        _append_unique_lines(matched_graph_lines, facts[:2])

    formulas_added = 0
    for concept_id in selected_concept_ids[: stage_limits["concepts"]]:
        payload = concepts.get(concept_id, {})
        if not isinstance(payload, dict):
            continue
        label = concept_id.replace("_", " ")
        summary = str(payload.get("summary", "")).strip()
        if summary:
            _append_unique_lines(concept_lines, [f"{label.title()}: {summary}"])

        variables = [str(v).replace("_", " ") for v in payload.get("variables", []) if str(v).strip()]
        if variables and stage in {"problem_exploring", "problem_generating"}:
            _append_unique_lines(concept_lines, [f"Relevant variables: {', '.join(variables[:4])}"])

        formulas = [str(v) for v in payload.get("formulas", []) if str(v).strip()]
        if formulas and stage_limits["formulas"] > formulas_added:
            room = stage_limits["formulas"] - formulas_added
            chosen = formulas[:room]
            _append_unique_lines(concept_lines, [f"Formula: {formula}" for formula in chosen])
            formulas_added += len(chosen)

    stage_payload = kg.get("stage_rules", {})
    if isinstance(stage_payload, dict):
        _append_unique_lines(stage_lines, [str(v) for v in stage_payload.get(stage, [])[:2]])

    if difficulty:
        difficulty_payload = kg.get("difficulty_rules", {})
        if isinstance(difficulty_payload, dict):
            _append_unique_lines(difficulty_lines, [str(v) for v in difficulty_payload.get(str(difficulty), [])[:2]])

    misconceptions = kg.get("misconceptions_by_stage", {})
    if isinstance(misconceptions, dict):
        _append_unique_lines(misconception_lines, [str(v) for v in misconceptions.get(stage, [])[:2]])

    sections: List[str] = []
    if stage == "problem_finding":
        if matched_object_lines:
            sections.append("Object grounding:\n- " + "\n- ".join(matched_object_lines[:2]))
        if concept_lines:
            sections.append("Concept focus:\n- " + "\n- ".join(concept_lines[:3]))
    elif stage == "problem_exploring":
        if matched_graph_lines:
            sections.append("Graph grounding:\n- " + "\n- ".join(matched_graph_lines[:3]))
        if concept_lines:
            sections.append("Concept and variable focus:\n- " + "\n- ".join(concept_lines[:4]))
        if matched_object_lines:
            sections.append("Optional object grounding:\n- " + "\n- ".join(matched_object_lines[:1]))
    else:
        if matched_object_lines:
            sections.append("Image grounding:\n- " + "\n- ".join(matched_object_lines[:2]))
        if matched_graph_lines:
            sections.append("Graph grounding:\n- " + "\n- ".join(matched_graph_lines[:2]))
        if concept_lines:
            sections.append("Physics constraints:\n- " + "\n- ".join(concept_lines[:5]))

    if stage_lines:
        sections.append("Stage rules:\n- " + "\n- ".join(stage_lines[:2]))
    if difficulty_lines and stage in {"problem_exploring", "problem_generating"}:
        sections.append("Difficulty rules:\n- " + "\n- ".join(difficulty_lines[:2]))
    if misconception_lines:
        sections.append("Misconception guardrails:\n- " + "\n- ".join(misconception_lines[:2]))

    return "\n\n".join(section for section in sections if section).strip()


def maybe_inject_knowledge_context(
    prompt: str,
    stage: str,
    context: str,
    difficulty: Optional[str],
    multimodal_asset_paths: List[str],
    knowledge_mode: str = "none",
    kg_path: Optional[str] = None,
) -> str:
    if str(knowledge_mode).strip().lower() != "kg":
        return prompt
    knowledge_context = retrieve_knowledge_context(
        stage=stage,
        context=context,
        difficulty=difficulty,
        multimodal_asset_paths=multimodal_asset_paths,
        kg_path=kg_path,
    )
    if not knowledge_context:
        return prompt
    stage_instruction = {
        "problem_finding": (
            "Use the grounding mainly to anchor the example to a realistic circular-motion object. "
            "Do not turn the response into a full explanation."
        ),
        "problem_exploring": (
            "Use the grounding to focus on one main variable relationship or graph pattern. "
            "Avoid mixing too many concepts in one question."
        ),
        "problem_generating": (
            "Use the grounding to ensure the final problem is solvable and explicitly supported by both the graph and the image."
        ),
    }.get(
        stage,
        "Use the grounding to keep the output physically correct and relevant.",
    )
    return (
        prompt
        + "\n\nKnowledge Graph Context:\n"
        + knowledge_context
        + "\n\nUse the knowledge graph context as grounding support. "
          + stage_instruction
          + " Keep the final output concise and do not mention the knowledge graph explicitly."
    )


def plot_knowledge_graph_nodes(
    output_dir: str,
    kg_path: Optional[str] = None,
    filename: str = "kg_graph_nodes.png",
) -> Optional[str]:
    """
    Visualisasi struktur knowledge graph (node + edge) dari JSON KG.
    Menyimpan plot ke output_dir/filename. Mengembalikan path file atau None jika gagal.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
        import networkx as nx
    except ImportError:
        return None

    kg = _load_knowledge_graph(kg_path)
    if not kg or not isinstance(kg, dict):
        return None

    G = nx.DiGraph()
    topic = (kg.get("topic") or "topic").strip()
    topic_id = "topic"
    G.add_node(topic_id, label=topic, node_type="topic")

    # Concepts
    concepts = kg.get("concepts") or {}
    for cid, cdata in concepts.items():
        if not cid or cid in G:
            continue
        label = (cdata.get("summary") or cid)[:40] + ("..." if len((cdata.get("summary") or cid)) > 40 else "")
        G.add_node(cid, label=cid.replace("_", " ").title(), node_type="concept")
        G.add_edge(topic_id, cid, rel="has_concept")
        for var in (cdata.get("variables") or []):
            if var and var not in G:
                G.add_node(var, label=var.replace("_", " "), node_type="variable")
            if var:
                G.add_edge(cid, var, rel="has_variable")

    # Objects -> concepts
    objects = kg.get("objects") or {}
    for oid, odata in objects.items():
        if not oid or oid in G:
            continue
        G.add_node(oid, label=oid.replace("_", " ").title(), node_type="object")
        G.add_edge(topic_id, oid, rel="has_object")
        for c_ref in (odata.get("concepts") or []):
            if c_ref in G:
                G.add_edge(oid, c_ref, rel="uses_concept")

    # Graph patterns
    patterns = kg.get("graph_patterns") or {}
    for pid, pdata in patterns.items():
        if not pid or pid in G:
            continue
        G.add_node(pid, label=pid.replace("_", " ").title(), node_type="pattern")
        G.add_edge(topic_id, pid, rel="has_pattern")

    if G.number_of_nodes() == 0:
        return None

    fig, ax = plt.subplots(figsize=(12, 10))
    pos = nx.spring_layout(G, k=1.2, seed=42, iterations=50)
    node_types = nx.get_node_attributes(G, "node_type")
    type_color = {"topic": "#2ecc71", "concept": "#3498db", "object": "#e74c3c", "variable": "#95a5a6", "pattern": "#9b59b6"}
    type_label = {"topic": "Topic", "concept": "Concept", "object": "Object", "variable": "Variable", "pattern": "Graph pattern"}
    colors = [type_color.get(node_types.get(n, ""), "#bdc3c7") for n in G.nodes()]
    labels = {n: G.nodes[n].get("label", n) for n in G.nodes()}
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=1300, alpha=0.9, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color="#7f8c8d", arrows=True, arrowsize=12, ax=ax)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=9, font_weight="bold", ax=ax)
    legend_handles = [
        Patch(facecolor=type_color[t], edgecolor="none", label=type_label[t])
        for t in ("topic", "concept", "object", "variable", "pattern")
    ]
    ax.legend(handles=legend_handles, loc="upper left", framealpha=0.9)
    ax.set_title("Knowledge Graph: " + (topic or "centripetal acceleration"))
    ax.axis("off")
    plt.tight_layout()
    out_path = os.path.join(output_dir, filename)
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


def _resolve_llm_config(allow_openai_fallback: bool = True) -> Dict[str, str]:
    """
    Resolve provider config with priority:
    1) LLM_API_KEY + LLM_API_BASE_URL
    2) OPENROUTER_API_KEY (+ default OpenRouter base URL)
    3) OPENAI_API_KEY (+ default OpenAI base URL)
    """
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

    raise RuntimeError(
        "No API key found. Set one of: LLM_API_KEY, OPENROUTER_API_KEY, or OPENAI_API_KEY."
    )


def _llm_timeout() -> int:
    """Request timeout in seconds (Ollama/lokal sering butuh >60s). Env LLM_REQUEST_TIMEOUT override."""
    try:
        return max(60, int(os.getenv("LLM_REQUEST_TIMEOUT", "500")))
    except ValueError:
        return 180


def _build_llm_headers(api_key: str, base_url: str) -> Dict[str, str]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if "openrouter.ai" in base_url:
        referer = os.getenv("OPENROUTER_HTTP_REFERER", "").strip()
        title = os.getenv("OPENROUTER_X_TITLE", "").strip() or "G-Uphysic multimodal evaluation"
        if referer:
            headers["HTTP-Referer"] = referer
        headers["X-Title"] = title
    return headers


def _is_openai_model(model: str) -> bool:
    """True if model should use OpenAI API (not Ollama/OpenRouter)."""
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


# OpenRouter model ID (nama di API); alias -> id resmi
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
        raise RuntimeError("Model qwen3-vl:30b requires OPENROUTER_API_KEY.")
    return {
        "api_key": openrouter_key,
        "base_url": os.getenv("OPENROUTER_API_BASE_URL", "").strip() or "https://openrouter.ai/api/v1/chat/completions",
    }


def _resolve_llm_config_for_model(model: str) -> Dict[str, str]:
    """Resolve API config with strict provider routing per model."""
    if _is_openai_model(model):
        openai_key = os.getenv("OPENAI_API_KEY", "").strip()
        if openai_key:
            return {
                "api_key": openai_key,
                "base_url": "https://api.openai.com/v1/chat/completions",
            }
        raise RuntimeError("Model is OpenAI (gpt-4o etc.) but OPENAI_API_KEY is not set.")
    if _is_openrouter_model(model):
        return _resolve_openrouter_config()
    raise RuntimeError(
        f"Model '{model}' is disabled. Use gpt-4o (OpenAI) or an allowed OpenRouter model (e.g. qwen3-vl:30b, qwen/qwen3.5-397b-a17b, qwen/qwen3-vl-235b-a22b-thinking)."
    )


# P2: Language/complexity by ability band (personalization)
WORD_COUNT_RULES = {
    "low": (
        "The inquiry MUST be between 10 and 16 words (inclusive). "
        "Avoid subordinate clauses (e.g. 'when...', 'that...', 'which...'). "
        "Avoid two-step tasks (e.g. 'find X and then compare Y')."
    ),
    "medium": (
        "The inquiry MUST be between 16 and 20 words (inclusive). "
        "In problem_exploring stage only, you may include exactly one relation variable (e.g. 'between A and B')."
    ),
    "high": (
        "The inquiry MUST be between 20 and 25 words (inclusive). "
        "In problem_exploring stage only, you may include comparison or modification variables."
    ),
    "neutral": (
        "Use medium language load. No strict word-count or difficulty scaling."
    ),
}

# P3: Targeting instructions (personalization) — targeted vs screening
def get_targeting_instructions(is_targeted: bool, stage: str) -> str:
    if is_targeted:
        return (
            "The student has known misconceptions. Generate an inquiry that probes their understanding "
            "(e.g. direction of acceleration or velocity, or direction toward the center). "
            "Do not explain; elicit their thinking. Avoid phrasing that suggests 'outward force'."
        )
    if stage == "problem_finding":
        return (
            "No specific misconception is targeted. Generate a screening-style inquiry. "
            "Screen only object/phenomenon identification; no concepts. "
            "Do not target any specific misconception ID. Keep the screening focus minimal."
        )
    return (
        "No specific misconception is targeted. Generate a screening-style inquiry. "
        "Screen one foundational idea only (e.g. direction of acceleration OR inward direction, not both). "
        "Prefer screening direction evidence first. Keep the screening focus minimal."
    )


def load_prompt_template(prompt_type: str) -> str:
    """
    Load prompt template dari file.
    
    Args:
        prompt_type: zero_shot, few_shot, cot, atau draft_critique_revise
        
    Returns:
        Template string dengan placeholders
    """
    template_file = PROMPTS_DIR / f"{prompt_type}.md"
    
    if not template_file.exists():
        raise FileNotFoundError(
            f"Prompt template not found: {template_file}\n"
            f"Available templates: {list(PROMPTS_DIR.glob('*.md'))}"
        )
    
    with open(template_file, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Hapus header markdown jika ada (baris pertama yang dimulai dengan #)
    lines = content.split("\n")
    if lines[0].startswith("#"):
        # Skip header dan baris kosong setelahnya
        start_idx = 1
        while start_idx < len(lines) and not lines[start_idx].strip():
            start_idx += 1
        content = "\n".join(lines[start_idx:])
    
    return content.strip()


def build_zero_shot_prompt(stage: str, topic: str, audience: str, input_text: str) -> str:
    """Build zero-shot prompt berdasarkan template dari file."""
    template = load_prompt_template("zero_shot")
    prompt = template.replace("{STAGE}", stage)
    prompt = prompt.replace("{TOPIC}", topic)
    prompt = prompt.replace("{AUDIENCE}", audience)
    prompt = prompt.replace("{INPUT_TEXT}", input_text)
    return prompt


def build_few_shot_prompt(
    stage: str,
    topic: str,
    input_text: str,
    ability_band: str = "neutral",
    is_targeted: bool = False,
) -> str:
    """Build few-shot prompt dari template file. Supports P2 (ability band) and P3 (targeted vs screening)."""
    template = load_prompt_template("few_shot")
    word_count = WORD_COUNT_RULES.get(ability_band.lower(), WORD_COUNT_RULES["neutral"])
    targeting = get_targeting_instructions(is_targeted, stage)
    prompt = template.replace("{WORD_COUNT_AND_COMPLEXITY}", word_count)
    prompt = prompt.replace("{TARGETING_INSTRUCTIONS}", targeting)
    prompt = prompt.replace("{STAGE}", stage)
    prompt = prompt.replace("{TOPIC}", topic)
    prompt = prompt.replace("{INPUT_TEXT}", input_text)
    return prompt


def build_cot_prompt(stage: str, topic: str, audience: str, input_text: str) -> str:
    """Build Chain of Thought prompt dari template file."""
    template = load_prompt_template("cot")
    
    # Replace placeholders
    prompt = template.replace("{STAGE}", stage)
    prompt = prompt.replace("{TOPIC}", topic)
    prompt = prompt.replace("{AUDIENCE}", audience)
    prompt = prompt.replace("{INPUT_TEXT}", input_text)
    
    return prompt


def build_draft_critique_revise_prompt(stage: str, topic: str, audience: str, input_text: str) -> str:
    """Build draft-critique-revise prompt dari template file."""
    template = load_prompt_template("draft_critique_revise")
    
    # Replace placeholders
    prompt = template.replace("{STAGE}", stage)
    prompt = prompt.replace("{TOPIC}", topic)
    prompt = prompt.replace("{AUDIENCE}", audience)
    prompt = prompt.replace("{INPUT_TEXT}", input_text)
    
    return prompt


def generate_reference_only(
    model: str,
    stage: str,
    topic: str,
    audience: str,
    num_references: int = 3,
    difficulty: Optional[str] = None,
) -> List[str]:
    """
    Generate reference questions sekali per stage (bukan per context).
    Reference akan digunakan untuk semua contexts dan strategies pada stage yang sama.
    
    Args:
        num_references: Jumlah reference questions yang akan di-generate (default: 3)
    
    Returns: List of reference inquiry questions.
    """
    # Build reference list example based on num_references
    ref_examples = ", ".join([f'"question {i+1}"' for i in range(num_references)])
    format_example = f'{{"reference": [{ref_examples}]}}'
    
    # System message dengan instruksi khusus untuk problem_finding
    if stage == "problem_finding":
        system_msg = (
            f"You are an educational designer for physics. "
            f"IMPORTANT: Return JSON with ONLY the 'reference' key. Do NOT include 'inquiry' in your response. "
            f"The 'reference' is a list of exactly {num_references} example good inquiry questions (for evaluation comparison). "
            f"These reference questions will be used as gold standard to evaluate inquiry questions generated by different prompt strategies. "
            f"All reference questions are inquiry questions, not student responses. "
            f"CRITICAL for problem_finding stage: "
            f"DO NOT use the word 'object' or 'objects' in any reference question. "
            f"DO NOT mention specific object names (e.g., 'ceiling fan', 'bicycle wheel'). "
            f"The reference questions should be GENERAL and help students discover things, not mention specific objects. "
            f"Focus on finding general 'things', 'items', 'something' that demonstrate the topic. "
            f"Format: {format_example}"
        )
        user_prompt = (
            f"Stage: {stage}\n"
            f"Topic: {topic}\n"
            f"Audience: {audience}\n\n"
            f"Generate {num_references} general inquiry questions for problem_finding stage. "
            f"Questions should invite students to FIND or IDENTIFY things in their environment "
            f"that demonstrate {topic}, WITHOUT using the word 'object' or mentioning specific object names. "
            f"Make the questions general enough to apply to any context."
        )
    elif stage == PROBLEM_GENERATING_STAGE:
        level = difficulty or "intermediate"
        system_msg = (
            f"You are an educational designer for physics problem generation. "
            f"IMPORTANT: Return JSON with ONLY the 'reference' key. Do NOT include 'inquiry' in your response. "
            f"The 'reference' is a list of exactly {num_references} high-quality example {level}-level physics problem questions "
            f"(for evaluation comparison). "
            f"These references are used to evaluate generated problem questions in the same difficulty band. "
            f"Format: {format_example}"
        )
        user_prompt = (
            f"Stage: {stage}\n"
            f"Difficulty: {level}\n"
            f"Topic: {topic}\n"
            f"Audience: {audience}\n\n"
            f"Generate {num_references} reference problem questions at {level} level for centripetal acceleration. "
            f"Each question should be solvable, clear, and ask students to compute or reason from realistic given data. "
            f"Do not include solutions."
        )
    else: #problem exploring
        system_msg = (
            f"You are an educational designer for physics. "
            f"IMPORTANT: Return JSON with ONLY the 'reference' key. Do NOT include 'inquiry' in your response. "
            f"The 'reference' is a list of exactly {num_references} example good inquiry questions (for evaluation comparison). "
            f"These reference questions will be used as gold standard to evaluate inquiry questions generated by different prompt strategies. "
            f"All reference questions are inquiry questions, not student responses. "
            f"Format: {format_example}"
        )
        user_prompt = (
            f"Stage: {stage}\n"
            f"Topic: {topic}\n"
            f"Audience: {audience}\n\n"
            f"Generate {num_references} inquiry questions for problem_exploring stage. "
            f"Questions should focus on exploring and analyzing through hands-on experiments and real-world data. "
            f"Make the questions general enough to apply to any context."
        )
    
    # Retry loop
    llm_cfg = _resolve_llm_config_for_model(model)
    api_key = llm_cfg["api_key"]
    base_url = llm_cfg["base_url"]
    headers = _build_llm_headers(api_key=api_key, base_url=base_url)
    for attempt in range(4):
        try:
            response = requests.post(
                base_url,
                headers=headers,
                json={
                    "model": _api_model_id(model, base_url),
                    "messages": [
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0.2,
                },
                timeout=_llm_timeout(),
            )
            if response.status_code == 402:
                raise RuntimeError("402 Payment Required (e.g. OpenRouter quota). Use another model or provider.") from None
            if response.status_code == 500:
                time.sleep(10)  # Beri waktu Ollama pulih (OOM/overload)
                raise RuntimeError("500 Internal Server Error (Ollama overload/OOM?). Coba model lebih kecil atau tunggu.") from None
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            
            # Try to extract JSON
            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                # Try to extract JSON from text
                import re
                match = re.search(r"\{.*\}", content, re.DOTALL)
                if match:
                    data = json.loads(match.group(0))
                else:
                    raise RuntimeError(f"Invalid JSON from model: {content[:200]}")
            
            reference = data.get("reference", [])
            if not isinstance(reference, list):
                reference = [reference] if reference else []
            
            if not reference:
                raise ValueError("Generated reference is empty")
            
            if len(reference) != num_references:
                print(f"[WARNING] Expected {num_references} references, got {len(reference)}")
            
            return reference[:num_references]  # Ensure correct number
        except Exception as exc:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
    
    raise RuntimeError("Failed to generate reference after retries")


def _collect_image_files(target_dir: Path) -> List[Path]:
    if not target_dir.exists():
        return []
    files = [p for p in target_dir.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES]
    files.sort(key=lambda p: p.name.lower())
    return files


def build_problem_generating_contexts(count_per_difficulty: int) -> List[Dict[str, Any]]:
    """
    Build multimodal contexts for problem_generating stage by pairing one graph and one image.
    """
    image_files = _collect_image_files(PROBLEM_GENERATING_IMAGE_DIR)
    if not image_files:
        raise RuntimeError(
            f"No images found in {PROBLEM_GENERATING_IMAGE_DIR}. "
            "Please ensure fastapi-gpt/image_result contains image files."
        )

    entries: List[Dict[str, Any]] = []
    image_idx = 0
    for difficulty in PROBLEM_GENERATING_DIFFICULTIES:
        graph_dir = PROBLEM_GENERATING_GRAPH_DIRS[difficulty]
        graph_files = _collect_image_files(graph_dir)
        if not graph_files:
            print(f"[WARNING] No graph files found for difficulty '{difficulty}' in {graph_dir}. Skipping.")
            continue

        items_count = min(count_per_difficulty, len(graph_files))
        for i in range(items_count):
            graph_path = graph_files[i]
            image_path = image_files[image_idx % len(image_files)]
            image_idx += 1
            context = (
                f"Difficulty: {difficulty}. "
                f"Use the provided graph ({graph_path.name}) and real-world image ({image_path.name}) "
                "to construct one centripetal acceleration problem question."
            )
            entries.append(
                {
                    "context": context,
                    "difficulty": difficulty,
                    "multimodal_asset_paths": [str(graph_path), str(image_path)],
                }
            )
    if not entries:
        raise RuntimeError("No multimodal contexts available for problem_generating stage.")
    return entries


def _encode_image_as_data_url(path_str: str) -> Optional[str]:
    image_path = Path(path_str)
    if not image_path.exists() or image_path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
        return None
    mime_type = mimetypes.guess_type(str(image_path))[0] or "image/png"
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def build_problem_generating_prompt(
    prompt_type: str,
    topic: str,
    audience: str,
    context: str,
    difficulty: str,
) -> str:
    base = (
        f"Stage: {PROBLEM_GENERATING_STAGE}\n"
        f"Topic: {topic}\n"
        f"Audience: {audience}\n"
        f"Difficulty: {difficulty}\n"
        f"Context: {context}\n\n"
        "Generate exactly ONE physics problem question (not inquiry prompt, not answer) about centripetal acceleration.\n"
        "The question must explicitly use information inferred from BOTH the graph and the image.\n"
        "Do not provide solution steps or final numerical answer."
    )
    if prompt_type == "few_shot":
        return (
            base
            + "\n\nQuality examples:\n"
            "- Easy: 'A fan blade of radius 0.12 m rotates at 8 rad/s. Calculate the centripetal acceleration.'\n"
            "- Intermediate: 'A wheel has r=0.20 m and angular speed increases from 6 to 9 rad/s. Compare centripetal acceleration at both speeds.'\n"
            "- Advanced: 'Given measurement uncertainty in radius (+/-0.01 m), estimate the possible range of centripetal acceleration.'"
        )
    if prompt_type == "cot":
        return base + "\n\nThink step by step internally about observed objects, variables, and difficulty alignment, then return only final question."
    if prompt_type == "draft_critique_revise":
        return base + "\n\nDraft one question, critique if it fits the requested difficulty and multimodal evidence, revise once internally, then return final question only."
    return base


def build_chat_messages(system_msg: str, user_prompt: str, image_paths: List[str]) -> List[Dict[str, Any]]:
    if not image_paths:
        return [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_prompt},
        ]

    user_content: List[Dict[str, Any]] = [{"type": "text", "text": user_prompt}]
    for asset_path in image_paths:
        data_url = _encode_image_as_data_url(asset_path)
        if not data_url:
            continue
        user_content.append(
            {
                "type": "image_url",
                "image_url": {"url": data_url},
            }
        )

    if len(user_content) == 1:
        return [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_prompt},
        ]

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_content},
    ]


def generate_inquiry_only(
    model: str,
    payload: Dict[str, Any],
    prompt_type: str = "zero_shot",
    ability_band: str = "neutral",
    is_targeted: bool = False,
    knowledge_mode: str = "none",
    kg_path: Optional[str] = None,
) -> str:
    """
    Generate inquiry saja (tanpa reference).
    Reference sudah di-generate sebelumnya dan akan digunakan untuk semua strategies.
    For few_shot: ability_band (low/medium/high/neutral) and is_targeted (P3) are applied.
    Returns: inquiry question string.
    """
    stage = payload["stage"]
    topic = payload.get("topic", TOPIC)
    audience = payload.get("audience", AUDIENCE)
    context = payload.get("context", "")
    difficulty = payload.get("difficulty", "intermediate")
    multimodal_asset_paths = payload.get("multimodal_asset_paths", [])
    
    # Build prompt berdasarkan strategy
    if stage == PROBLEM_GENERATING_STAGE:
        user_prompt = build_problem_generating_prompt(
            prompt_type=prompt_type,
            topic=topic,
            audience=audience,
            context=context,
            difficulty=str(difficulty),
        )
    else:
        if prompt_type == "zero_shot":
            user_prompt = build_zero_shot_prompt(stage, topic, audience, context)
        elif prompt_type == "few_shot":
            user_prompt = build_few_shot_prompt(
                stage, topic, context,
                ability_band=ability_band,
                is_targeted=is_targeted,
            )
        elif prompt_type == "cot":
            user_prompt = build_cot_prompt(stage, topic, audience, context)
        elif prompt_type == "draft_critique_revise":
            user_prompt = build_draft_critique_revise_prompt(stage, topic, audience, context)
        else:
            raise ValueError(f"Unknown prompt type: {prompt_type}")

    user_prompt = maybe_inject_knowledge_context(
        prompt=user_prompt,
        stage=stage,
        context=str(context),
        difficulty=str(difficulty) if difficulty is not None else None,
        multimodal_asset_paths=[str(p) for p in multimodal_asset_paths],
        knowledge_mode=knowledge_mode,
        kg_path=kg_path,
    )
    
    # System message dengan instruksi khusus untuk problem_finding
    if stage == "problem_finding":
        system_msg = (
            "You are an educational designer for physics. "
            "Return JSON only with key: inquiry. "
            "The 'inquiry' is the generated inquiry question for students. "
            "Do NOT include 'reference' in your response, only 'inquiry'. "
            "CRITICAL for problem_finding stage: "
            "The inquiry MUST ask students to FIND or IDENTIFY things in their environment. "
            "DO NOT ask 'why', 'how', 'explain', or 'what happens if'. "
            "DO NOT mention specific object names from the context (e.g., if context is 'Classroom with a ceiling fan', do NOT mention 'ceiling fan' in the inquiry). "
            "The question should be GENERAL and help students discover things around them, not mention specific objects from the context. "
            "DO use words like 'find', 'identify', 'items', 'things', 'something', 'look for'."
        )
    elif stage == PROBLEM_GENERATING_STAGE:
        system_msg = (
            "You are an educational designer for physics problem generation. "
            "Return JSON only with key: inquiry. "
            "The 'inquiry' value must be exactly one generated problem question. "
            "Do NOT include solution, explanation, or additional keys."
        )
    else:
        system_msg = (
            "You are an educational designer for physics. "
            "Return JSON only with key: inquiry. "
            "The 'inquiry' is the generated inquiry question for students. "
            "Do NOT include 'reference' in your response, only 'inquiry'."
        )
    
    # Retry loop
    llm_cfg = _resolve_llm_config_for_model(model)
    api_key = llm_cfg["api_key"]
    base_url = llm_cfg["base_url"]
    headers = _build_llm_headers(api_key=api_key, base_url=base_url)
    for attempt in range(4):
        try:
            response = requests.post(
                base_url,
                headers=headers,
                json={
                    "model": _api_model_id(model, base_url),
                    "messages": build_chat_messages(
                        system_msg=system_msg,
                        user_prompt=user_prompt,
                        image_paths=[str(p) for p in multimodal_asset_paths],
                    ),
                    "temperature": 0.2,
                },
                timeout=_llm_timeout(),
            )
            if response.status_code == 402:
                raise RuntimeError("402 Payment Required (e.g. OpenRouter quota). Use another model or provider.") from None
            if response.status_code == 500:
                time.sleep(10)  # Beri waktu Ollama pulih (OOM/overload)
                raise RuntimeError("500 Internal Server Error (Ollama overload/OOM?). Coba model lebih kecil atau tunggu.") from None
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            
            # Try to extract JSON
            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                # Try to extract JSON from text
                import re
                match = re.search(r"\{.*\}", content, re.DOTALL)
                if match:
                    data = json.loads(match.group(0))
                else:
                    raise RuntimeError(f"Invalid JSON from model: {content[:200]}")
            
            inquiry = data.get("inquiry", "")
            if not inquiry:
                raise ValueError("Generated inquiry is empty")
            
            return inquiry
        except Exception as exc:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
    
    raise RuntimeError("Failed to generate inquiry after retries")


def generate_items_comparison(
    model: str,
    output_path: str,
    count_per_stage: int,
    prompt_strategies: List[str] = ["zero_shot", "few_shot", "cot", "draft_critique_revise"],
    num_references: int = 3,
    knowledge_modes: Optional[List[str]] = None,
    kg_path: Optional[str] = None,
) -> None:
    """
    Generate items dengan semua prompt strategies untuk perbandingan.
    Reference di-generate sekali per stage (bukan per context).
    
    Args:
        num_references: Jumlah reference questions per stage (default: 3)
    """
    knowledge_modes = knowledge_modes or DEFAULT_KNOWLEDGE_MODES
    try:
        llm_cfg = _resolve_llm_config_for_model(model)
        provider_url = llm_cfg["base_url"]
    except RuntimeError:
        print("\n" + "=" * 60)
        print("ERROR: No API key set for this model!")
        print("=" * 60)
        print(f"Model: {model}")
        print("Untuk OpenAI (gpt-4o dll.): set OPENAI_API_KEY")
        print("Untuk Ollama/OpenRouter: set LLM_API_KEY+LLM_API_BASE_URL atau OPENROUTER_API_KEY")
        print("=" * 60 + "\n")
        raise RuntimeError("No API key set for model. Configure OPENAI_API_KEY and/or LLM_API_KEY (or OPENROUTER_API_KEY).")
    print(f"[INFO] LLM endpoint for {model}: {provider_url}")
    
    stages = list(CONTEXTS.keys()) + [PROBLEM_GENERATING_STAGE]
    langs = ["en"]  # Only English

    stage_context_map: Dict[str, List[Dict[str, Any]]] = {}
    for stage in stages:
        if stage == PROBLEM_GENERATING_STAGE:
            stage_context_map[stage] = build_problem_generating_contexts(count_per_stage)
        else:
            stage_context_map[stage] = [
                {
                    "context": c,
                    "difficulty": None,
                    "multimodal_asset_paths": [],
                }
                for c in CONTEXTS[stage][:count_per_stage]
            ]
    
    # Total items = reference generations + inquiry generations
    # Reference: sekali per stage (bukan per context)
    # Inquiry: sekali per (stage, context, strategy, knowledge mode)
    total_references = 0
    for stage in stages:
        if stage == PROBLEM_GENERATING_STAGE:
            total_references += len(langs) * len(PROBLEM_GENERATING_DIFFICULTIES)
        else:
            total_references += len(langs)

    total_inquiries = 0
    for stage in stages:
        total_inquiries += len(langs) * len(stage_context_map[stage]) * len(prompt_strategies) * len(knowledge_modes)
    total_items = total_references + total_inquiries
    current_item = 0
    
    # Dictionary untuk menyimpan reference per stage
    stage_references: Dict[str, List[str]] = {}
    
    with open(output_path, "w", encoding="utf-8") as f:
        idx = 1
        for stage in stages:
            contexts = stage_context_map[stage]
            
            # Generate reference SEKALI per stage (bukan per context)
            for lang in langs:
                if stage == PROBLEM_GENERATING_STAGE:
                    for difficulty in PROBLEM_GENERATING_DIFFICULTIES:
                        current_item += 1
                        print(
                            f"[{current_item}/{total_items}] Generating reference for {stage} {difficulty} {lang}...",
                            end=" ",
                            flush=True,
                        )
                        try:
                            reference = generate_reference_only(
                                model=model,
                                stage=stage,
                                topic=TOPIC,
                                audience=AUDIENCE,
                                num_references=num_references,
                                difficulty=difficulty,
                            )
                            stage_references[f"{stage}_{lang}_{difficulty}"] = reference
                            print("[OK]")
                        except Exception as exc:
                            print(f"[ERROR] {exc}")
                            print(f"[WARNING] Skipping references for {stage} {difficulty}")
                            stage_references[f"{stage}_{lang}_{difficulty}"] = []
                else:
                    current_item += 1
                    print(f"[{current_item}/{total_items}] Generating reference for {stage} {lang}...", end=" ", flush=True)
                    try:
                        reference = generate_reference_only(
                            model=model,
                            stage=stage,
                            topic=TOPIC,
                            audience=AUDIENCE,
                            num_references=num_references
                        )
                        stage_references[f"{stage}_{lang}"] = reference
                        print("[OK]")
                    except Exception as exc:
                        print(f"[ERROR] {exc}")
                        print(f"[WARNING] Skipping stage {stage} due to reference generation failure")
                        stage_references[f"{stage}_{lang}"] = []
                        continue
            
            # Generate inquiry untuk semua contexts dalam stage ini menggunakan reference yang sama
            for lang in langs:
                for i, context_entry in enumerate(contexts):
                    difficulty = context_entry.get("difficulty")
                    if stage == PROBLEM_GENERATING_STAGE:
                        reference = stage_references.get(f"{stage}_{lang}_{difficulty}", [])
                    else:
                        reference = stage_references.get(f"{stage}_{lang}", [])
                    # Generate inquiry dengan setiap prompt strategy menggunakan reference yang sama
                    for prompt_type in prompt_strategies:
                        item_group_id = f"{stage[:2].upper()}-{lang.upper()}-{idx:03d}-{prompt_type[:2]}"
                        for knowledge_mode in knowledge_modes:
                            current_item += 1
                            print(
                                f"[{current_item}/{total_items}] Generating {stage} {lang} {prompt_type} {knowledge_mode} context {i+1}...",
                                end=" ",
                                flush=True,
                            )

                            mode_suffix = "kg" if str(knowledge_mode).strip().lower() == "kg" else "base"
                            payload = {
                                "id": f"{item_group_id}-{mode_suffix}",
                                "item_group_id": item_group_id,
                                "lang": lang,
                                "stage": stage,
                                "context": context_entry["context"],
                                "topic": TOPIC,
                                "audience": AUDIENCE,
                                "prompt_type": prompt_type,
                                "knowledge_mode": knowledge_mode,
                                "difficulty": difficulty,
                                "multimodal_asset_paths": context_entry.get("multimodal_asset_paths", []),
                            }

                            try:
                                inquiry = generate_inquiry_only(
                                    model,
                                    payload,
                                    prompt_type,
                                    knowledge_mode=knowledge_mode,
                                    kg_path=kg_path,
                                )

                                if not inquiry:
                                    raise ValueError("Generated inquiry is empty")

                                item = {
                                    "id": payload["id"],
                                    "item_group_id": payload["item_group_id"],
                                    "lang": payload["lang"],
                                    "stage": payload["stage"],
                                    "inquiry": inquiry,
                                    "reference": reference,
                                    "context": payload["context"],
                                    "prompt_type": prompt_type,
                                    "knowledge_mode": knowledge_mode,
                                    "difficulty": payload.get("difficulty"),
                                    "multimodal_asset_paths": payload.get("multimodal_asset_paths", []),
                                }
                                f.write(json.dumps(item, ensure_ascii=True) + "\n")
                                f.flush()
                                print("[OK]")
                            except Exception as exc:
                                print(f"[ERROR] {exc}")
                                time.sleep(1)
                        idx += 1
    
    # Save to CSV
    save_jsonl_to_csv(output_path)


def save_jsonl_to_csv(jsonl_path: str) -> None:
    """
    Convert JSONL file to CSV format.
    """
    csv_path = jsonl_path.replace(".jsonl", ".csv")
    items = []
    
    # Read JSONL
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    items.append(json.loads(line))
    except FileNotFoundError:
        print(f"[WARNING] JSONL file not found: {jsonl_path}")
        return
    
    if not items:
        print(f"[WARNING] No items to save to CSV")
        return
    
    # Prepare CSV data
    csv_rows = []
    for item in items:
        # Convert reference list to string
        reference_str = ""
        if isinstance(item.get("reference"), list):
            reference_str = json.dumps(item.get("reference", []), ensure_ascii=False)
        else:
            reference_str = str(item.get("reference", ""))
        
        row = {
            "id": item.get("id", ""),
            "item_group_id": item.get("item_group_id", ""),
            "lang": item.get("lang", ""),
            "stage": item.get("stage", ""),
            "prompt_type": item.get("prompt_type", ""),
            "knowledge_mode": item.get("knowledge_mode", "none"),
            "inquiry": item.get("inquiry", ""),
            "reference": reference_str,
            "context": item.get("context", ""),
            "difficulty": item.get("difficulty", ""),
            "multimodal_asset_paths": json.dumps(item.get("multimodal_asset_paths", []), ensure_ascii=True),
        }
        csv_rows.append(row)
    
    # Write CSV with error handling
    fieldnames = [
        "id",
        "item_group_id",
        "lang",
        "stage",
        "prompt_type",
        "knowledge_mode",
        "difficulty",
        "inquiry",
        "reference",
        "context",
        "multimodal_asset_paths",
    ]
    try:
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"[INFO] CSV saved to {csv_path}")
    except PermissionError:
        print(f"[WARNING] Cannot write CSV file {csv_path} - file may be open in another program (Excel, etc.)")
        print(f"[WARNING] Please close the file and try again, or manually convert JSONL to CSV later")
    except Exception as exc:
        print(f"[WARNING] Error saving CSV: {exc}")


def compare_prompt_strategies(results_jsonl: str) -> Dict[str, Dict[str, float]]:
    """
    Bandingkan performa setiap prompt strategy.
    Jika knowledge_mode tersedia, split per kombinasi strategy + knowledge mode.
    """
    def mean(values: List[float]) -> float:
        if not values:
            return 0.0
        return sum(values) / len(values)

    def summarize(rows: List[Dict[str, Any]]) -> Dict[str, float]:
        metrics = {
            "bleu1": [],
            "bleu4": [],
            "rouge1": [],
            "rouge2": [],
            "rougeL": [],
            "bertscore_f1": [],
            "bertscore_precision": [],
            "bertscore_recall": [],
            "geval_overall": [],
        }
        for r in rows:
            for metric in metrics.keys():
                val = r.get(metric)
                if isinstance(val, (int, float)):
                    metrics[metric].append(float(val))
        return {
            "bleu1_mean": mean(metrics["bleu1"]),
            "bleu4_mean": mean(metrics["bleu4"]),
            "rouge1_mean": mean(metrics["rouge1"]),
            "rouge2_mean": mean(metrics["rouge2"]),
            "rougeL_mean": mean(metrics["rougeL"]),
            "bertscore_f1_mean": mean(metrics["bertscore_f1"]),
            "bertscore_precision_mean": mean(metrics["bertscore_precision"]),
            "bertscore_recall_mean": mean(metrics["bertscore_recall"]),
            "geval_overall_mean": mean(metrics["geval_overall"]),
            "count": len(rows),
        }

    strategies = ["zero_shot", "few_shot", "cot", "draft_critique_revise"]
    comparison = {}

    # Load results
    results = []
    with open(results_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))

    for strategy in strategies:
        strategy_results = [r for r in results if r.get("prompt_type") == strategy]
        if not strategy_results:
            continue

        knowledge_modes = sorted({str(r.get("knowledge_mode", "none")).strip().lower() or "none" for r in strategy_results})
        if knowledge_modes and (len(knowledge_modes) > 1 or knowledge_modes[0] != "none"):
            for knowledge_mode in knowledge_modes:
                mode_rows = [
                    r
                    for r in strategy_results
                    if str(r.get("knowledge_mode", "none")).strip().lower() == knowledge_mode
                ]
                if mode_rows:
                    comparison[f"{strategy}__{knowledge_mode}"] = summarize(mode_rows)
        else:
            comparison[strategy] = summarize(strategy_results)

    return comparison


def compare_knowledge_modes(
    results_jsonl: str,
    prompt_type_filter: Optional[str] = "few_shot",
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Compare with-KG vs without-KG per stage.
    Return: stage -> knowledge_mode -> metric summary.
    """
    def mean(values: List[float]) -> float:
        if not values:
            return 0.0
        return sum(values) / len(values)

    stage_summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    rows: List[Dict[str, Any]] = []
    with open(results_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if prompt_type_filter and str(obj.get("prompt_type", "")).strip().lower() != prompt_type_filter.lower():
                continue
            rows.append(obj)

    grouped: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for row in rows:
        stage = str(row.get("stage", "")).strip().lower()
        knowledge_mode = str(row.get("knowledge_mode", "none")).strip().lower() or "none"
        if not stage:
            continue
        grouped.setdefault(stage, {}).setdefault(knowledge_mode, []).append(row)

    for stage, mode_rows in grouped.items():
        stage_summary[stage] = {}
        for knowledge_mode, items in mode_rows.items():
            metrics = {
                "bleu1_mean": mean([float(r["bleu1"]) for r in items if isinstance(r.get("bleu1"), (int, float))]),
                "rougeL_mean": mean([float(r["rougeL"]) for r in items if isinstance(r.get("rougeL"), (int, float))]),
                "bertscore_f1_mean": mean([float(r["bertscore_f1"]) for r in items if isinstance(r.get("bertscore_f1"), (int, float))]),
                "geval_overall_mean": mean([float(r["geval_overall"]) for r in items if isinstance(r.get("geval_overall"), (int, float))]),
                "count": float(len(items)),
            }
            stage_summary[stage][knowledge_mode] = metrics

    return stage_summary


def build_knowledge_mode_model_summary_from_eval(
    results_by_model: Dict[str, Any],
    bleu_scale: str,
    prompt_type_filter: Optional[str] = "few_shot",
    stage_filter: Optional[str] = None,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Aggregate eval results by model and knowledge_mode.
    Return: model -> knowledge_mode -> summary metrics.
    """
    summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for model_name, payload in results_by_model.items():
        if not isinstance(payload, dict):
            continue
        eval_path = payload.get("evaluation_results_jsonl")
        if not isinstance(eval_path, str) or not os.path.exists(eval_path):
            continue

        grouped: Dict[str, List[Dict[str, Any]]] = {}
        with open(eval_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if stage_filter and str(obj.get("stage", "")).strip().lower() != stage_filter.lower():
                    continue
                if prompt_type_filter and str(obj.get("prompt_type", "")).strip().lower() != prompt_type_filter.lower():
                    continue
                knowledge_mode = str(obj.get("knowledge_mode", "none")).strip().lower() or "none"
                grouped.setdefault(knowledge_mode, []).append(obj)

        if not grouped:
            continue

        summary[model_name] = {}
        for knowledge_mode, items in grouped.items():
            def mean_metric(metric: str) -> float:
                vals = [float(r[metric]) for r in items if isinstance(r.get(metric), (int, float))]
                return (sum(vals) / len(vals)) if vals else 0.0

            bleu_raw = mean_metric("bleu1")
            bleu_norm = bleu_raw / 100.0 if bleu_scale == "0-100" else bleu_raw
            rouge_mean = mean_metric("rougeL")
            bert_mean = mean_metric("bertscore_f1")
            geval_mean = mean_metric("geval_overall")
            geval_norm = geval_mean / 5.0
            composite = (bleu_norm + rouge_mean + bert_mean + geval_norm) / 4.0
            summary[model_name][knowledge_mode] = {
                "bleu1_mean": bleu_raw,
                "bleu1_norm": bleu_norm,
                "rougeL_mean": rouge_mean,
                "bertscore_f1_mean": bert_mean,
                "geval_overall_mean": geval_mean,
                "geval_overall_norm_mean": geval_norm,
                "composite_score": composite,
                "item_count": float(len(items)),
            }
    return summary


def build_knowledge_mode_stage_summary_from_eval(
    results_by_model: Dict[str, Any],
    prompt_type_filter: Optional[str] = "few_shot",
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Aggregate eval results across models by stage and knowledge_mode.
    Return: stage -> knowledge_mode -> summary metrics.
    """
    collected_rows: List[Dict[str, Any]] = []
    for payload in results_by_model.values():
        if not isinstance(payload, dict):
            continue
        eval_path = payload.get("evaluation_results_jsonl")
        if not isinstance(eval_path, str) or not os.path.exists(eval_path):
            continue
        with open(eval_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if prompt_type_filter and str(obj.get("prompt_type", "")).strip().lower() != prompt_type_filter.lower():
                    continue
                collected_rows.append(obj)

    def mean(values: List[float]) -> float:
        if not values:
            return 0.0
        return sum(values) / len(values)

    summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    grouped: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for row in collected_rows:
        stage = str(row.get("stage", "")).strip().lower()
        mode = str(row.get("knowledge_mode", "none")).strip().lower() or "none"
        if not stage:
            continue
        grouped.setdefault(stage, {}).setdefault(mode, []).append(row)

    for stage, stage_modes in grouped.items():
        summary[stage] = {}
        for mode, items in stage_modes.items():
            summary[stage][mode] = {
                "bleu1_mean": mean([float(r["bleu1"]) for r in items if isinstance(r.get("bleu1"), (int, float))]),
                "rougeL_mean": mean([float(r["rougeL"]) for r in items if isinstance(r.get("rougeL"), (int, float))]),
                "bertscore_f1_mean": mean([float(r["bertscore_f1"]) for r in items if isinstance(r.get("bertscore_f1"), (int, float))]),
                "geval_overall_mean": mean([float(r["geval_overall"]) for r in items if isinstance(r.get("geval_overall"), (int, float))]),
                "count": float(len(items)),
            }
    return summary


def write_knowledge_mode_comparison_plot(
    knowledge_summary: Dict[str, Dict[str, Dict[str, float]]],
    output_dir: str,
    filename: str = "all_models_kg_comparison.png",
    title: str = "With KG vs Without KG (Few-Shot)",
) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:
        raise RuntimeError("matplotlib is required for KG comparison plot. Install with: pip install matplotlib") from exc

    if not knowledge_summary:
        return
    os.makedirs(output_dir, exist_ok=True)

    models = [m for m in knowledge_summary.keys() if m not in EXCLUDED_MODELS_FOR_PLOTS]
    if not models:
        return

    modes_present = sorted(
        {
            mode
            for model_name in models
            for mode in knowledge_summary.get(model_name, {}).keys()
        }
    )
    if not modes_present:
        return

    x = list(range(len(models)))
    width = 0.8 / max(len(modes_present), 1)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    for i, mode in enumerate(modes_present):
        offset = (i - len(modes_present) / 2 + 0.5) * width
        comp_vals = [knowledge_summary.get(model, {}).get(mode, {}).get("composite_score", 0.0) for model in models]
        bert_vals = [knowledge_summary.get(model, {}).get(mode, {}).get("bertscore_f1_mean", 0.0) for model in models]
        geval_vals = [knowledge_summary.get(model, {}).get(mode, {}).get("geval_overall_norm_mean", 0.0) for model in models]
        label = "with KG" if mode == "kg" else "without KG"
        axes[0].bar([xi + offset for xi in x], comp_vals, width=width * 0.9, label=label)
        axes[1].plot(x, bert_vals, marker="o", linewidth=1.4, label=f"BERT F1 ({label})")
        axes[1].plot(x, geval_vals, marker="o", linewidth=1.4, linestyle="--", label=f"G-Eval ({label})")

    axes[0].set_title(title)
    axes[0].set_ylabel("Composite score (0-1)")
    axes[0].set_ylim(0, 1)
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[0].legend(loc="best")

    axes[1].set_title("Key normalized metrics by model")
    axes[1].set_ylabel("Score (0-1)")
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best", fontsize=8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([m if len(m) <= 22 else m[:19] + "..." for m in models], rotation=25, ha="right")

    plt.tight_layout()
    out_path = os.path.join(output_dir, filename)
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()


def write_knowledge_mode_stage_metric_grid_plot(
    stage_summary: Dict[str, Dict[str, Dict[str, float]]],
    output_dir: str,
    filename: str = "kg_comparison_plot.png",
    title: str = "With KG vs Without KG by Stage (Few-Shot)",
) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:
        raise RuntimeError("matplotlib is required for KG metric grid plot. Install with: pip install matplotlib") from exc

    if not stage_summary:
        return
    os.makedirs(output_dir, exist_ok=True)

    json_path = os.path.join(output_dir, "kg_comparison_data.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(stage_summary, f, indent=2, ensure_ascii=False)

    stage_order = ["problem_finding", "problem_exploring", "problem_generating"]
    stage_labels = ["Problem Finding", "Problem Exploring", "Problem Generating"]
    modes_present = sorted(
        {
            mode
            for stage in stage_summary.values()
            for mode in stage.keys()
        }
    )
    if not modes_present:
        return

    metrics = [
        ("bleu1_mean", "BLEU-1"),
        ("rougeL_mean", "ROUGE-L"),
        ("bertscore_f1_mean", "BERTScore F1"),
        ("geval_overall_mean", "G-Eval"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes_flat = axes.flatten().tolist()
    colors = {"none": "#4C78A8", "kg": "#F58518"}

    for idx, (metric_key, metric_label) in enumerate(metrics):
        ax = axes_flat[idx]
        x = list(range(len(stage_order)))
        width = 0.8 / max(len(modes_present), 1)
        for i, mode in enumerate(modes_present):
            offset = (i - len(modes_present) / 2 + 0.5) * width
            vals = [stage_summary.get(stage, {}).get(mode, {}).get(metric_key, 0.0) for stage in stage_order]
            if metric_key == "bleu1_mean":
                vals = [v / 100.0 for v in vals]
            if metric_key == "geval_overall_mean":
                vals = [v / 5.0 for v in vals]
            label = "with KG" if mode == "kg" else "without KG"
            bars = ax.bar(
                [xi + offset for xi in x],
                vals,
                width=width * 0.9,
                label=label,
                color=colors.get(mode, None),
            )
            for b in bars:
                h = b.get_height()
                if h > 0:
                    ax.text(
                        b.get_x() + b.get_width() / 2,
                        h + 0.01,
                        f"{h:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        clip_on=True,
                    )
        ax.set_title(metric_label)
        ax.set_xticks(x)
        ax.set_xticklabels(stage_labels, rotation=12, ha="right")
        ax.set_ylim(0, 1)
        ax.grid(True, axis="y", alpha=0.3)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.96),
            ncol=max(1, len(labels)),
            frameon=False,
        )
    fig.suptitle(title, fontsize=13, y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.88])
    out_path = os.path.join(output_dir, filename)
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()


def write_knowledge_mode_stage_plot(
    stage_summary: Dict[str, Dict[str, Dict[str, float]]],
    output_dir: str,
    filename: str = "kg_stage_comparison.png",
    title: str = "KG Ablation by Stage (Few-Shot)",
) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:
        raise RuntimeError("matplotlib is required for KG stage plot. Install with: pip install matplotlib") from exc

    if not stage_summary:
        return
    os.makedirs(output_dir, exist_ok=True)

    stage_order = ["problem_finding", "problem_exploring", "problem_generating"]
    stage_labels = ["Problem Finding", "Problem Exploring", "Problem Generating"]
    modes_present = sorted(
        {
            mode
            for stage in stage_summary.values()
            for mode in stage.keys()
        }
    )
    if not modes_present:
        return

    metrics = [
        ("bleu1_mean", "BLEU-1"),
        ("rougeL_mean", "ROUGE-L"),
        ("bertscore_f1_mean", "BERTScore F1"),
        ("geval_overall_mean", "G-Eval"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes_flat = axes.flatten().tolist()

    for idx, (metric_key, metric_label) in enumerate(metrics):
        ax = axes_flat[idx]
        x = list(range(len(stage_order)))
        width = 0.8 / max(len(modes_present), 1)
        for i, mode in enumerate(modes_present):
            offset = (i - len(modes_present) / 2 + 0.5) * width
            vals = [stage_summary.get(stage, {}).get(mode, {}).get(metric_key, 0.0) for stage in stage_order]
            if metric_key == "bleu1_mean":
                vals = [v / 100.0 for v in vals]
            if metric_key == "geval_overall_mean":
                vals = [v / 5.0 for v in vals]
            label = "with KG" if mode == "kg" else "without KG"
            ax.bar([xi + offset for xi in x], vals, width=width * 0.9, label=label)
        ax.set_title(metric_label)
        ax.set_xticks(x)
        ax.set_xticklabels(stage_labels, rotation=12, ha="right")
        ax.set_ylim(0, 1)
        ax.grid(True, axis="y", alpha=0.3)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=max(1, len(labels)))
    fig.suptitle(title, fontsize=13, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out_path = os.path.join(output_dir, filename)
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()


def _weighted_mean(values: List[float], weights: List[float]) -> float:
    if not values or not weights or len(values) != len(weights):
        return 0.0
    total_w = sum(weights)
    if total_w <= 0:
        return 0.0
    return sum(v * w for v, w in zip(values, weights)) / total_w


def build_model_level_summary(
    results_by_model: Dict[str, Any],
    bleu_scale: str,
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate per-strategy comparison into one score set per model.
    """
    summary: Dict[str, Dict[str, float]] = {}
    for model_name, payload in results_by_model.items():
        if not isinstance(payload, dict):
            continue
        comp = payload.get("comparison_data")
        if not isinstance(comp, dict) or not comp:
            continue

        bleu_vals: List[float] = []
        rouge_vals: List[float] = []
        bert_vals: List[float] = []
        geval_vals: List[float] = []
        weights: List[float] = []
        for _, v in comp.items():
            if not isinstance(v, dict):
                continue
            comp_key = str(_)
            if "__" in comp_key:
                _, mode = comp_key.rsplit("__", 1)
                if mode.strip().lower() != "none":
                    continue
            cnt = float(v.get("count", 0) or 0)
            if cnt <= 0:
                continue
            weights.append(cnt)
            bleu_vals.append(float(v.get("bleu1_mean", 0.0) or 0.0))
            rouge_vals.append(float(v.get("rougeL_mean", 0.0) or 0.0))
            bert_vals.append(float(v.get("bertscore_f1_mean", 0.0) or 0.0))
            geval_vals.append(float(v.get("geval_overall_mean", 0.0) or 0.0))

        if not weights:
            continue

        bleu_raw = _weighted_mean(bleu_vals, weights)
        bleu_norm = bleu_raw / 100.0 if bleu_scale == "0-100" else bleu_raw
        rouge_mean = _weighted_mean(rouge_vals, weights)
        bert_mean = _weighted_mean(bert_vals, weights)
        geval_mean = _weighted_mean(geval_vals, weights)
        geval_norm = geval_mean / 5.0

        # Composite score to rank models (normalized 0..1).
        composite = (bleu_norm + rouge_mean + bert_mean + geval_norm) / 4.0

        summary[model_name] = {
            "bleu1_mean": bleu_raw,
            "bleu1_norm": bleu_norm,
            "rougeL_mean": rouge_mean,
            "bertscore_f1_mean": bert_mean,
            "geval_overall_mean": geval_mean,
            "geval_overall_norm_mean": geval_norm,
            "composite_score": composite,
        }
    return summary


def build_model_level_summary_from_eval(
    results_by_model: Dict[str, Any],
    bleu_scale: str,
    stage_filter: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate model-level metrics directly from evaluation_results_all.jsonl.
    If stage_filter is set, only rows from that stage are included.
    """
    summary: Dict[str, Dict[str, float]] = {}
    for model_name, payload in results_by_model.items():
        if not isinstance(payload, dict):
            continue
        eval_path = payload.get("evaluation_results_jsonl")
        if not isinstance(eval_path, str) or not os.path.exists(eval_path):
            continue

        rows: List[Dict[str, Any]] = []
        with open(eval_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if str(obj.get("knowledge_mode", "none")).strip().lower() != "none":
                    continue
                if stage_filter and str(obj.get("stage", "")) != stage_filter:
                    continue
                rows.append(obj)

        if not rows:
            continue

        def mean_metric(metric: str) -> float:
            vals: List[float] = []
            for r in rows:
                v = r.get(metric)
                if isinstance(v, (int, float)):
                    vals.append(float(v))
            if not vals:
                return 0.0
            return sum(vals) / len(vals)

        bleu_raw = mean_metric("bleu1")
        bleu_norm = bleu_raw / 100.0 if bleu_scale == "0-100" else bleu_raw
        rouge_mean = mean_metric("rougeL")
        bert_mean = mean_metric("bertscore_f1")
        geval_mean = mean_metric("geval_overall")
        geval_norm = geval_mean / 5.0
        composite = (bleu_norm + rouge_mean + bert_mean + geval_norm) / 4.0

        summary[model_name] = {
            "bleu1_mean": bleu_raw,
            "bleu1_norm": bleu_norm,
            "rougeL_mean": rouge_mean,
            "bertscore_f1_mean": bert_mean,
            "geval_overall_mean": geval_mean,
            "geval_overall_norm_mean": geval_norm,
            "composite_score": composite,
            "item_count": float(len(rows)),
        }
    return summary


def write_all_models_comparison_plot(
    model_summary: Dict[str, Dict[str, float]],
    output_dir: str,
    bleu_scale: str,
    filename: str = "all_models_comparison.png",
    title: str = "All Models Comparison - Composite Ranking",
) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:
        raise RuntimeError("matplotlib is required for model comparison plot. Install with: pip install matplotlib") from exc

    if not model_summary:
        return
    os.makedirs(output_dir, exist_ok=True)

    models_sorted = [
        m
        for m in sorted(
            model_summary.keys(),
            key=lambda m: model_summary[m].get("composite_score", 0.0),
            reverse=True,
        )
        if m not in EXCLUDED_MODELS_FOR_PLOTS
    ]
    if not models_sorted:
        return

    x = list(range(len(models_sorted)))
    comp_vals = [model_summary[m]["composite_score"] for m in models_sorted]
    rouge_vals = [model_summary[m]["rougeL_mean"] for m in models_sorted]
    bert_vals = [model_summary[m]["bertscore_f1_mean"] for m in models_sorted]
    geval_vals = [model_summary[m]["geval_overall_norm_mean"] for m in models_sorted]
    bleu_vals = [model_summary[m]["bleu1_mean"] for m in models_sorted]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Panel 1: overall composite ranking
    axes[0].bar(x, comp_vals, color="#4C78A8")
    axes[0].set_title(title)
    axes[0].set_ylabel("Composite score (0-1)")
    axes[0].set_ylim(0, 1)
    axes[0].grid(True, axis="y", alpha=0.3)

    # Panel 2: key metrics line plot
    axes[1].plot(x, rouge_vals, marker="o", linewidth=1.5, label="ROUGE-L mean")
    axes[1].plot(x, bert_vals, marker="o", linewidth=1.5, label="BERTScore F1 mean")
    axes[1].plot(x, geval_vals, marker="o", linewidth=1.5, label="G-eval norm mean")
    axes[1].set_title("Normalized metrics by model")
    axes[1].set_ylabel("Score (0-1)")
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best")

    # Show model names on x-axis (shortened for readability)
    short_labels = [m if len(m) <= 22 else m[:19] + "..." for m in models_sorted]
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(short_labels, rotation=25, ha="right")

    # Add BLEU raw note in footer context
    bleu_range = "0-100" if bleu_scale == "0-100" else "0-1"
    fig.text(
        0.01,
        0.01,
        f"BLEU-1 mean scale: {bleu_range}. Top model: {models_sorted[0]} | BLEU={bleu_vals[0]:.3f}",
        fontsize=9,
        alpha=0.8,
    )
    plt.tight_layout()
    out_path = os.path.join(output_dir, filename)
    plt.savefig(out_path, dpi=160)
    plt.close()


_PROMPT_STRATEGY_ORDER = ["cot", "draft_critique_revise", "few_shot", "zero_shot"]
_PROMPT_STRATEGY_LABELS = ["CoT", "Draft-Critique-Revise", "Few-Shot", "Zero-Shot"]
_PROMPT_STRATEGY_COLORS = ["#4472C4", "#ED7D31", "#70AD47", "#C00000"]
_ALLOWED_PROMPT_TYPES_FOR_COMPARISON = frozenset(_PROMPT_STRATEGY_ORDER)
# Urutan sumbu X seperti referensi: Exploring → Finding → Generating
_COMPARISON_PLOT_STAGE_ORDER = ["problem_exploring", "problem_finding", "problem_generating"]
_COMPARISON_PLOT_STAGE_LABELS = ["Problem Exploring", "Problem Finding", "Problem Generating"]


def _load_metrics_by_stage_and_strategy(
    results_by_model: Dict[str, Any],
    bleu_scale: str,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Per model: eval JSONL, knowledge_mode==none, semua prompt strategy.
    Mean per (stage, strategy, metric), lalu rata-rata antar model (macro-average).
    Return: stage -> strategy -> metric_key -> nilai (0–1).
    """
    metric_keys = [
        "bleu1", "bleu4", "rouge1", "rouge2", "rougeL",
        "bertscore_precision", "bertscore_recall", "bertscore_f1",
        "geval_overall",
    ]
    stage_order = list(_COMPARISON_PLOT_STAGE_ORDER)

    def mean_metric(items: List[Dict[str, Any]], key: str) -> float:
        vals = [float(r[key]) for r in items if isinstance(r.get(key), (int, float))]
        return (sum(vals) / len(vals)) if vals else 0.0

    per_model: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}

    for model_name, payload in results_by_model.items():
        if not isinstance(payload, dict):
            continue
        if model_name in EXCLUDED_MODELS_FOR_PLOTS:
            continue
        eval_path = payload.get("evaluation_results_jsonl")
        if not eval_path or not os.path.exists(eval_path):
            continue
        rows: List[Dict[str, Any]] = []
        with open(eval_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                pt = str(obj.get("prompt_type", "")).strip().lower()
                if pt not in _ALLOWED_PROMPT_TYPES_FOR_COMPARISON:
                    continue
                if str(obj.get("knowledge_mode", "none")).strip().lower() != "none":
                    continue
                rows.append(obj)

        if not rows:
            continue

        buckets: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
            s: {st: [] for st in _PROMPT_STRATEGY_ORDER} for s in stage_order
        }
        for r in rows:
            s = str(r.get("stage", "")).strip().lower()
            pt = str(r.get("prompt_type", "")).strip().lower()
            if s in buckets and pt in buckets[s]:
                buckets[s][pt].append(r)

        per_model[model_name] = {}
        for stage in stage_order:
            per_model[model_name][stage] = {}
            for strategy in _PROMPT_STRATEGY_ORDER:
                items = buckets[stage][strategy]
                if not items:
                    continue
                metric_vals: Dict[str, float] = {}
                for key in metric_keys:
                    v = mean_metric(items, key)
                    if key in ("bleu1", "bleu4") and bleu_scale == "0-100":
                        v = v / 100.0
                    if key == "geval_overall":
                        v = v / 5.0
                    metric_vals[key] = v
                per_model[model_name][stage][strategy] = metric_vals

    if not per_model:
        return {}

    aggregated: Dict[str, Dict[str, Dict[str, float]]] = {}
    for stage in stage_order:
        aggregated[stage] = {}
        for strategy in _PROMPT_STRATEGY_ORDER:
            model_slices: List[Dict[str, float]] = []
            for _m, stages in per_model.items():
                st_map = stages.get(stage, {})
                if strategy in st_map:
                    model_slices.append(st_map[strategy])
            if not model_slices:
                continue
            aggregated[stage][strategy] = {}
            for key in metric_keys:
                vs = [ms[key] for ms in model_slices if key in ms]
                if vs:
                    aggregated[stage][strategy][key] = sum(vs) / len(vs)
    return aggregated


def write_models_few_shot_comparison_plot(
    results_by_model: Dict[str, Any],
    output_dir: str,
    bleu_scale: str = "0-1",
    filename: str = "comparison_plot.png",
) -> None:
    """
    Grid 2×3: sumbu X = stage (Exploring → Finding → Generating); per stage empat batang strategi.
    Selalu empat strategi + legenda (CoT, D-C-R, Few-Shot, Zero-Shot); tanpa data = batang 0.
    Hanya knowledge_mode none; nilai = rata-rata macro antar model.
    """
    try:
        import matplotlib.pyplot as plt  # type: ignore
        from matplotlib.patches import Patch  # type: ignore
    except Exception as exc:
        raise RuntimeError("matplotlib required for comparison plot. pip install matplotlib") from exc

    data = _load_metrics_by_stage_and_strategy(results_by_model, bleu_scale)
    if not data:
        return
    os.makedirs(output_dir, exist_ok=True)

    stage_order = list(_COMPARISON_PLOT_STAGE_ORDER)
    stage_labels = list(_COMPARISON_PLOT_STAGE_LABELS)
    metrics = [
        ("bleu1", "Bleu"),
        ("geval_overall", "G-Eval"),
        ("rougeL", "Rouge-L"),
        ("bertscore_precision", "Bert Precision"),
        ("bertscore_recall", "Bert Recall"),
        ("bertscore_f1", "Bert F1"),
    ]

    stages_used = [s for s in stage_order if s in data and data[s]]
    if not stages_used:
        return

    n_strategies = len(_PROMPT_STRATEGY_ORDER)
    n_metrics = len(metrics)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.4 * n_cols, 4.5 * n_rows))
    axes_flat = (axes.flatten().tolist() if hasattr(axes, "flatten") else [axes]) if n_rows * n_cols > 1 else [axes]

    x = list(range(len(stages_used)))
    width = 0.8 / max(n_strategies, 1)

    legend_handles = [
        Patch(
            facecolor=_PROMPT_STRATEGY_COLORS[i],
            edgecolor="#333333",
            label=_PROMPT_STRATEGY_LABELS[i],
        )
        for i in range(n_strategies)
    ]

    for idx, (metric_key, metric_label) in enumerate(metrics):
        ax = axes_flat[idx]
        for si, strategy in enumerate(_PROMPT_STRATEGY_ORDER):
            vals = []
            for stage in stages_used:
                v = 0.0
                if stage in data and strategy in data[stage]:
                    v = float(data[stage][strategy].get(metric_key, 0.0) or 0.0)
                vals.append(v)
            offset = (si - n_strategies / 2 + 0.5) * width
            color = _PROMPT_STRATEGY_COLORS[si % len(_PROMPT_STRATEGY_COLORS)]
            bars = ax.bar(
                [xi + offset for xi in x],
                vals,
                width=width * 0.9,
                color=color,
                edgecolor="#333333",
                linewidth=0.25,
            )
            for b in bars:
                h = b.get_height()
                if h > 0:
                    ax.text(
                        b.get_x() + b.get_width() / 2,
                        min(h + 0.02, 0.98),
                        f"{h:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=6,
                        clip_on=True,
                    )

        ax.set_xticks(x)
        ax.set_xticklabels(
            [stage_labels[stage_order.index(s)] for s in stages_used],
            rotation=12,
            ha="right",
            fontsize=9,
        )
        ax.set_xlabel("Stage")
        ax.set_ylabel(metric_label)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.yaxis.grid(True, linestyle="--", which="major", color="gray", alpha=0.45)
        ax.set_axisbelow(True)

    for j in range(n_metrics, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Evaluation Metrics Comparison: Prompt Strategies vs Stages", fontsize=13, y=0.99)
    plt.tight_layout(rect=[0, 0.11, 1, 0.93])
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=4,
        fontsize=8,
        frameon=True,
        fancybox=False,
        edgecolor="#cccccc",
    )
    out_path = os.path.join(output_dir, filename)
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()


def _pick_prompt_strategy_summary(
    comparison: Dict[str, Dict[str, float]],
    strategy: str,
    knowledge_mode_preference: Optional[str],
) -> Optional[Dict[str, float]]:
    """Pilih entri agregasi untuk satu strategi: kunci polos atau strategy__mode."""
    if strategy in comparison:
        return comparison[strategy]
    if knowledge_mode_preference:
        key = f"{strategy}__{knowledge_mode_preference}"
        if key in comparison:
            return comparison[key]
    prefix = f"{strategy}__"
    candidates = sorted(k for k in comparison if k.startswith(prefix))
    if candidates:
        return comparison[candidates[0]]
    return None


def write_prompt_strategy_comparison_grid_plot(
    comparison: Dict[str, Dict[str, float]],
    output_dir: str,
    bleu_scale: str = "0-1",
    filename: str = "prompt_strategy_comparison.png",
    title: Optional[str] = None,
    knowledge_mode_preference: Optional[str] = "none",
    model_display_name: Optional[str] = None,
) -> None:
    """
    Grid 2×3: satu batang per strategi prompt (tanpa membandingkan mode KG).
    Jika comparison memakai kunci strategy__mode, gunakan knowledge_mode_preference
    (default \"none\") agar hanya baris tanpa KG yang diplot.
    """
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "matplotlib is required for prompt strategy comparison plot. pip install matplotlib"
        ) from exc

    if not comparison:
        return
    os.makedirs(output_dir, exist_ok=True)

    strategy_display_order = ["cot", "draft_critique_revise", "few_shot", "zero_shot"]
    summaries: List[Optional[Dict[str, float]]] = [
        _pick_prompt_strategy_summary(comparison, s, knowledge_mode_preference) for s in strategy_display_order
    ]
    if not any(summaries):
        return

    def norm_bleu(v: float) -> float:
        return (v / 100.0) if bleu_scale == "0-100" else float(v)

    def norm_geval(v: float) -> float:
        return float(v) / 5.0

    metric_specs: List[tuple] = [
        ("bleu4_mean", "BLEU", "bleu"),
        ("rougeL_mean", "ROUGE-L", "identity"),
        ("bertscore_f1_mean", "BERT F1", "identity"),
        ("bertscore_recall_mean", "BERT RECALL", "identity"),
        ("bertscore_precision_mean", "BERT PRECISION", "identity"),
        ("geval_overall_mean", "G-EVAL", "geval"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=False)
    axes_flat = axes.flatten().tolist()
    title_bbox = dict(
        boxstyle="round,pad=0.35",
        facecolor="#ffe4cc",
        edgecolor="#888888",
        linewidth=0.8,
    )
    x = list(range(len(strategy_display_order)))
    bar_color = "#4472C4"

    for idx, (metric_key, metric_label, norm_kind) in enumerate(metric_specs):
        ax = axes_flat[idx]
        vals: List[float] = []
        for sm in summaries:
            if not sm:
                vals.append(0.0)
                continue
            raw = float(sm.get(metric_key, 0.0) or 0.0)
            if norm_kind == "bleu":
                vals.append(norm_bleu(raw))
            elif norm_kind == "geval":
                vals.append(norm_geval(raw))
            else:
                vals.append(raw)
        ax.bar(x, vals, width=0.55, color=bar_color, edgecolor="#333333", linewidth=0.3)
        ax.set_title(metric_label, bbox=title_bbox, pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(strategy_display_order, rotation=20, ha="right")
        ax.set_ylim(0, 1)
        ax.yaxis.grid(True, linestyle="-", which="major", color="lightgray", alpha=0.7)
        ax.set_axisbelow(True)

    st = title or (
        f"Prompt strategy comparison — {model_display_name}"
        if model_display_name
        else "Prompt strategy comparison"
    )
    fig.suptitle(st, fontsize=13, y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = os.path.join(output_dir, filename)
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()


_FLAT_EVAL_RESULTS_KEY = "flat_eval"


def _load_results_for_plots_only(output_dir: str, eval_plots_dir: str) -> Optional[Dict[str, Any]]:
    """
    Build results_by_model for plots-only mode. Prefer scanning output_dir so that
    every model with evaluation_results_all.jsonl is included (gpt-4o + qwen etc.).
    Juga mendukung satu file: output_dir/evaluation_results_all.jsonl (tanpa subfolder model).
    If scan is empty, fall back to model_comparison_summary.json.
    """
    results_by_model: Dict[str, Any] = {}
    if os.path.isdir(output_dir):
        for name in os.listdir(output_dir):
            subdir = os.path.join(output_dir, name)
            if not os.path.isdir(subdir):
                continue
            jsonl_path = os.path.join(subdir, "evaluation_results_all.jsonl")
            if os.path.isfile(jsonl_path):
                results_by_model[name] = {"evaluation_results_jsonl": jsonl_path}

    if not results_by_model:
        flat_jsonl = os.path.join(output_dir, "evaluation_results_all.jsonl")
        if os.path.isfile(flat_jsonl):
            results_by_model[_FLAT_EVAL_RESULTS_KEY] = {
                "evaluation_results_jsonl": os.path.abspath(flat_jsonl),
                "display_name": "evaluation_results_all (flat)",
            }

    if results_by_model:
        summary_path = os.path.join(output_dir, "model_comparison_summary.json")
        if os.path.exists(summary_path):
            try:
                with open(summary_path, "r", encoding="utf-8") as f:
                    summary = json.load(f)
                if isinstance(summary, dict):
                    for model_key, payload in summary.items():
                        if not isinstance(payload, dict):
                            continue
                        slug = _sanitize_model_name(model_key)
                        if slug in results_by_model:
                            results_by_model[slug] = {**results_by_model[slug], **payload, "display_name": model_key}
                        elif payload.get("evaluation_results_jsonl") and os.path.exists(str(payload.get("evaluation_results_jsonl", ""))):
                            results_by_model[slug] = {**payload, "display_name": model_key}
            except (json.JSONDecodeError, IOError):
                pass
        return results_by_model

    summary_path = os.path.join(output_dir, "model_comparison_summary.json")
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and data:
                return data
        except (json.JSONDecodeError, IOError):
            pass
    return None


def run_plots_only(
    output_dir: str,
    eval_plots_dir: str,
    bleu_scale: str,
) -> Dict[str, Any]:
    """
    Regenerate only comparison plots from existing evaluation results.
    Uses model_comparison_summary.json if present, else scans output_dir for
    <model>/evaluation_results_all.jsonl, atau satu file output_dir/evaluation_results_all.jsonl.
    """
    os.makedirs(eval_plots_dir, exist_ok=True)
    results_by_model = _load_results_for_plots_only(output_dir, eval_plots_dir)
    if not results_by_model:
        return {
            "error": "No evaluation results found. Put evaluation_results_all.jsonl in output_dir or in output_dir/<model>/, then retry.",
        }

    summary_file = os.path.join(output_dir, "model_comparison_summary.json")
    model_level_summary_file = os.path.join(output_dir, "model_level_summary.json")
    model_level_pg_summary_file = os.path.join(output_dir, "model_level_summary_problem_generating.json")
    knowledge_mode_summary_file = os.path.join(output_dir, "knowledge_mode_summary.json")

    has_comparison_data = any(
        isinstance(p.get("comparison_data"), dict) and p.get("comparison_data")
        for p in results_by_model.values()
        if isinstance(p, dict)
    )
    if has_comparison_data:
        model_level_summary = build_model_level_summary(results_by_model, bleu_scale=bleu_scale)
    else:
        model_level_summary = build_model_level_summary_from_eval(
            results_by_model=results_by_model,
            bleu_scale=bleu_scale,
            stage_filter=None,
        )

    with open(model_level_summary_file, "w", encoding="utf-8") as f:
        json.dump(model_level_summary, f, indent=2, ensure_ascii=True)
    write_all_models_comparison_plot(
        model_summary=model_level_summary,
        output_dir=eval_plots_dir,
        bleu_scale=bleu_scale,
    )

    model_level_pg_summary = build_model_level_summary_from_eval(
        results_by_model=results_by_model,
        bleu_scale=bleu_scale,
        stage_filter=PROBLEM_GENERATING_STAGE,
    )
    with open(model_level_pg_summary_file, "w", encoding="utf-8") as f:
        json.dump(model_level_pg_summary, f, indent=2, ensure_ascii=True)
    write_all_models_comparison_plot(
        model_summary=model_level_pg_summary,
        output_dir=eval_plots_dir,
        bleu_scale=bleu_scale,
        filename="all_models_problem_generating_comparison.png",
        title="Problem Generating Only - Model Composite Ranking",
    )

    write_models_few_shot_comparison_plot(
        results_by_model=results_by_model,
        output_dir=eval_plots_dir,
        bleu_scale=bleu_scale,
        filename="comparison_plot.png",
    )

    knowledge_mode_summary = build_knowledge_mode_model_summary_from_eval(
        results_by_model=results_by_model,
        bleu_scale=bleu_scale,
        prompt_type_filter="few_shot",
        stage_filter=None,
    )
    with open(knowledge_mode_summary_file, "w", encoding="utf-8") as f:
        json.dump(knowledge_mode_summary, f, indent=2, ensure_ascii=True)
    write_knowledge_mode_comparison_plot(
        knowledge_summary=knowledge_mode_summary,
        output_dir=eval_plots_dir,
    )
    write_knowledge_mode_stage_metric_grid_plot(
        stage_summary=build_knowledge_mode_stage_summary_from_eval(
            results_by_model=results_by_model,
            prompt_type_filter="few_shot",
        ),
        output_dir=eval_plots_dir,
        filename="kg_comparison_plot.png",
    )
    plot_knowledge_graph_nodes(eval_plots_dir, filename="kg_graph_nodes.png")

    for model_name, payload in results_by_model.items():
        if not isinstance(payload, dict):
            continue
        eval_results_jsonl = payload.get("evaluation_results_jsonl")
        if not isinstance(eval_results_jsonl, str) or not os.path.exists(eval_results_jsonl):
            continue
        model_slug = _sanitize_model_name(str(results_by_model.get(model_name, {}).get("display_name") or model_name))
        prompt_strategy_comparison = compare_prompt_strategies(eval_results_jsonl)
        write_prompt_strategy_comparison_grid_plot(
            comparison=prompt_strategy_comparison,
            output_dir=os.path.join(eval_plots_dir, model_slug),
            bleu_scale=bleu_scale,
            filename="prompt_strategy_comparison.png",
            title=f"Prompt strategy comparison — {model_name}",
            knowledge_mode_preference="none",
            model_display_name=model_name,
        )
        model_stage_summary = compare_knowledge_modes(eval_results_jsonl, prompt_type_filter="few_shot")
        write_knowledge_mode_stage_plot(
            stage_summary=model_stage_summary,
            output_dir=os.path.join(eval_plots_dir, model_slug),
            filename="kg_stage_comparison.png",
            title=f"KG Ablation by Stage - {model_name}",
        )
        write_knowledge_mode_stage_metric_grid_plot(
            stage_summary=model_stage_summary,
            output_dir=os.path.join(eval_plots_dir, model_slug),
            filename="kg_comparison_plot.png",
            title=f"With KG vs Without KG - {model_name}",
        )

    return {
        "summary_file": summary_file,
        "model_level_summary_file": model_level_summary_file,
        "model_level_summary_problem_generating_file": model_level_pg_summary_file,
        "knowledge_mode_summary_file": knowledge_mode_summary_file,
        "models": results_by_model,
    }


def run_eval(
    eval_script: str,
    input_path: str,
    bleu_scale: str,
    plots_dir: str,
    output_dir: str,
    geval_model: str,
) -> int:
    """
    Run evaluation script untuk evaluasi inquiry quality.
    """
    eval_results_csv = os.path.join(output_dir, "evaluation_results_all.csv")
    eval_results_jsonl = os.path.join(output_dir, "evaluation_results_all.jsonl")
    
    cmd = [
        sys.executable,
        eval_script,
        "--input",
        input_path,
        "--output-jsonl",
        eval_results_jsonl,
        "--output-csv",
        eval_results_csv,
        "--bleu-scale",
        bleu_scale,
        "--plots-dir",
        plots_dir,
        "--geval-model",
        geval_model,
    ]
    return subprocess.call(cmd)


def complete_workflow(
    models: List[str],
    count_per_stage: int = 10,
    prompt_strategies: List[str] = ["zero_shot", "few_shot", "cot", "draft_critique_revise"],
    output_dir: str = "data",
    eval_plots_dir: str = "eval_plots",
    bleu_scale: str = "0-100",
    skip_generation: bool = False,
    input_jsonl: Optional[str] = None,
    output_jsonl: Optional[str] = None,
    num_references: int = 3,
    use_same_model_for_geval: bool = True,
    fixed_geval_model: Optional[str] = None,
    knowledge_modes: Optional[List[str]] = None,
    kg_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Workflow lengkap: Generation → Evaluation → Comparison
    
    Args:
        num_references: Jumlah reference questions per (stage, context) (default: 3)
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(eval_plots_dir, exist_ok=True)
    
    if not models:
        raise ValueError("At least one model is required.")

    results_by_model: Dict[str, Any] = {}
    eval_script = os.path.join(os.path.dirname(__file__), "eval_inquiry_metrics.py")
    knowledge_modes = knowledge_modes or DEFAULT_KNOWLEDGE_MODES

    for model in models:
        model_slug = _sanitize_model_name(model)
        model_output_dir = os.path.join(output_dir, model_slug)
        model_plots_dir = os.path.join(eval_plots_dir, model_slug)
        os.makedirs(model_output_dir, exist_ok=True)
        os.makedirs(model_plots_dir, exist_ok=True)

        print("\n" + "=" * 60)
        print(f"Running workflow for model: {model}")
        print("=" * 60)

        if skip_generation:
            if not input_jsonl:
                raise ValueError("--input-jsonl required when --skip-generation is set")
            model_output_jsonl = input_jsonl
        else:
            # 1. GENERATION
            if output_jsonl and len(models) == 1:
                model_output_jsonl = output_jsonl
            else:
                model_output_jsonl = os.path.join(model_output_dir, "inquiry_samples_all_strategies.jsonl")
            out_dir = os.path.dirname(model_output_jsonl)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            print("Step 1: Generating inquiry samples with all prompt strategies...")
            generate_items_comparison(
                model=model,
                output_path=model_output_jsonl,
                count_per_stage=count_per_stage,
                prompt_strategies=prompt_strategies,
                num_references=num_references,
                knowledge_modes=knowledge_modes,
                kg_path=kg_path,
            )
            print(f"✓ Generated samples saved to {model_output_jsonl}\n")

        if fixed_geval_model:
            geval_model = fixed_geval_model
        elif use_same_model_for_geval:
            geval_model = model
        else:
            geval_model = os.getenv("G_EVAL_MODEL", "openai/gpt-4o")

        # 2. EVALUATION
        print("Step 2: Evaluating with multiple metrics...")
        eval_exit_code = run_eval(
            eval_script=eval_script,
            input_path=model_output_jsonl,
            bleu_scale=bleu_scale,
            plots_dir=model_plots_dir,
            output_dir=model_output_dir,
            geval_model=geval_model,
        )

        eval_results_csv = os.path.join(model_output_dir, "evaluation_results_all.csv")
        eval_results_jsonl = os.path.join(model_output_dir, "evaluation_results_all.jsonl")

        if eval_exit_code != 0:
            print(f"⚠ Evaluation exited with code {eval_exit_code} for model {model}")
            results_by_model[model] = {"error": f"Evaluation failed (exit code {eval_exit_code})"}
            continue

        # 3. COMPARISON
        print("Step 3: Comparing prompt strategies...")
        if not os.path.exists(eval_results_jsonl):
            results_by_model[model] = {
                "generated_samples": model_output_jsonl,
                "evaluation_results": eval_results_csv,
                "warning": "evaluation JSONL not found; comparison skipped",
            }
            continue

        comparison = compare_prompt_strategies(eval_results_jsonl)
        comparison_file = os.path.join(model_output_dir, "prompt_strategy_comparison.json")
        with open(comparison_file, "w", encoding="utf-8") as f:
            json.dump(comparison, f, indent=2, ensure_ascii=True)

        write_prompt_strategy_comparison_grid_plot(
            comparison=comparison,
            output_dir=model_plots_dir,
            bleu_scale=bleu_scale,
            filename="prompt_strategy_comparison.png",
            title=f"Prompt strategy comparison — {model}",
            knowledge_mode_preference="none",
            model_display_name=model,
        )

        knowledge_mode_comparison = compare_knowledge_modes(eval_results_jsonl, prompt_type_filter="few_shot")
        knowledge_mode_comparison_file = os.path.join(model_output_dir, "knowledge_mode_comparison.json")
        with open(knowledge_mode_comparison_file, "w", encoding="utf-8") as f:
            json.dump(knowledge_mode_comparison, f, indent=2, ensure_ascii=True)
        write_knowledge_mode_stage_plot(
            stage_summary=knowledge_mode_comparison,
            output_dir=model_plots_dir,
            filename="kg_stage_comparison.png",
            title=f"KG Ablation by Stage - {model}",
        )
        write_knowledge_mode_stage_metric_grid_plot(
            stage_summary=knowledge_mode_comparison,
            output_dir=model_plots_dir,
            filename="kg_comparison_plot.png",
            title=f"With KG vs Without KG - {model}",
        )

        results_by_model[model] = {
            "generated_samples": model_output_jsonl,
            "evaluation_results": eval_results_csv,
            "evaluation_results_jsonl": eval_results_jsonl,
            "comparison": comparison_file,
            "comparison_data": comparison,
            "knowledge_mode_comparison": knowledge_mode_comparison_file,
            "knowledge_mode_comparison_data": knowledge_mode_comparison,
            "geval_model": geval_model,
        }

    summary_file = os.path.join(output_dir, "model_comparison_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(results_by_model, f, indent=2, ensure_ascii=True)

    # Build one-plot overview for all models.
    model_level_summary = build_model_level_summary(results_by_model, bleu_scale=bleu_scale)
    model_level_summary_file = os.path.join(output_dir, "model_level_summary.json")
    with open(model_level_summary_file, "w", encoding="utf-8") as f:
        json.dump(model_level_summary, f, indent=2, ensure_ascii=True)
    write_all_models_comparison_plot(
        model_summary=model_level_summary,
        output_dir=eval_plots_dir,
        bleu_scale=bleu_scale,
    )

    knowledge_mode_summary = build_knowledge_mode_model_summary_from_eval(
        results_by_model=results_by_model,
        bleu_scale=bleu_scale,
        prompt_type_filter="few_shot",
        stage_filter=None,
    )
    knowledge_mode_summary_file = os.path.join(output_dir, "knowledge_mode_summary.json")
    with open(knowledge_mode_summary_file, "w", encoding="utf-8") as f:
        json.dump(knowledge_mode_summary, f, indent=2, ensure_ascii=True)
    write_knowledge_mode_comparison_plot(
        knowledge_summary=knowledge_mode_summary,
        output_dir=eval_plots_dir,
    )
    write_knowledge_mode_stage_metric_grid_plot(
        stage_summary=build_knowledge_mode_stage_summary_from_eval(
            results_by_model=results_by_model,
            prompt_type_filter="few_shot",
        ),
        output_dir=eval_plots_dir,
        filename="kg_comparison_plot.png",
    )
    plot_knowledge_graph_nodes(eval_plots_dir, kg_path=kg_path, filename="kg_graph_nodes.png")

    # Build multimodal-only (problem_generating) model comparison.
    model_level_pg_summary = build_model_level_summary_from_eval(
        results_by_model=results_by_model,
        bleu_scale=bleu_scale,
        stage_filter=PROBLEM_GENERATING_STAGE,
    )
    model_level_pg_summary_file = os.path.join(output_dir, "model_level_summary_problem_generating.json")
    with open(model_level_pg_summary_file, "w", encoding="utf-8") as f:
        json.dump(model_level_pg_summary, f, indent=2, ensure_ascii=True)
    write_all_models_comparison_plot(
        model_summary=model_level_pg_summary,
        output_dir=eval_plots_dir,
        bleu_scale=bleu_scale,
        filename="all_models_problem_generating_comparison.png",
        title="Problem Generating Only - Model Composite Ranking",
    )

    # Grid: Models vs Stages, few_shot only (seperti comparison_plot strategi tapi per model)
    write_models_few_shot_comparison_plot(
        results_by_model=results_by_model,
        output_dir=eval_plots_dir,
        bleu_scale=bleu_scale,
        filename="comparison_plot.png",
    )

    return {
        "summary_file": summary_file,
        "model_level_summary_file": model_level_summary_file,
        "model_level_summary_problem_generating_file": model_level_pg_summary_file,
        "knowledge_mode_summary_file": knowledge_mode_summary_file,
        "models": results_by_model,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Complete workflow: Generate with multiple prompt strategies and evaluate"
    )
    parser.add_argument("--model", default=None, help="Single model to use (backward-compatible).")
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="List of models for comparison. If omitted, uses default comparison list (includes Qwen and LLaVA).",
    )
    parser.add_argument("--count-per-stage", type=int, default=10, help="Items per stage per language")
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["zero_shot", "few_shot", "cot", "draft_critique_revise"],
        choices=["zero_shot", "few_shot", "cot", "draft_critique_revise"],
        help="Prompt strategies to use"
    )
    parser.add_argument("--output-dir", default="data", help="Output directory")
    parser.add_argument("--output-jsonl", default=None, help="Output JSONL file path (default: <output-dir>/inquiry_samples_all_strategies.jsonl)")
    parser.add_argument("--plots-dir", default="eval_plots", help="Plots directory")
    parser.add_argument("--bleu-scale", default="0-100", choices=["0-1", "0-100"], help="BLEU scale")
    parser.add_argument("--num-references", type=int, default=3, help="Number of reference questions per (stage, context). Default: 3. Recommended: 3-5 for optimal reliability.")
    parser.add_argument(
        "--knowledge-modes",
        nargs="+",
        default=DEFAULT_KNOWLEDGE_MODES,
        choices=["none", "kg"],
        help="Knowledge grounding variants to generate and compare. Default: none kg",
    )
    parser.add_argument(
        "--kg-path",
        default=str(KNOWLEDGE_GRAPH_PATH),
        help="Path to the knowledge graph JSON file used when knowledge mode is 'kg'.",
    )
    parser.add_argument("--skip-generation", action="store_true", help="Skip generation, only evaluate")
    parser.add_argument("--input-jsonl", help="Input JSONL for evaluation (if skipping generation)")
    parser.add_argument("--no-eval", action="store_true", help="Skip evaluation")
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Regenerate only comparison plots from existing evaluation results (no generation, no evaluation).",
    )
    parser.add_argument(
        "--fixed-geval-model",
        default=None,
        help="Use one fixed model as G-eval judge for all generation models.",
    )
    parser.add_argument(
        "--no-same-model-geval",
        action="store_true",
        help="Do not use each generation model as G-eval model. Uses G_EVAL_MODEL or fixed judge model.",
    )
    
    args = parser.parse_args()

    if args.plots_only:
        print("[INFO] Plots-only mode: regenerating comparison plots from existing results.")
        result = run_plots_only(
            output_dir=args.output_dir,
            eval_plots_dir=args.plots_dir,
            bleu_scale=args.bleu_scale,
        )
        if result.get("error"):
            print(f"ERROR: {result['error']}", file=sys.stderr)
            return 1
        print("✓ Plots saved to", args.plots_dir)
        return 0

    selected_models = args.models or ([args.model] if args.model else DEFAULT_MODEL_COMPARISON)
    print(f"[INFO] Models to run ({len(selected_models)}): {selected_models}")

    if args.no_eval:
        # Only generation
        for model in selected_models:
            model_slug = _sanitize_model_name(model)
            model_output_dir = os.path.join(args.output_dir, model_slug)
            os.makedirs(model_output_dir, exist_ok=True)
            if args.output_jsonl and len(selected_models) == 1:
                output_jsonl = args.output_jsonl
            else:
                output_jsonl = os.path.join(model_output_dir, "inquiry_samples_all_strategies.jsonl")
            out_dir = os.path.dirname(output_jsonl)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            print(f"Generating inquiry samples for model: {model}")
            generate_items_comparison(
                model=model,
                output_path=output_jsonl,
                count_per_stage=args.count_per_stage,
                prompt_strategies=args.strategies,
                num_references=args.num_references,
                knowledge_modes=args.knowledge_modes,
                kg_path=args.kg_path,
            )
            print(f"✓ Generated samples saved to {output_jsonl}")
        return 0
    
    # Complete workflow
    result = complete_workflow(
        models=selected_models,
        count_per_stage=args.count_per_stage,
        prompt_strategies=args.strategies,
        output_dir=args.output_dir,
        eval_plots_dir=args.plots_dir,
        bleu_scale=args.bleu_scale,
        skip_generation=args.skip_generation,
        input_jsonl=args.input_jsonl,
        output_jsonl=args.output_jsonl,
        num_references=args.num_references,
        use_same_model_for_geval=not args.no_same_model_geval,
        fixed_geval_model=args.fixed_geval_model,
        knowledge_modes=args.knowledge_modes,
        kg_path=args.kg_path,
    )
    
    if "error" in result:
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
