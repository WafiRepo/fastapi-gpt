"""
Inquiry API: generate inquiry berbasis stage.
- POST /inquiry/generate/problem-finding: generate inquiry untuk problem_finding.
- POST /inquiry/generate/problem-exploring-stage-1: generate inquiry untuk problem_exploring stage 1.
- POST /inquiry/generate/problem-exploring-stage-1/submit-response:
  submit jawaban stage 1 dan balas dengan inquiry/feedback berbasis few-shot testset CSV.
- POST /inquiry/generate/problem-exploring-stage-2: generate inquiry stage 2 (grafik/sensor) + opsional gambar base64.
- POST /inquiry/generate/problem-exploring-stage-2/submit-response: submit jawaban stage 2.
"""
import csv
import json
import logging
import random
import os
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

import requests
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from inquiry_db import (
    get_db_connection,
    get_latest_calculated_radius,
    get_latest_detected_object,
    get_student_profile,
    increment_questions_answered,
    ensure_inquiry_tables,
)

router_inquiry = APIRouter(prefix="/inquiry", tags=["inquiry"])


class GenerateInquiryRequest(BaseModel):
    user_id: Optional[str] = "default"
    topic: Optional[str] = "centripetal acceleration"
    location: Optional[str] = ""
    language: Optional[str] = "id"


class GenerateInquiryProblemExploringStage1Request(BaseModel):
    user_id: Optional[str] = "default"
    topic: Optional[str] = "centripetal acceleration"
    language: Optional[str] = "id"


class GenerateInquiryProblemExploringStage2Request(BaseModel):
    """Stage 2: analisis berbasis grafik/sensor; kirim screenshot grafik (base64) jika ada."""

    user_id: Optional[str] = "default"
    topic: Optional[str] = "centripetal acceleration"
    language: Optional[str] = "id"
    session_id: Optional[str] = None
    graph_context: Optional[str] = ""
    graph_image_base64: Optional[str] = None


class GenerateInquiryResponse(BaseModel):
    inquiry: str
    stage: Optional[str] = None
    ability_band: Optional[str] = None
    targeting_mode: Optional[str] = None
    detected_object: Optional[str] = None


class SubmitResponseRequest(BaseModel):
    user_id: str
    response_text: str
    inquiry_id: Optional[str] = None
    # optional context fields untuk stage-1 submit flow berbasis few-shot
    topic: Optional[str] = "centripetal acceleration"
    location: Optional[str] = ""
    session_id: Optional[str] = None
    language: Optional[str] = "id"
    graph_context: Optional[str] = None
    graph_image_base64: Optional[str] = None
    current_inquiry_text: Optional[str] = None


class SubmitResponseResponse(BaseModel):
    feedback: Optional[str] = None
    next_step: Optional[str] = None
    matched_answer_type: Optional[str] = None
    # Jalur "bingung / kurang paham" (matched_answer_type=uncertain): struktur terpisah
    step_by_step: Optional[List[str]] = None
    example: Optional[str] = None
    comprehension_check: Optional[str] = None


INQUIRY_GENERATION_MODEL = os.getenv("INQUIRY_GENERATION_MODEL", "gpt-4o-mini")

# Path ke folder prompts (data/prompts/few_shot.md)
PROMPTS_DIR = Path(__file__).resolve().parent / "data" / "prompts"
TESTSET_CSV_PATH = Path(__file__).resolve().parent / "docs" / "dataset problem finding and exploring - Augmented_Data_SMA_-_Centripetal_Concept.csv"
TESTSET_DIR = Path(__file__).resolve().parent / "data" / "testsets"
TESTSET_STAGE1_JSONL = TESTSET_DIR / "problem_exploring_stage1.jsonl"
TESTSET_STAGE2_JSONL = TESTSET_DIR / "problem_exploring_stage2.jsonl"
_STAGE1_CONVERSATION_MEMORY: Dict[str, List[Dict[str, str]]] = {}
_STAGE2_CONVERSATION_MEMORY: Dict[str, List[Dict[str, str]]] = {}
_MAX_MEMORY_TURNS = 6

# Lokasi Indonesia → Inggris untuk keperluan normalisasi konteks.
LOCATION_ID_TO_EN = {
    "kelas": "the classroom",
    "classroom": "the classroom",
    "playground": "the playground",
    "lapangan": "the field",
    "rumah": "home",
    "home": "home",
    "laboratorium": "the lab",
    "lab": "the lab",
    "perpustakaan": "the library",
    "library": "the library",
    "kantor": "the office",
    "office": "the office",
    "taman": "the park",
    "park": "the park",
}

def _location_to_english(location: str) -> str:
    """Ubah lokasi (mis. 'Kelas') ke Inggris ('the classroom') untuk normalisasi prompt."""
    if not location or not location.strip():
        return "your surroundings"
    key = location.strip().lower()
    return LOCATION_ID_TO_EN.get(key, location.strip())

# P2: Word count / complexity by ability band (untuk few_shot)
WORD_COUNT_RULES = {
    "low": (
        "The inquiry MUST be between 10 and 16 words (inclusive). "
        "Avoid subordinate clauses. Avoid two-step tasks."
    ),
    "medium": (
        "The inquiry MUST be between 16 and 20 words (inclusive). "
        "In problem_exploring stage only, you may include exactly one relation variable (e.g. 'between A and B')."
    ),
    "high": (
        "The inquiry MUST be between 20 and 25 words (inclusive). "
        "In problem_exploring stage only, you may include comparison or modification variables."
    ),
    "neutral": "Use medium language load. No strict word-count or difficulty scaling.",
}


def _get_targeting_instructions(is_targeted: bool, stage: str) -> str:
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


def _load_prompt_template(prompt_type: str) -> str:
    """Load prompt template dari data/prompts/<prompt_type>.md"""
    template_file = PROMPTS_DIR / f"{prompt_type}.md"
    if not template_file.exists():
        return ""
    with open(template_file, "r", encoding="utf-8") as f:
        content = f.read()
    lines = content.split("\n")
    if lines and lines[0].startswith("#"):
        start_idx = 1
        while start_idx < len(lines) and not lines[start_idx].strip():
            start_idx += 1
        content = "\n".join(lines[start_idx:])
    return content.strip()


def _normalize_language(lang: Optional[str], text_hint: Optional[str] = None) -> str:
    raw = (lang or "").strip().lower()
    if raw in ("id", "id-id", "indonesia", "indonesian", "bahasa", "bahasa indonesia"):
        return "id"
    if raw in ("en", "en-us", "en-gb", "english"):
        return "en"
    hint = (text_hint or "").strip().lower()
    id_markers = (
        "yang", "dan", "atau", "dengan", "bagaimana", "mengapa", "apa", "saat", "pada", "dari", "kecepatan",
    )
    if any(m in hint for m in id_markers):
        return "id"
    return "en"


def _language_label(lang: str) -> str:
    return "Bahasa Indonesia" if lang == "id" else "English"


def _build_few_shot_prompt(
    stage: str,
    topic: str,
    input_text: str,
    ability_band: str = "neutral",
    is_targeted: bool = False,
    language: str = "en",
) -> str:
    """Build few-shot prompt dari template data/prompts/few_shot.md. Kondisional problem_finding vs problem_exploring."""
    template = _load_prompt_template("few_shot")
    if not template:
        return ""
    word_count = WORD_COUNT_RULES.get((ability_band or "neutral").lower(), WORD_COUNT_RULES["neutral"])
    targeting = _get_targeting_instructions(is_targeted, stage or "problem_finding")
    prompt = template.replace("{WORD_COUNT_AND_COMPLEXITY}", word_count)
    prompt = prompt.replace("{TARGETING_INSTRUCTIONS}", targeting)
    prompt = prompt.replace("{STAGE}", stage or "problem_finding")
    prompt = prompt.replace("{TOPIC}", topic or "centripetal acceleration")
    prompt = prompt.replace("{INPUT_TEXT}", input_text or "Think about daily life objects that move in circles.")
    prompt = prompt.replace("{TARGET_LANGUAGE}", _language_label(language))
    return prompt


# Beberapa variasi konteks problem_finding agar pertanyaan LLM tidak selalu sama
_PROBLEM_FINDING_INPUT_VARIATIONS_EN = [
    "Look around {loc} for things that spin or rotate, or move in a circular path.",
    "Think about objects in {loc} that move in a circle or rotate.",
    "In {loc}, what things or activities involve circular motion?",
    "Consider your surroundings in {loc}: what rotates or follows a circular path?",
    "What in {loc} could you point to that demonstrates circular motion?",
]

_PROBLEM_FINDING_INPUT_VARIATIONS_ID = [
    "Lihat di sekitar {loc} untuk benda yang berputar atau bergerak pada lintasan melingkar.",
    "Pikirkan benda di {loc} yang bergerak melingkar atau berotasi.",
    "Di {loc}, benda atau aktivitas apa yang menunjukkan gerak melingkar?",
    "Perhatikan lingkunganmu di {loc}: apa yang berputar atau mengikuti lintasan melingkar?",
    "Apa di {loc} yang bisa kamu tunjukkan sebagai contoh gerak melingkar?",
]


def _varied_problem_finding_input(location_desc: str, user_id: str, language: str = "en") -> str:
    """Pilih variasi INPUT_TEXT secara acak agar pertanyaan LLM bervariasi tiap request."""
    if language == "id":
        if not location_desc or location_desc.lower() in ("sekitarmu", "lingkunganmu", "sekitar kamu"):
            loc = "lingkunganmu"
        else:
            loc = location_desc
        template = random.choice(_PROBLEM_FINDING_INPUT_VARIATIONS_ID)
    else:
        if not location_desc or location_desc.lower() in ("your surroundings", "around you"):
            loc = "your environment"
        else:
            loc = location_desc
        template = random.choice(_PROBLEM_FINDING_INPUT_VARIATIONS_EN)
    return template.format(loc=loc)


def generate_inquiry_text(
    stage: str,
    topic: str,
    location: str,
    user_id: str,
    ability_band: str = "neutral",
    is_targeted: bool = False,
    experiment_context: Optional[str] = None,
    language: str = "en",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> str:
    """Generate one inquiry question via LLM menggunakan few_shot.md. Kondisional problem_finding vs problem_exploring."""
    model_name = model or INQUIRY_GENERATION_MODEL
    stage = (stage or "problem_finding").strip().lower()
    if stage not in ("problem_finding", "problem_exploring"):
        stage = "problem_finding"
    location_desc = (location or "your surroundings").strip()
    if not location_desc or location_desc.lower() in ("lokasi tidak diisi", "none", "null"):
        location_desc = "sekitarmu" if language == "id" else "your surroundings"
    else:
        if language != "id":
            location_desc = _location_to_english(location_desc)
    key = (api_key or os.getenv("OPENAI_API_KEY", "")).strip()
    if not key:
        logger.warning("inquiry: OPENAI_API_KEY missing, returning 'none'")
        return "none"

    logger.info(
        "inquiry generate: stage=%s topic=%s location=%s user_id=%s ability_band=%s is_targeted=%s",
        stage, topic or "centripetal acceleration", location_desc, user_id, ability_band, is_targeted,
    )

    if stage == "problem_finding":
        input_text = _varied_problem_finding_input(location_desc, user_id, language=language)
    else:
        if experiment_context and experiment_context.strip():
            input_text = experiment_context.strip()
        elif location_desc and location_desc != "your surroundings":
            if language == "id":
                input_text = (
                    f"Objek terdeteksi adalah: {location_desc}. "
                    f"Buat SATU pertanyaan inkuiri yang secara eksplisit merujuk pada '{location_desc}' atau 'objek ini' — "
                    f"siswa akan mengeksplorasi geraknya dan percepatan sentripetal."
                )
            else:
                input_text = (
                    f"The detected object is: {location_desc}. "
                    f"Generate ONE inquiry question that explicitly refers to 'the {location_desc}' or 'this object' — "
                    f"the student will explore its motion and centripetal acceleration."
                )
        else:
            input_text = (
                "Siswa telah memilih objek. Mereka akan mengeksplorasi gerak objek dan percepatan sentripetal. "
                "Rujuk ke 'objek yang dipilih' atau 'objek ini' di pertanyaan."
                if language == "id"
                else "The student has chosen an object. They will explore its motion and centripetal acceleration. Refer to 'the chosen object' or 'this object' in the question."
            )

    user_content = _build_few_shot_prompt(
        stage=stage,
        topic=topic or "centripetal acceleration",
        input_text=input_text,
        ability_band=ability_band or "neutral",
        is_targeted=is_targeted,
        language=language,
    )
    if not user_content:
        logger.warning("inquiry: few_shot template empty, returning 'none' (check data/prompts/few_shot.md)")
        return "none"

    system_msg = (
        "You are an expert in physics education. Output ONLY valid JSON with key 'inquiry' (one-sentence question). "
        "Optionally include 'reference' array. No other text. "
        f"The inquiry MUST be in {_language_label(language)}. "
        "If the location is given in another language, preserve context naturally in the requested output language. "
        "For problem_exploring: if the input specifies a detected/chosen object (e.g. 'The detected object is: cup'), "
        "the inquiry MUST refer to that specific object (e.g. 'the cup', 'the chosen object') — do NOT use generic 'objects' or 'things'. "
        "For problem_finding: vary your phrasing (e.g. 'What can you find...', 'Identify something...', 'What things...', 'Can you spot...')."
    )

    for attempt in range(2):
        try:
            resp = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_content},
                    ],
                    "temperature": 0.7,
                },
                timeout=30,
            )
            resp.raise_for_status()
            content = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "")
            if not content:
                logger.warning("inquiry: LLM returned empty content, returning 'none'")
                return "none"
            content = content.strip()
            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                match = re.search(r"\{\s*\"inquiry\"\s*:\s*\"([^\"]+)\"\s*\}", content)
                if match:
                    inquiry = match.group(1).strip()
                else:
                    logger.warning("inquiry: LLM response invalid JSON and no inquiry match, returning 'none' (content=%s)", content[:200])
                    return "none"
            else:
                inquiry = (data.get("inquiry") or "").strip()
            if not inquiry:
                logger.warning("inquiry: inquiry field empty in JSON, returning 'none'")
                return "none"
            if stage == "problem_finding":
                inquiry = _enforce_problem_finding_location(inquiry, location_desc, language=language)
            if stage == "problem_exploring":
                inquiry = _enforce_problem_exploring_object(inquiry, location_desc, language=language)
            if stage == "problem_finding" and inquiry.count("?") >= 2:
                first_q = inquiry.split("?", 1)[0].strip()
                if first_q:
                    inquiry = first_q + "?"
                    logger.info("inquiry: problem_finding had two questions, using first only")
                else:
                    logger.warning("inquiry: problem_finding produced invalid first question, returning 'none'")
                    return "none"
            logger.info("inquiry generated: %s", inquiry[:100] + ("..." if len(inquiry) > 100 else ""))
            return inquiry
        except Exception as e:
            logger.warning("inquiry: LLM request failed (attempt %s), will return 'none' on last attempt: %s", attempt + 1, e, exc_info=True)
            if attempt == 1:
                return "none"
            continue
    logger.warning("inquiry: max retries exceeded, returning 'none'")
    return "none"


def _enforce_problem_finding_location(inquiry: str, location_desc: str, language: str = "en") -> str:
    """
    Untuk problem_finding, jika lokasi spesifik diberikan (mis. 'the classroom'),
    paksa inquiry menyebut lokasi agar konteks tidak hilang.
    """
    text = (inquiry or "").strip()
    loc = (location_desc or "").strip()
    if not text:
        return text
    if not loc or loc.lower() in ("your surroundings", "around you"):
        return text

    lower = text.lower()
    loc_lower = loc.lower()
    # Jika sudah menyebut lokasi, pertahankan.
    if loc_lower in lower:
        return text

    # Hindari phrasing yang salah konteks pada problem_finding.
    if "chosen object" in lower or "this object" in lower:
        if language == "id":
            return f"Apa yang bisa kamu temukan di {loc} yang bergerak pada lintasan melingkar?"
        return f"What can you find in {loc} that moves in a circular path?"

    # Rephrase pattern umum agar lokasi masuk.
    if "around you" in lower:
        return re.sub(r"around you", f"in {loc}", text, flags=re.IGNORECASE)
    if "your environment" in lower:
        return re.sub(r"your environment", loc, text, flags=re.IGNORECASE)
    if "your surroundings" in lower:
        return re.sub(r"your surroundings", loc, text, flags=re.IGNORECASE)

    # Fallback aman: bentuk pertanyaan generik dengan lokasi eksplisit.
    if language == "id":
        return f"Apa yang bisa kamu temukan di {loc} yang bergerak pada lintasan melingkar?"
    return f"What can you find in {loc} that moves in a circular path?"


def _enforce_problem_exploring_object(inquiry: str, object_desc: str, language: str = "en") -> str:
    """
    Untuk problem_exploring, paksa inquiry menyebut objek hasil deteksi DB
    agar tidak kembali ke phrasing generik seperti "di sekitar Anda".
    """
    text = (inquiry or "").strip()
    obj = (object_desc or "").strip()
    if not text:
        return text
    if not obj or obj.lower() in ("the chosen object", "objek yang dipilih"):
        return text

    lower = text.lower()
    obj_lower = obj.lower()
    if obj_lower in lower:
        return text

    # Jika model mengembalikan inquiry generik, rewrite agar menyebut objek eksplisit.
    generic_markers = (
        "di sekitar anda", "di sekitarmu", "sekitar anda", "sekitar kamu",
        "around you", "your surroundings", "your environment",
        "objek di sekitarmu", "objects around you",
    )
    if any(m in lower for m in generic_markers):
        if language == "id":
            return f"Bagaimana hubungan percepatan sentripetal dengan gerak {obj}?"
        return f"How does centripetal acceleration relate to the motion of the {obj}?"

    # Jika objek belum disebut, pertahankan inti kalimat LLM lalu injeksikan nama objek.
    # Tujuan: tetap tampilkan hasil LLM, tapi selalu menyebut object dari DB.
    if text.endswith("?"):
        core = text[:-1].strip()
        if language == "id":
            return f"{core} pada {obj}?"
        return f"{core} for the {obj}?"

    if language == "id":
        return f"{text} pada {obj}"
    return f"{text} for the {obj}"


def _normalize_csv_stage(value: str) -> str:
    s = (value or "").strip().lower()
    if s in ("stage 1", "stage1", "1"):
        return "stage_1"
    if s in ("stage 2", "stage2", "2"):
        return "stage_2"
    return ""


def _convert_testset_csv_to_jsonl() -> None:
    """
    Convert CSV testset ke dua file JSONL:
    - problem_exploring_stage1.jsonl
    - problem_exploring_stage2.jsonl
    """
    if not TESTSET_CSV_PATH.exists():
        logger.warning("testset CSV not found: %s", TESTSET_CSV_PATH)
        return
    TESTSET_DIR.mkdir(parents=True, exist_ok=True)
    stage1_rows: List[dict] = []
    stage2_rows: List[dict] = []
    with open(TESTSET_CSV_PATH, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stage = _normalize_csv_stage(str(row.get("stage 1", "") or row.get("stage", "")))
            if not stage:
                continue
            norm = {
                "topic": (row.get("problem_exploring") or "").strip(),
                "stage": stage,
                "input": (row.get("input") or "").strip(),
                "information_1": (row.get("information_1") or "").strip(),
                "system_q": (row.get("system_q") or row.get("user ask") or "").strip(),
                "user_answer_type": (row.get("user_answer_type") or "").strip().lower(),
                "user_a": (row.get("user_a") or "").strip(),
                "system_feedback": (row.get("system_feedback") or "").strip(),
                "scaffolding_or_next_step": (row.get("scaffolding_or_next_step") or "").strip(),
            }
            if not norm["system_q"]:
                continue
            if stage == "stage_1":
                stage1_rows.append(norm)
            elif stage == "stage_2":
                stage2_rows.append(norm)

    with open(TESTSET_STAGE1_JSONL, "w", encoding="utf-8") as f1:
        for item in stage1_rows:
            f1.write(json.dumps(item, ensure_ascii=False) + "\n")
    with open(TESTSET_STAGE2_JSONL, "w", encoding="utf-8") as f2:
        for item in stage2_rows:
            f2.write(json.dumps(item, ensure_ascii=False) + "\n")
    logger.info(
        "testset converted: stage1=%s rows, stage2=%s rows",
        len(stage1_rows),
        len(stage2_rows),
    )


def _load_stage1_testset_rows() -> List[dict]:
    if not TESTSET_STAGE1_JSONL.exists():
        _convert_testset_csv_to_jsonl()
    if not TESTSET_STAGE1_JSONL.exists():
        return []
    out: List[dict] = []
    with open(TESTSET_STAGE1_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _load_stage2_testset_rows() -> List[dict]:
    if not TESTSET_STAGE2_JSONL.exists():
        _convert_testset_csv_to_jsonl()
    if not TESTSET_STAGE2_JSONL.exists():
        return []
    out: List[dict] = []
    with open(TESTSET_STAGE2_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _normalize_graph_image_base64(raw: Optional[str]) -> Optional[str]:
    if not raw or not str(raw).strip():
        return None
    s = str(raw).strip()
    if s.startswith("data:"):
        idx = s.find("base64,")
        if idx >= 0:
            return s[idx + 7 :].strip()
    return s


def _build_stage2_fewshot_block(rows: List[dict], limit: int = 8) -> str:
    if not rows:
        return ""
    if len(rows) > limit:
        picked = random.sample(rows, limit)
    else:
        picked = list(rows)
    lines: List[str] = []
    for idx, ex in enumerate(picked, start=1):
        lines.append(f"Example {idx}:")
        lines.append(f"- Graph type / input: {ex.get('input', '')}")
        lines.append(f"- Information: {ex.get('information_1', '')}")
        lines.append(f"- Inquiry: {ex.get('system_q', '')}")
        lines.append(f"- Pedagogy note: {ex.get('user_answer_type', '')}")
    return "\n".join(lines)


def _chat_messages_with_optional_image(
    system_msg: str,
    text_prompt: str,
    image_b64: Optional[str],
) -> List[Dict[str, Any]]:
    b64 = _normalize_graph_image_base64(image_b64)
    if b64:
        data_url = f"data:image/jpeg;base64,{b64}"
        return [
            {"role": "system", "content": system_msg},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text_prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ]
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": text_prompt},
    ]


def _extract_picked_xy_from_graph_image(
    *,
    api_key: str,
    model_name: str,
    image_b64: Optional[str],
) -> Optional[Dict[str, str]]:
    """
    Coba ekstrak titik pick (x,y) yang terlihat pada screenshot grafik phyphox.
    Mengembalikan string numerik agar aman untuk ditampilkan kembali.
    """
    if not image_b64:
        return None
    system_msg = (
        "You read graph screenshots. Extract only the currently highlighted picked data point if visible. "
        "Output JSON only."
    )
    user_prompt = (
        "Find the picked point label in the graph screenshot (usually shown as x: ... and y: ...).\n"
        "Return ONLY valid JSON with keys:\n"
        "- picked_x: string number or empty string\n"
        "- picked_y: string number or empty string\n"
        "If not visible, return both as empty strings."
    )
    try:
        messages = _chat_messages_with_optional_image(system_msg, user_prompt, image_b64)
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model_name,
                "messages": messages,
                "temperature": 0.0,
            },
            timeout=35,
        )
        resp.raise_for_status()
        content = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "")
        data = _safe_json_loads_from_llm(content or "")
        if not data:
            return None
        x = str(data.get("picked_x") or "").strip()
        y = str(data.get("picked_y") or "").strip()
        if not x or not y:
            return None
        return {"picked_x": x, "picked_y": y}
    except Exception:
        logger.warning("stage2: failed to extract picked xy from image", exc_info=True)
        return None


def _safe_json_loads_from_llm(content: str) -> Optional[Dict[str, Any]]:
    """
    LLM kadang mengirim output JSON yang dibungkus code-fence atau ada teks tambahan.
    Fungsi ini berusaha mengekstrak 1 objek JSON dari string.
    """
    if not content:
        return None
    raw = str(content).strip()

    # Buang code-fence jika ada.
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\\s*", "", raw)
        raw = re.sub(r"\\s*```$", "", raw)

    # Coba parse langsung dulu.
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Ekstrak dari karakter pertama `{` sampai terakhir `}`.
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        candidate = raw[start : end + 1]
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    return None


def _llm_generate_stage2_inquiry(req: GenerateInquiryProblemExploringStage2Request) -> Optional[str]:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        return None
    model_name = os.getenv("INQUIRY_STAGE2_GENERATE_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
    rows = _load_stage2_testset_rows()
    fewshot = _build_stage2_fewshot_block(rows, limit=8)
    topic = (req.topic or "centripetal acceleration").strip()
    language = _normalize_language(req.language, text_hint=topic)
    graph_ctx = (req.graph_context or "").strip() or "student phyphox graph screenshot"
    detected_object = _resolve_detected_object_label(req.user_id) or "the chosen object"
    ctx_block = _resolve_stage1_context(
        SubmitResponseRequest(
            user_id=req.user_id or "default",
            response_text="",
            topic=topic,
            language=req.language or "id",
        )
    )
    user_prompt = (
        f"Stage: problem_exploring stage 2 (graph / sensor data interpretation).\n"
        f"Topic: {topic}\n"
        f"Target language: {_language_label(language)}\n"
        f"Graph context label (from app): {graph_ctx}\n"
        f"Detected object to mention: {detected_object}\n"
        f"{ctx_block}\n\n"
        f"Few-shot Stage 2 examples:\n{fewshot}\n\n"
        "You may use the attached graph image if provided.\n"
        "Generate exactly ONE inquiry question that helps the student interpret the graph "
        "(patterns, trends, comparisons). Do NOT give the final numeric answer or full formula as a statement.\n"
        "The inquiry MUST explicitly mention the detected object name.\n"
        'Return ONLY valid JSON with key: "inquiry" (string).'
    )
    system_msg = (
        "You are a physics tutor for centripetal acceleration (problem exploring stage 2: graphs). "
        "Output JSON only."
    )
    try:
        messages = _chat_messages_with_optional_image(system_msg, user_prompt, req.graph_image_base64)
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model_name,
                "messages": messages,
                "temperature": 0.35,
            },
            timeout=55,
        )
        resp.raise_for_status()
        content = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "")
        raw = (content or "").strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
        data = json.loads(raw)
        inquiry = str(data.get("inquiry") or "").strip()
        if not inquiry:
            return None
        inquiry = _enforce_problem_exploring_object(inquiry, detected_object, language=language)
        return inquiry if inquiry and inquiry.lower() != "none" else None
    except Exception:
        logger.warning("stage2 generate: LLM failed, returning None", exc_info=True)
        return None


def _get_stage2_session_key(req: SubmitResponseRequest) -> str:
    if req.session_id and req.session_id.strip():
        return f"stage2:{req.session_id.strip()}"
    return f"stage2:user:{(req.user_id or 'default').strip()}"


def _llm_stage2_feedback(
    req: SubmitResponseRequest,
    stage: str,
    rows: List[dict],
) -> Optional[Dict[str, str]]:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        return None
    model_name = os.getenv("INQUIRY_STAGE2_SUBMIT_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
    context = _resolve_stage1_context(req)
    session_key = _get_stage2_session_key(req)
    memory = _STAGE2_CONVERSATION_MEMORY.get(session_key, [])
    memory_block = "\n".join(
        [f"{m.get('role', 'user')}: {m.get('content', '')}" for m in memory[-_MAX_MEMORY_TURNS:]]
    ) or "(no previous turns)"
    fewshot = _build_stage2_fewshot_block(rows, limit=6)
    topic = (req.topic or "centripetal acceleration").strip()
    user_answer = (req.response_text or "").strip()
    language = _normalize_language(req.language, text_hint=f"{topic} {user_answer}")
    graph_ctx = (req.graph_context or "").strip() or "student phyphox graph"
    detected_object = _resolve_detected_object_label(req.user_id) or "the chosen object"
    radius_row = get_latest_calculated_radius((req.user_id or "default").strip() or "default")
    radius_value: Optional[float] = None
    try:
        if radius_row and radius_row.get("radius") is not None:
            radius_value = float(radius_row.get("radius"))
    except Exception:
        radius_value = None
    var_hint = _infer_stage2_graph_variables(graph_ctx, user_answer, topic, language)
    x_var = var_hint.get("x_var") or ("waktu (t)" if language == "id" else "time (t)")
    y_var = var_hint.get("y_var") or ("kecepatan sudut (ω)" if language == "id" else "angular velocity (ω)")
    picked_xy = _extract_picked_xy_from_graph_image(
        api_key=key,
        model_name=model_name,
        image_b64=req.graph_image_base64,
    )
    picked_x = (picked_xy or {}).get("picked_x")
    picked_y = (picked_xy or {}).get("picked_y")
    picked_line = (
        f"Detected picked point from image: x={picked_x}, y={picked_y}\n"
        if picked_x and picked_y
        else "Detected picked point from image: not found\n"
    )
    radius_line = (
        f"Known radius from DB: r={radius_value} cm\n"
        if radius_value is not None
        else "Known radius from DB: not found\n"
    )
    user_prompt = (
        f"Stage: {stage} (graph / sensor data)\n"
        f"Topic: {topic}\n"
        f"Target language: {_language_label(language)}\n"
        f"Graph context: {graph_ctx}\n"
        f"Detected graph variables: x-axis={x_var}, y-axis={y_var}\n"
        f"{picked_line}"
        f"{radius_line}"
        f"Detected object to mention: {detected_object}\n"
        f"{context}\n\n"
        f"Conversation memory:\n{memory_block}\n\n"
        f"Few-shot Stage 2 examples:\n{fewshot}\n\n"
        f"Current student answer:\n{user_answer}\n\n"
        "You may use the attached graph image if provided.\n"
        "Return ONLY valid JSON with keys: inquiry, feedback, next_step, matched_answer_type.\n"
        "matched_answer_type must be one of: correct, wrong, uncertain.\n"
        "The inquiry MUST explicitly mention the detected object name.\n"
        "IMPORTANT: inquiry, feedback, and next_step must be written in the target language. "
        "First read the ATTACHED IMAGE axis labels. If they disagree with Detected graph variables, trust the image. "
        "a_c (percepatan sentripetal) is ONLY valid if the y-axis shows acceleration/centripetal. "
        "If the graph is ω versus time t, do NOT discuss a_c as if it were plotted; you may optionally mention "
        "linking later via a_c = ω² r only when asking to compute r from a known a_c and ω from a data point. "
        "feedback and next_step MUST explicitly reference the detected object and the actual graph variables. "
        "Ground guidance on visible graph evidence from the attached image (points, trend, slope/delta), not generic advice. "
        "Stage-2 requirement: feedback must focus on graph-data analysis actions on the canvas "
        "(e.g., circle points, draw trend line, add arrow) and must not ask for long text explanation. "
        "Prioritize variable relationship or calculation from two picked (x,y) data points: "
        "e.g. slope Δy/Δx, or use known radius r from DB (in centimeters, cm) with picked point to compute a related variable. "
        "If a picked point is detected, next_step must explicitly use that x and y value in the question. "
        "If DB radius is available, next_step must include a concrete calculation question using r (in cm) and the picked point. "
        "Keep output concise: feedback should be 1-2 short action sentences; "
        "next_step should be one short guiding question for observation on canvas (must end with '?')."
    )
    system_msg = (
        "You are a physics tutor for circular motion / centripetal acceleration (problem exploring stage 2: graphs). "
        "Always ground feedback in the graph axes visible in the image. "
        "Use few-shot examples and conversation memory to generate feedback. "
        "Output JSON only."
    )
    try:
        messages = _chat_messages_with_optional_image(system_msg, user_prompt, req.graph_image_base64)
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model_name,
                "messages": messages,
                "temperature": 0.4,
            },
            timeout=55,
        )
        resp.raise_for_status()
        content = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "")
        if not content:
            return None
        data = _safe_json_loads_from_llm(content)
        if not data:
            return None
        inquiry = str(data.get("inquiry") or "").strip()
        feedback = str(data.get("feedback") or "").strip()
        next_step = str(data.get("next_step") or "").strip()
        matched = str(data.get("matched_answer_type") or "").strip().lower()
        if not inquiry:
            return None
        inquiry = _enforce_problem_exploring_object(inquiry, detected_object, language=language)
        if not inquiry or inquiry.lower() == "none":
            return None
        # Stage 2: jangan pakai pass-2 rewrite (sering menyuntikkan a_c/topik generik di luar sumbu grafik).
        # Jangan pakai aturan Stage 1 di Stage 2 agar tidak melebar ke topik lain.
        stage2_adjusted = _ensure_stage2_canvas_focused_feedback(
            feedback=feedback,
            next_step=next_step,
            matched=matched,
            language=language,
            detected_object=detected_object,
            x_var=x_var,
            y_var=y_var,
            picked_x=picked_x,
            picked_y=picked_y,
            radius_value=radius_value,
        )
        feedback = stage2_adjusted.get("feedback") or feedback
        next_step = stage2_adjusted.get("next_step") or next_step
        compact = _compact_stage2_feedback_and_next_step(feedback, next_step, language)
        feedback = compact.get("feedback") or feedback
        next_step = compact.get("next_step") or next_step
        mem = _STAGE2_CONVERSATION_MEMORY.setdefault(session_key, [])
        mem.append({"role": "user", "content": user_answer})
        mem.append(
            {
                "role": "assistant",
                "content": f"inquiry={inquiry} | feedback={feedback} | next_step={next_step}",
            }
        )
        if len(mem) > (_MAX_MEMORY_TURNS * 2):
            _STAGE2_CONVERSATION_MEMORY[session_key] = mem[-(_MAX_MEMORY_TURNS * 2) :]
        return {
            "inquiry": inquiry,
            "feedback": feedback,
            "next_step": next_step,
            "matched_answer_type": matched,
        }
    except Exception:
        logger.warning("stage2 submit: LLM feedback failed, returning None", exc_info=True)
        return None


def _get_stage1_session_key(req: SubmitResponseRequest) -> str:
    if req.session_id and req.session_id.strip():
        return req.session_id.strip()
    if req.inquiry_id and req.inquiry_id.strip():
        return req.inquiry_id.strip()
    return f"user:{(req.user_id or 'default').strip()}"


def _resolve_detected_object_label(user_id: Optional[str]) -> Optional[str]:
    uid = (user_id or "default").strip() or "default"
    detected = get_latest_detected_object(uid)
    if detected and detected.get("label"):
        label = str(detected.get("label") or "").strip().replace("_", " ")
        if label:
            return label
    return None


def _resolve_stage1_context(req: SubmitResponseRequest) -> str:
    label = _resolve_detected_object_label(req.user_id)
    if label:
        return f"Detected object from session: {label}"
    return "Detected object from session: the chosen object"


def _build_stage1_fewshot_block(rows: List[dict], limit: int = 6) -> str:
    if not rows:
        return ""
    # campur dan ambil contoh correct/wrong agar seimbang
    correct = [r for r in rows if str(r.get("user_answer_type", "")).lower() == "correct"]
    wrong = [r for r in rows if str(r.get("user_answer_type", "")).lower() == "wrong"]
    picked: List[dict] = []
    for i in range(max(1, limit // 2)):
        if i < len(correct):
            picked.append(correct[i])
        if i < len(wrong):
            picked.append(wrong[i])
        if len(picked) >= limit:
            break
    if len(picked) < limit:
        for r in rows:
            if r not in picked:
                picked.append(r)
            if len(picked) >= limit:
                break
    lines: List[str] = []
    for idx, ex in enumerate(picked, start=1):
        lines.append(f"Example {idx}:")
        lines.append(f"- Inquiry: {ex.get('system_q', '')}")
        lines.append(f"- Student answer: {ex.get('user_a', '')}")
        lines.append(f"- Answer type: {ex.get('user_answer_type', '')}")
        lines.append(f"- Feedback: {ex.get('system_feedback', '')}")
        lines.append(f"- Next step: {ex.get('scaffolding_or_next_step', '')}")
    return "\n".join(lines)


def _llm_rewrite_wrong_feedback_non_reveal(
    *,
    api_key: str,
    model_name: str,
    language: str,
    topic: str,
    student_answer: str,
    feedback: str,
    next_step: str,
) -> Dict[str, str]:
    """
    Pass-2 GPT self-check: rewrite feedback/next_step untuk jawaban wrong agar
    tetap Socratic dan tidak memberi jawaban final secara eksplisit.
    """
    checker_prompt = (
        f"Target language: {_language_label(language)}\n"
        f"Topic: {topic}\n"
        f"Student answer: {student_answer}\n\n"
        "Review the draft feedback below.\n"
        "If it reveals the final correct concept/value/formula directly, rewrite it to be Socratic (guiding questions only).\n"
        "If it is already non-revealing, keep it with minimal edits.\n"
        "Do NOT state the final correct answer explicitly.\n\n"
        "Keep it concise:\n"
        "- feedback: max 1 guiding question sentence.\n"
        "- next_step: 1 short action sentence, no question mark.\n\n"
        f"Draft feedback:\n{feedback}\n\n"
        f"Draft next_step:\n{next_step}\n\n"
        "Return ONLY valid JSON with keys: feedback, next_step."
    )
    system_msg = (
        "You are a physics pedagogy reviewer. "
        "Your job is to keep guidance non-revealing for incorrect student answers, while staying helpful and specific. "
        "Output JSON only."
    )
    try:
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": checker_prompt},
                ],
                "temperature": 0.2,
            },
            timeout=30,
        )
        resp.raise_for_status()
        content = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "")
        data = json.loads((content or "").strip())
        rewritten_feedback = str(data.get("feedback") or "").strip()
        rewritten_next = str(data.get("next_step") or "").strip()
        return {
            "feedback": rewritten_feedback or feedback,
            "next_step": rewritten_next or next_step,
        }
    except Exception:
        logger.warning("stage1 submit: non-reveal rewrite failed, using pass-1 output", exc_info=True)
        return {"feedback": feedback, "next_step": next_step}


def _compact_feedback_and_next_step(feedback: str, next_step: str, language: str) -> Dict[str, str]:
    fb = (feedback or "").strip()
    ns = (next_step or "").strip()

    # Keep only first question in feedback (or first sentence if no question).
    if "?" in fb:
        first_q = fb.split("?", 1)[0].strip()
        fb = (first_q + "?") if first_q else fb
    elif "." in fb:
        fb = fb.split(".", 1)[0].strip() + "."

    # next_step should be one short action sentence (no question mark).
    if "?" in ns:
        ns = ns.split("?", 1)[0].strip()
    if "." in ns:
        ns = ns.split(".", 1)[0].strip()
    if ns:
        if language == "id":
            if not ns.lower().startswith(("amati", "bandingkan", "jelaskan", "coba", "catat", "diskusikan")):
                ns = "Coba " + ns[:1].lower() + ns[1:] if len(ns) > 1 else "Coba lakukan pengamatan."
        else:
            if not ns.lower().startswith(("observe", "compare", "explain", "try", "note", "discuss")):
                ns = "Try " + ns[:1].lower() + ns[1:] if len(ns) > 1 else "Try observing the motion."
        if not ns.endswith("."):
            ns += "."

    return {"feedback": fb, "next_step": ns}


def _compact_stage2_feedback_and_next_step(feedback: str, next_step: str, language: str) -> Dict[str, str]:
    fb = (feedback or "").strip()
    ns = (next_step or "").strip()

    # Keep feedback concise (max first sentence/question).
    if "?" in fb:
        first_q = fb.split("?", 1)[0].strip()
        fb = (first_q + "?") if first_q else fb
    elif "." in fb:
        fb = fb.split(".", 1)[0].strip() + "."

    # Stage-2: next_step harus pertanyaan observasi di canvas.
    if ns:
        if "?" in ns:
            ns = ns.split("?", 1)[0].strip() + "?"
        else:
            ns = ns.rstrip(".")
            if language == "id":
                ns = f"Apa yang kamu amati di grafik dari langkah ini: {ns.lower()}?"
            else:
                ns = f"What do you observe in the graph from this step: {ns.lower()}?"

    return {"feedback": fb, "next_step": ns}


def _infer_stage2_graph_variables(graph_ctx: str, user_answer: str, topic: str, language: str) -> Dict[str, str]:
    """
    Estimasi nama variabel sumbu dari konteks teks (bukan asumsi default ω vs a_c).
    Banyak eksperimen phyphox menampilkan ω terhadap waktu t — a_c hanya muncul jika sumbu y memang percepatan.
    """
    # Jangan pakai field `topic` untuk deteksi sumbu: nilai default sering mengandung "acceleration" dan memicu has_ac palsu.
    text_signal = f"{graph_ctx or ''} {user_answer or ''}".lower()

    has_omega = any(
        k in text_signal for k in ("omega", "ω", "angular velocity", "kecepatan sudut", "kecepatan putar", "rad/s", "rpm")
    )
    has_time = any(
        k in text_signal
        for k in (
            "t (s)",
            "t(s)",
            "s)",
            "waktu",
            "detik",
            " time",
            "vs t",
            "versus t",
            "terhadap t",
            "sumbu x: t",
        )
    )
    has_ac = any(
        k in text_signal
        for k in (
            "a_c",
            "centripetal acceleration",
            "percepatan sentripetal",
            "percepatan (m/s",
            "m/s^2",
            "m/s²",
            "acceleration (m/s",
        )
    )
    has_v = any(k in text_signal for k in ("linear velocity", "kecepatan linear", "tangensial", "v "))
    has_r = any(k in text_signal for k in ("radius", "jari-jari", "r "))

    # Prioritas: ω vs t (rekaman kipas/roda) — deteksi waktu + ω tidak boleh tertimpa oleh topik sentripetal.
    if has_omega and has_time:
        return {
            "x_var": "waktu (t)" if language == "id" else "time (t)",
            "y_var": "kecepatan sudut (ω)" if language == "id" else "angular velocity (ω)",
        }

    # Sebaran ω vs a_c (grafik hubungan tanpa sumbu waktu eksplisit)
    if has_omega and has_ac and not has_time:
        return {
            "x_var": "kecepatan sudut (ω)" if language == "id" else "angular velocity (ω)",
            "y_var": "percepatan sentripetal (a_c)" if language == "id" else "centripetal acceleration (a_c)",
        }

    if has_v and not has_omega:
        return {
            "x_var": "kecepatan linear (v)" if language == "id" else "linear velocity (v)",
            "y_var": "percepatan sentripetal (a_c)" if language == "id" else "centripetal acceleration (a_c)",
        }

    if has_r and has_ac and not has_omega:
        return {
            "x_var": "radius (r)",
            "y_var": "percepatan sentripetal (a_c)" if language == "id" else "centripetal acceleration (a_c)",
        }

    if has_ac:
        return {
            "x_var": "kecepatan sudut (ω)" if language == "id" else "angular velocity (ω)",
            "y_var": "percepatan sentripetal (a_c)" if language == "id" else "centripetal acceleration (a_c)",
        }

    # Tidak ada petunjuk a_c di teks: jangan mengasumsikan a_c di sumbu (hindari salah seperti screenshot ω vs t)
    return {
        "x_var": "waktu (t)" if language == "id" else "time (t)",
        "y_var": "kecepatan sudut (ω)" if language == "id" else "angular velocity (ω)",
    }


def _axis_y_is_acceleration(y_var: str) -> bool:
    y = (y_var or "").lower()
    return any(k in y for k in ("a_c", "centripetal", "percepatan sentripetal", "acceleration (", "acceleration"))


def _stage2_llm_matches_inferred_axes(feedback: str, next_step: str, y_var: str) -> bool:
    """Jika sumbu y bukan a_c, tolak output LLM yang membahas a_c seolah ada di grafik."""
    if _axis_y_is_acceleration(y_var):
        return True
    combined = f"{feedback or ''} {next_step or ''}".lower()
    # Izinkan rumus penghubung (bukan mengklaim a_c ada di sumbu).
    if any(
        k in combined
        for k in (
            "ω²",
            "ω^2",
            "omega^2",
            "a_c =",
            "a_c=",
            "r =",
            "r=",
            "tidak ada di sumbu",
            "not on the axis",
        )
    ):
        return True
    if any(
        k in combined
        for k in (
            "a_c",
            "percepatan sentripetal",
            "centripetal acceleration",
            "sentripetal",
            "centripetal",
        )
    ):
        return False
    return True


def _ensure_stage2_canvas_focused_feedback(
    *,
    feedback: str,
    next_step: str,
    matched: str,
    language: str,
    detected_object: str,
    x_var: str,
    y_var: str,
    picked_x: Optional[str] = None,
    picked_y: Optional[str] = None,
    radius_value: Optional[float] = None,
) -> Dict[str, str]:
    """
    Pastikan feedback Stage 2 mendorong aksi analisa visual di canvas.
    """
    fb = (feedback or "").strip()
    ns = (next_step or "").strip()
    low_fb = fb.lower()
    low_ns = ns.lower()
    obj = (detected_object or "objek").strip()

    # LLM-first: pertahankan output asli jika sudah spesifik ke variabel/objek/grafik DAN konsisten dengan sumbu.
    var_markers = ("omega", "ω", "a_c", "sentripetal", "centripetal", "radius", "kecepatan", "angular", "velocity", "acceleration", "grafik", "graph")
    grounded_by_object = bool(obj) and (obj.lower() in low_fb or obj.lower() in low_ns)
    grounded_by_variables = any(m in low_fb for m in var_markers) and any(m in low_ns for m in var_markers)
    if (
        fb
        and ns
        and (grounded_by_object or grounded_by_variables)
        and _stage2_llm_matches_inferred_axes(fb, ns, y_var)
    ):
        return {"feedback": fb, "next_step": ns}

    has_canvas_action = any(
        k in low_fb
        for k in ("lingkari", "tandai", "garis", "panah", "coret", "circle", "mark", "draw", "arrow")
    )
    has_relation_or_calc = any(
        k in low_fb
        for k in (
            "hubungan",
            "relasi",
            "tren",
            "gradient",
            "kemiringan",
            "delta",
            "Δ",
            "rasio",
            "hitung",
            "estimasi",
            "relationship",
            "trend",
            "slope",
            "calculate",
            "estimate",
            "ratio",
        )
    )
    asks_text_question = ("?" in fb) or any(
        k in low_fb
        for k in ("jelaskan", "mengapa", "kenapa", "apa yang terjadi", "bagaimana", "why", "explain", "what happens", "how")
    )

    xv = x_var or ""
    yv = y_var or ""
    is_omega_vs_time = ("waktu" in xv or "time" in xv.lower()) and ("ω" in yv or "omega" in yv.lower())
    is_ac_vs_omega = _axis_y_is_acceleration(yv) and ("ω" in xv or "omega" in xv.lower())
    picked_pair = (picked_x and picked_y)

    if language == "id":
        if is_omega_vs_time:
            if matched == "correct":
                fb = (
                    f"Pada grafik {obj}, tandai dua titik (t₁, ω₁) dan (t₂, ω₂) dari data, "
                    "lalu tulis perkiraan Δω/Δt di canvas (bukan a_c; a_c tidak ada di sumbu ini)."
                )
                ns = (
                    "Dari Δω/Δt yang kamu tulis, bagaimana pola perubahan ω terhadap waktu pada grafik ini "
                    "(misalnya naik/turun/berfluktuasi)?"
                )
                if picked_pair:
                    ns = f"Dengan titik pick (x={picked_x}, y={picked_y}), bagaimana estimasi Δω/Δt terhadap titik terdekat di grafik?"
            elif (not has_canvas_action) or asks_text_question or (not has_relation_or_calc):
                fb = (
                    f"Pada grafik {obj} (ω vs t), tandai dua titik lalu hitung Δω/Δt; "
                    "a_c = ω²r hanya dipakai jika kamu menghubungkan ke percepatan sentripetal dengan r yang diketahui dari eksperimen."
                )
                ns = (
                    "Jika kamu punya satu nilai a_c dari pengaturan atau pengukuran lain, "
                    "bisa kamu hitung r = a_c/ω² dari salah satu titik (t, ω) yang kamu tandai?"
                )
                if picked_pair:
                    ns = f"Dari titik pick (x={picked_x}, y={picked_y}), titik pembanding mana yang kamu pilih untuk menghitung Δω/Δt di canvas?"
        elif is_ac_vs_omega:
            if matched == "correct":
                fb = (
                    f"Pada grafik {obj} (a_c vs ω), lingkari dua titik dan tulis estimasi Δa_c/Δω di canvas."
                )
                ns = (
                    "Jika hubungan mendekati a_c ∝ ω², berapa perkiraan r dari a_c/ω² pada salah satu titik yang kamu tandai?"
                )
                if picked_pair:
                    ns = f"Dengan titik pick (x={picked_x}, y={picked_y}), berapa perkiraan r = a_c/ω² dari titik itu?"
            elif (not has_canvas_action) or asks_text_question or (not has_relation_or_calc):
                fb = (
                    f"Tandai dua titik pada grafik {obj} ({x_var} vs {y_var}), lalu hitung estimasi kemiringan lokal di canvas."
                )
                ns = (
                    "Apa hubungan antara a_c dan ω yang kamu lihat dari dua titik dan kemiringannya pada grafik?"
                )
                if picked_pair:
                    ns = f"Dengan titik pick (x={picked_x}, y={picked_y}), bagaimana kamu membandingkannya dengan satu titik lain untuk menilai hubungan a_c terhadap ω?"
        else:
            if matched == "correct":
                fb = (
                    f"Analisis grafik {obj} kamu sudah sesuai data. "
                    f"Di canvas, lingkari dua titik kunci pada grafik {x_var} vs {y_var}, "
                    "lalu tulis estimasi Δy/Δx kecil di sampingnya."
                )
                ns = f"Setelah menandai titik, bagaimana hubungan {x_var} terhadap {y_var} yang kamu lihat dari grafik pada canvas?"
                if picked_pair:
                    ns = f"Dengan titik pick (x={picked_x}, y={picked_y}), bagaimana hubungan {x_var} terhadap {y_var} dibanding titik lain yang kamu pilih?"
            elif (not has_canvas_action) or asks_text_question or (not has_relation_or_calc):
                fb = (
                    f"Fokus pada data grafik {obj}: tandai titik rendah dan titik tinggi pada {x_var} vs {y_var}, "
                    "lalu tarik garis tren untuk melihat hubungan antar variabel."
                )
                ns = f"Dari dua titik yang kamu tandai, berapa perkiraan Δy/Δx dan apa artinya untuk hubungan variabel pada grafik?"
                if picked_pair:
                    ns = f"Dari titik pick (x={picked_x}, y={picked_y}) dan satu titik pembanding, berapa perkiraan Δy/Δx pada grafik?"
    else:
        if is_omega_vs_time:
            if matched == "correct":
                fb = (
                    f"On the {obj} graph, mark two points (t₁, ω₁) and (t₂, ω₂), "
                    "then write an estimated Δω/Δt on the canvas (not a_c; this axis is not a_c)."
                )
                ns = (
                    "From your Δω/Δt, how does ω change with time on this plot (e.g., increasing/decreasing/oscillating)?"
                )
            elif (not has_canvas_action) or asks_text_question or (not has_relation_or_calc):
                fb = (
                    f"On the {obj} ω-vs-t graph, mark two points and compute Δω/Δt; "
                    "use a_c = ω²r only when linking to centripetal acceleration with a known r from the experiment."
                )
                ns = (
                    "If you have one measured a_c from elsewhere, can you compute r = a_c/ω² from a point (t, ω) you marked?"
                )
        elif is_ac_vs_omega:
            if matched == "correct":
                fb = (
                    f"On the {obj} a_c-vs-ω graph, circle two points and write an estimated Δa_c/Δω on the canvas."
                )
                ns = (
                    "If the trend is roughly a_c ∝ ω², what is r ≈ a_c/ω² at one marked point?"
                )
            elif (not has_canvas_action) or asks_text_question or (not has_relation_or_calc):
                fb = (
                    f"Mark two points on the {obj} graph ({x_var} vs {y_var}), then estimate local slope on the canvas."
                )
                ns = (
                    "What relationship between a_c and ω do you see from the two points and the slope?"
                )
        else:
            if matched == "correct":
                fb = (
                    "Your graph analysis matches the data. "
                    f"On the canvas, circle two key points on {x_var} vs {y_var}, "
                    "and add a small estimated Δy/Δx note."
                )
                ns = f"After marking the points, what relationship of {x_var} to {y_var} do you observe from the graph on the canvas?"
            elif (not has_canvas_action) or asks_text_question or (not has_relation_or_calc):
                fb = (
                    f"Focus on graph data: mark a low point and a high point on {x_var} vs {y_var}, "
                    "then draw a trend line to infer the variable relationship."
                )
                ns = f"From your two marked points, what is the estimated Δy/Δx, and what does it mean for the variable relationship?"

    ns = _build_stage2_forced_calc_next_step(
        language=language,
        x_var=x_var,
        y_var=y_var,
        picked_x=picked_x,
        picked_y=picked_y,
        radius_value=radius_value,
    )
    return {"feedback": fb, "next_step": ns}


def _build_stage2_forced_calc_next_step(
    *,
    language: str,
    x_var: str,
    y_var: str,
    picked_x: Optional[str],
    picked_y: Optional[str],
    radius_value: Optional[float],
) -> str:
    """
    Paksa next_step menjadi pertanyaan hitung variabel dari data point + radius DB (nilai dalam cm) jika ada.
    """
    xv = (x_var or "").lower()
    yv = (y_var or "").lower()
    is_omega_vs_time = ("waktu" in xv or "time" in xv) and ("ω" in yv or "omega" in yv)
    is_ac_vs_omega = _axis_y_is_acceleration(y_var) and ("ω" in xv or "omega" in xv)
    has_pick = bool(picked_x and picked_y)
    r_txt = f"{radius_value:.4f}".rstrip("0").rstrip(".") if isinstance(radius_value, (int, float)) else None

    if language == "id":
        if has_pick and is_omega_vs_time and r_txt:
            return (
                f"Dengan titik pick (t={picked_x}, ω={picked_y}) dan radius DB r={r_txt} cm, "
                "berapa nilai a_c = ω²r pada titik itu?"
            )
        if has_pick and is_ac_vs_omega and r_txt:
            return (
                f"Dengan titik pick (ω={picked_x}, a_c={picked_y}) dan radius DB r={r_txt} cm, "
                "apakah a_c dari data mendekati ω²r pada titik itu?"
            )
        if has_pick and is_ac_vs_omega:
            return (
                f"Dengan titik pick (ω={picked_x}, a_c={picked_y}), "
                "berapa estimasi r = a_c/ω² pada titik tersebut?"
            )
        if has_pick and is_omega_vs_time:
            return (
                f"Dengan titik pick (t={picked_x}, ω={picked_y}), "
                "pilih satu titik pembanding terdekat lalu berapa estimasi Δω/Δt?"
            )
        if has_pick:
            return (
                f"Dengan titik pick (x={picked_x}, y={picked_y}), "
                "pilih satu titik pembanding lalu berapa estimasi Δy/Δx dari dua titik itu?"
            )
        if r_txt:
            return (
                f"Pilih satu data point (x,y) di grafik, lalu dengan radius DB r={r_txt} cm "
                "variabel apa yang bisa kamu hitung dari titik itu?"
            )
        return "Pilih satu data point (x,y) lalu hitung satu variabel turunan dari hubungan dua variabel pada grafik?"

    if has_pick and is_omega_vs_time and r_txt:
        return (
            f"Using the picked point (t={picked_x}, ω={picked_y}) and DB radius r={r_txt} cm, "
            "what is a_c = ω²r at that point?"
        )
    if has_pick and is_ac_vs_omega and r_txt:
        return (
            f"Using the picked point (ω={picked_x}, a_c={picked_y}) and DB radius r={r_txt} cm, "
            "does data a_c match ω²r at that point?"
        )
    if has_pick and is_ac_vs_omega:
        return f"Using the picked point (ω={picked_x}, a_c={picked_y}), what is r = a_c/ω² at that point?"
    if has_pick and is_omega_vs_time:
        return (
            f"Using the picked point (t={picked_x}, ω={picked_y}), choose one nearby comparison point "
            "and estimate Δω/Δt?"
        )
    if has_pick:
        return (
            f"Using the picked point (x={picked_x}, y={picked_y}), choose one comparison point "
            "and estimate Δy/Δx from the two points?"
        )
    if r_txt:
        return (
            f"Pick one (x,y) data point, then with DB radius r={r_txt} cm, "
            "which variable can you calculate from that point?"
        )
    return "Pick one (x,y) point and calculate one derived variable from the graph relationship?"


def _is_confusion_intent(text: str, language: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    id_markers = (
        "kurang paham", "tidak paham", "ga paham", "gak paham", "bingung",
        "belum paham", "belum ngerti", "tidak ngerti", "saya bingung",
    )
    en_markers = ("i dont understand", "i don't understand", "confused", "not sure", "still confused")
    markers = id_markers if language == "id" else en_markers
    return any(m in t for m in markers)


def _score_inquiry_quality_0_3(
    inquiry_text: str,
    detected_object_label: Optional[str],
    language: str,
) -> Dict[str, Any]:
    """
    Rubrik skor inquiry 0-3:
    0: tidak relevan topik
    1: relevan tapi tidak ada variabel jelas
    2: ada variabel + konteks grafik
    3: ada variabel + hubungan yang ditanya + konteks objek/data jelas
    """
    text = (inquiry_text or "").strip().lower()
    if not text:
        return {"score": 0, "reason": ("tidak relevan topik" if language == "id" else "not relevant")}

    relevant_markers = (
        "sentripetal", "centripetal", "percepatan", "acceleration",
        "kecepatan sudut", "angular velocity", "omega", "ω", "a_c",
        "gerak melingkar", "circular motion",
        "variabel", "hubungan", "pergerakan", "rotasi", "berputar",
    )
    variable_markers = (
        "ω", "omega", "kecepatan sudut", "angular velocity",
        "a", "a_c", "percepatan", "acceleration",
        "r", "radius", "jari-jari", "v", "kecepatan",
        "variabel", "variables",
    )
    graph_markers = (
        "grafik", "grafk", "graph", "plot", "kurva", "curve", "sumbu", "axis",
        "titik", "tren", "trend", "trebd", "data", "nilai",
        "berdasarkan", "bedasarkan", "bedasatkan",
    )
    relation_markers = (
        "hubungan", "relasi", "pengaruh", "berbanding", "linear", "kuadrat",
        "meningkat", "menurun", "naik", "turun", "sebanding",
        "relationship", "relation", "affect", "proportional", "quadratic", "linear",
    )
    object_markers = (
        "objek", "object", "roda", "wheel", "kipas", "fan", "data ini", "this data", "grafik ini", "this graph"
    )

    is_relevant = any(m in text for m in relevant_markers)
    if not is_relevant:
        return {"score": 0, "reason": ("tidak relevan topik" if language == "id" else "not relevant")}

    has_variable = any(m in text for m in variable_markers)
    if not has_variable:
        return {
            "score": 1,
            "reason": (
                "relevan tapi belum menyebut variabel fisika dengan jelas"
                if language == "id"
                else "relevant but variable is not explicit"
            ),
        }

    has_graph_context = any(m in text for m in graph_markers)
    if not has_graph_context:
        return {
            "score": 1,
            "reason": (
                "variabel sudah ada, namun konteks grafik/data belum jelas"
                if language == "id"
                else "variables present but graph/data context missing"
            ),
        }

    has_relation = ("?" in text) or any(m in text for m in relation_markers)
    obj = (detected_object_label or "").strip().lower()
    has_object_context = any(m in text for m in object_markers) or (obj and obj in text) or _matches_detected_object(
        inquiry_text, detected_object_label
    )

    if has_relation and has_object_context:
        return {
            "score": 3,
            "reason": (
                "variabel, relasi yang ditanya, dan konteks objek/data sudah jelas"
                if language == "id"
                else "variables, relationship, and object/data context are clear"
            ),
        }

    return {
        "score": 2,
        "reason": (
            "variabel dan konteks grafik sudah ada, tapi relasi/objek belum lengkap"
            if language == "id"
            else "variables and graph context are present, relation/object context is incomplete"
        ),
    }


def _matches_detected_object(inquiry_text: str, detected_object_label: Optional[str]) -> bool:
    """
    Cocokkan teks inquiry siswa dengan object terakhir dari DB.
    Mendukung alias ID/EN umum agar 'kipas' bisa cocok dengan 'fan', dst.
    """
    text = (inquiry_text or "").strip().lower()
    obj = (detected_object_label or "").strip().lower()
    if not text or not obj:
        return False
    if obj in text:
        return True

    alias_groups = {
        "fan": {"fan", "kipas"},
        "wheel": {"wheel", "roda"},
        "ball": {"ball", "bola"},
        "cup": {"cup", "gelas"},
        "bottle": {"bottle", "botol"},
    }

    tokens_obj = set(re.findall(r"[a-zA-Z_]+", obj.replace("-", " ").replace("_", " ")))
    for key, aliases in alias_groups.items():
        if key in tokens_obj or any(a in obj for a in aliases):
            if any(a in text for a in aliases):
                return True
            # Toleransi typo ringan pada alias object.
            text_tokens = set(re.findall(r"[a-zA-Z_]+", text))
            for alias in aliases:
                if _contains_marker_fuzzy(" ".join(text_tokens), [alias], min_ratio=0.84):
                    return True

    # Fallback token overlap minimum 1 kata bermakna.
    tokens_text = set(re.findall(r"[a-zA-Z_]+", text))
    meaningful = {t for t in tokens_obj if len(t) >= 3}
    return len(tokens_text & meaningful) > 0


def _normalize_text_for_fuzzy(text: str) -> str:
    t = (text or "").lower()
    t = re.sub(r"[^a-z0-9_\sω]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _tokenize_for_fuzzy(text: str) -> List[str]:
    t = _normalize_text_for_fuzzy(text)
    return [x for x in t.split(" ") if x]


def _word_fuzzy_in_tokens(word: str, tokens: List[str], min_ratio: float = 0.86) -> bool:
    if not word:
        return False
    if word in tokens:
        return True
    if len(word) <= 2:
        return False
    for tok in tokens:
        if abs(len(tok) - len(word)) > 2:
            continue
        if SequenceMatcher(None, word, tok).ratio() >= min_ratio:
            return True
    return False


def _contains_marker_fuzzy(text: str, markers: tuple, min_ratio: float = 0.86) -> bool:
    normalized = _normalize_text_for_fuzzy(text)
    tokens = _tokenize_for_fuzzy(normalized)
    if not normalized:
        return False
    for marker in markers:
        mk = _normalize_text_for_fuzzy(marker)
        if not mk:
            continue
        # Exact substring tetap prioritas.
        if mk in normalized:
            return True
        parts = [p for p in mk.split(" ") if p]
        # Fuzzy untuk marker 1-2 kata (cukup aman untuk typo ringan).
        if len(parts) == 1:
            if _word_fuzzy_in_tokens(parts[0], tokens, min_ratio=min_ratio):
                return True
        elif len(parts) == 2:
            if _word_fuzzy_in_tokens(parts[0], tokens, min_ratio=min_ratio) and _word_fuzzy_in_tokens(
                parts[1], tokens, min_ratio=min_ratio
            ):
                return True
    return False


def _stage1_detect_concept_coverage(answer_text: str) -> Dict[str, bool]:
    """
    Deteksi sederhana cakupan konsep jawaban siswa:
    - omega_up_ac_up: saat kecepatan sudut naik, percepatan sentripetal naik
    - radius_up_ac_down: saat radius naik, percepatan sentripetal turun (dengan variabel lain konstan)
    """
    t = (answer_text or "").strip().lower()
    has_omega = _contains_marker_fuzzy(
        t,
        (
            "omega",
            "ω",
            "kecepatan sudut",
            "angular velocity",
            "kecepatan putar",
            "semakin cepat berputar",
            "cepat berputar",
            "putaran",
            "rpm",
            "frekuensi putar",
        ),
        min_ratio=0.84,
    )
    has_radius = _contains_marker_fuzzy(t, ("radius", "jari-jari"), min_ratio=0.84) or (" r " in f" {t} ")
    has_ac = _contains_marker_fuzzy(t, ("sentripetal", "centripetal", "percepatan"), min_ratio=0.84)
    has_up = _contains_marker_fuzzy(
        t,
        (
            "naik",
            "meningkat",
            "bertambah",
            "increase",
            "increases",
            "semakin",
            "dipercepat",
            "makin cepat",
            "semakin cepat",
        ),
        min_ratio=0.84,
    )
    has_down = _contains_marker_fuzzy(t, ("turun", "berkurang", "menurun", "decrease", "decreases"), min_ratio=0.84)
    has_inverse = _contains_marker_fuzzy(t, ("berbanding terbalik", "invers", "inverse"), min_ratio=0.84)

    mentions_rotation_rate = has_omega or (_contains_marker_fuzzy(t, ("berputar", "putaran", "rotasi", "spin", "gerak", "gerakan"), min_ratio=0.84) and has_up)
    omega_up_ac_up = mentions_rotation_rate and has_ac and has_up
    radius_up_ac_down = has_radius and has_ac and (has_down or has_inverse)
    return {
        "omega_up_ac_up": omega_up_ac_up,
        "radius_up_ac_down": radius_up_ac_down,
    }


def _extract_llm_confidence(data: Dict[str, Any]) -> float:
    evaluator = data.get("evaluator")
    raw = None
    if isinstance(evaluator, dict):
        raw = evaluator.get("confidence")
    if raw is None:
        raw = data.get("confidence")
    try:
        c = float(raw)
    except Exception:
        return 50.0
    return max(0.0, min(100.0, c))


def _stage1_detect_answer_signals(
    *,
    answer_text: str,
    active_inquiry: str,
    detected_object: str,
) -> Dict[str, bool]:
    t = (answer_text or "").strip().lower()
    q = (active_inquiry or "").strip().lower()
    obj = (detected_object or "").strip().lower()
    tokens = re.findall(r"[a-zA-Z_]+", t)

    has_motion_terms = _contains_marker_fuzzy(t, ("gerak", "berputar", "putaran", "rotasi", "motion", "rotate", "spin"), min_ratio=0.84)
    has_centripetal_terms = _contains_marker_fuzzy(t, ("sentripetal", "centripetal", "a_c", "percepatan"), min_ratio=0.84)
    has_relation_terms = _contains_marker_fuzzy(
        t,
        ("jika", "maka", "hubungan", "berbanding", "naik", "turun", "meningkat", "menurun", "if", "then", "increase", "decrease"),
        min_ratio=0.84,
    )
    has_object = _matches_detected_object(answer_text, detected_object) if obj else False
    contradiction = any(k in t for k in ("tidak berpengaruh", "tidak ada hubungan", "no relation", "independent"))
    formula_like = any(k in t for k in ("=", "ω", "omega", "a_c", "v²", "r", "kuadrat"))
    mentions_question_focus = any(k in q for k in ("gerak", "sentripetal", "hubungan")) and (
        has_motion_terms or has_centripetal_terms
    )
    off_topic = not (has_motion_terms or has_centripetal_terms or mentions_question_focus)
    too_short = len(tokens) < 4

    return {
        "empty_answer": not t,
        "too_short": too_short,
        "off_topic": off_topic,
        "has_motion_terms": has_motion_terms,
        "has_centripetal_terms": has_centripetal_terms,
        "has_relation_terms": has_relation_terms,
        "has_object": has_object,
        "contradiction": contradiction,
        "formula_like": formula_like,
    }


def _apply_stage1_evaluator_decision(
    *,
    llm_matched: str,
    llm_confidence: float,
    answer_text: str,
    active_inquiry: str,
    detected_object: str,
    language: str,
) -> Dict[str, Any]:
    """
    Decision table evaluasi Stage-1.
    Aturan dibuat eksplisit untuk banyak pola jawaban siswa agar tidak terlalu strict.
    """
    cov = _stage1_detect_concept_coverage(answer_text)
    sig = _stage1_detect_answer_signals(
        answer_text=answer_text,
        active_inquiry=active_inquiry,
        detected_object=detected_object,
    )
    at_least_one_core = cov["omega_up_ac_up"] or cov["radius_up_ac_down"]
    both_core = cov["omega_up_ac_up"] and cov["radius_up_ac_down"]

    # Decision table (prioritas dari atas ke bawah):
    # R1 empty/confused -> uncertain
    if sig["empty_answer"] or _is_confusion_intent(answer_text, language):
        return {"matched": "uncertain", "confidence": max(35.0, llm_confidence), "rule_id": "R1"}
    # R2 off-topic kuat -> wrong
    if sig["off_topic"] and not at_least_one_core:
        return {"matched": "wrong", "confidence": max(65.0, llm_confidence), "rule_id": "R2"}
    # R3 kontradiksi konsep -> wrong
    if sig["contradiction"] and not at_least_one_core:
        return {"matched": "wrong", "confidence": max(70.0, llm_confidence), "rule_id": "R3"}
    # R4 dua konsep inti terpenuhi -> correct
    if both_core:
        return {"matched": "correct", "confidence": max(85.0, llm_confidence), "rule_id": "R4"}
    # R5 satu konsep inti + relasi -> correct (longgar)
    if at_least_one_core and sig["has_relation_terms"]:
        return {"matched": "correct", "confidence": max(72.0, llm_confidence), "rule_id": "R5"}
    # R6 satu konsep inti saja -> uncertain / correct berdasarkan confidence
    if at_least_one_core:
        return {
            "matched": "correct" if llm_confidence >= 70.0 else "uncertain",
            "confidence": max(60.0, llm_confidence),
            "rule_id": "R6",
        }
    # R7 jawaban sangat pendek tapi on-topic -> uncertain
    if sig["too_short"] and not sig["off_topic"]:
        return {"matched": "uncertain", "confidence": max(45.0, llm_confidence), "rule_id": "R7"}
    # R8 formula-like on-topic -> uncertain
    if sig["formula_like"] and (sig["has_motion_terms"] or sig["has_centripetal_terms"]):
        return {"matched": "uncertain", "confidence": max(55.0, llm_confidence), "rule_id": "R8"}
    # R9 fallback confidence gate (sesuai request >=70% dianggap benar, selama tidak off-topic)
    if llm_confidence >= 70.0 and not sig["off_topic"]:
        return {"matched": "correct", "confidence": llm_confidence, "rule_id": "R9"}
    # R10 gunakan verdict LLM bila tidak terkena rule lain
    if llm_matched in ("correct", "wrong", "uncertain"):
        return {"matched": llm_matched, "confidence": llm_confidence, "rule_id": "R10"}
    # R11 default aman
    return {"matched": "uncertain", "confidence": llm_confidence, "rule_id": "R11"}


def _adjust_stage1_feedback_by_coverage(
    *,
    answer_text: str,
    language: str,
    matched: str,
    feedback: str,
    next_step: str,
) -> Dict[str, str]:
    """
    Cegah feedback redundant: jika konsep sudah disebut benar oleh siswa,
    jangan menanyakan konsep yang sama lagi.
    """
    cov = _stage1_detect_concept_coverage(answer_text)
    has_omega_concept = cov["omega_up_ac_up"]
    has_radius_concept = cov["radius_up_ac_down"]
    both_core_concepts = has_omega_concept and has_radius_concept
    at_least_one_core_concept = has_omega_concept or has_radius_concept
    fb_low = (feedback or "").lower()

    asks_radius_again = any(k in fb_low for k in ("radius", "jari-jari")) and any(
        k in fb_low for k in ("apa yang terjadi", "bagaimana", "what happens", "how")
    )
    asks_omega_again = any(k in fb_low for k in ("omega", "kecepatan sudut", "angular velocity")) and any(
        k in fb_low for k in ("apa yang terjadi", "bagaimana", "what happens", "how")
    )

    # Mode lebih longgar: jika siswa sudah benar pada satu variabel inti,
    # terima sebagai jawaban benar dasar (tidak memaksa menambah variabel lain).
    if at_least_one_core_concept and matched == "wrong":
        if language == "id":
            feedback = "Jawabanmu sudah tepat untuk hubungan variabel yang kamu jelaskan."
            if not (next_step or "").strip():
                next_step = "Kalau mau, tambahkan contoh singkat dari grafik sebagai penguat."
        else:
            feedback = "Your answer is correct for the variable relationship you explained."
            if not (next_step or "").strip():
                next_step = "Optionally add one short graph-based example to strengthen your claim."
        matched = "correct"

    if both_core_concepts and (matched == "wrong" or asks_radius_again):
        if language == "id":
            feedback = (
                "Jawabanmu sudah bagus dan mencakup poin inti. "
                "Kamu sudah tepat tentang pengaruh kecepatan sudut dan radius terhadap percepatan sentripetal."
            )
            if not (next_step or "").strip():
                next_step = "Langkah berikutnya: sebutkan dari grafik variabel mana yang dijaga konstan."
        else:
            feedback = (
                "Your answer already covers the core ideas: when angular velocity increases, centripetal acceleration increases, "
                "and when radius increases, centripetal acceleration decreases (with other variables held constant)."
            )
            if not (next_step or "").strip():
                next_step = "Use the graph to state which variable is held constant."
        matched = "uncertain"

    # Jika satu konsep sudah benar, hindari menanyakan ulang konsep yang sama.
    if has_radius_concept and asks_radius_again:
        if language == "id":
            feedback = "Bagian radiusmu sudah benar."
        else:
            feedback = "Your radius part is already correct."
    if has_omega_concept and asks_omega_again:
        if language == "id":
            feedback = "Bagian kecepatan sudutmu sudah benar."
        else:
            feedback = "Your angular-velocity part is already correct."

    return {
        "matched": matched,
        "feedback": feedback,
        "next_step": next_step,
    }


def _enforce_stage1_same_question_focus(
    *,
    active_inquiry: str,
    feedback: str,
    next_step: str,
    matched: str,
    language: str,
) -> Dict[str, str]:
    """
    Jaga agar feedback Stage 1 tetap fokus pada pertanyaan awal (active inquiry),
    terutama saat jawaban dinilai wrong.
    """
    ai = (active_inquiry or "").strip()
    fb = (feedback or "").strip()
    ns = (next_step or "").strip()
    if not ai:
        return {"feedback": fb, "next_step": ns}

    low = f"{fb} {ns}".lower()
    shifts_to_new_condition = any(
        k in low
        for k in (
            "tetap konstan",
            "held constant",
            "if constant",
            "jika konstan",
            "variabel lain konstan",
        )
    )

    if matched == "wrong" or shifts_to_new_condition:
        if language == "id":
            fb = f"Jawabanmu belum sepenuhnya menjawab pertanyaan awal: \"{ai}\"."
            ns = "Coba jawab ulang pertanyaan awal itu secara langsung dengan satu hubungan sebab-akibat yang jelas."
        else:
            fb = f'Your response does not fully answer the original question: "{ai}".'
            ns = "Answer that original question directly using one clear cause-effect relationship."
    return {"feedback": fb, "next_step": ns}


def _normalize_confusion_json(data: Dict[str, Any], language: str) -> Dict[str, Any]:
    """Normalisasi output LLM ke schema step_by_step / example / comprehension_check."""
    steps_raw = data.get("step_by_step")
    steps: List[str] = []
    if isinstance(steps_raw, list):
        for s in steps_raw[:3]:
            t = str(s or "").strip()
            if t:
                steps.append(t)
    elif isinstance(steps_raw, str) and steps_raw.strip():
        steps = [steps_raw.strip()]

    example = str(data.get("example") or "").strip()
    check = str(data.get("comprehension_check") or "").strip()
    # Satu pertanyaan saja untuk comprehension_check (ambil kalimat pertama jika ada newline)
    if "\n" in check:
        check = check.split("\n", 1)[0].strip()
    if check and "?" not in check and language == "id":
        check = check.rstrip(".") + "?"

    feedback_lines = steps if steps else []
    feedback_text = "\n".join(f"{i + 1}. {line}" for i, line in enumerate(feedback_lines))
    next_text = ""
    if example and check:
        next_text = f"{example}\n\n{check}"
    elif example:
        next_text = example
    elif check:
        next_text = check

    return {
        "step_by_step": steps or None,
        "example": example or None,
        "comprehension_check": check or None,
        "feedback": feedback_text or None,
        "next_step": next_text or None,
    }


def _build_confusion_help_response(req: SubmitResponseRequest, language: str) -> SubmitResponseResponse:
    context = _resolve_stage1_context(req)
    topic = (req.topic or "centripetal acceleration").strip()
    user_answer = (req.response_text or "").strip()
    key = os.getenv("OPENAI_API_KEY", "").strip()
    model_name = os.getenv("INQUIRY_STAGE1_SUBMIT_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
    norm: Dict[str, Any] = {}
    if key:
        prompt = (
            f"Target language: {_language_label(language)}\n"
            f"Topic: {topic}\n"
            f"Context: {context}\n"
            f"Student says: {user_answer}\n\n"
            "The student indicates confusion or does not understand.\n"
            "Return ONLY valid JSON with exactly these keys:\n"
            '- "step_by_step": array of 2 to 3 short strings (each step one sentence, no numbering prefix in the string)\n'
            '- "example": one short concrete everyday example (one or two sentences)\n'
            '- "comprehension_check": exactly ONE short question to check understanding (must end with ?)\n'
            "Do not add any other keys. Do not use single quotes; use double quotes in JSON.\n"
            "Keep physics accurate for centripetal acceleration in circular motion."
        )
        system_msg = (
            "You are a physics tutor. Output strictly valid JSON only, no markdown, no code fences."
        )
        try:
            resp = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.35,
                },
                timeout=35,
            )
            resp.raise_for_status()
            content = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "")
            raw = (content or "").strip()
            if raw.startswith("```"):
                raw = re.sub(r"^```(?:json)?\s*", "", raw)
                raw = re.sub(r"\s*```$", "", raw)
            data = json.loads(raw)
            norm = _normalize_confusion_json(data, language)
        except Exception:
            logger.warning("stage1 submit: confusion-help LLM failed, returning None fields", exc_info=True)
            norm = {}

    return SubmitResponseResponse(
        feedback=norm.get("feedback"),
        next_step=norm.get("next_step"),
        matched_answer_type="uncertain",
        step_by_step=norm.get("step_by_step"),
        example=norm.get("example"),
        comprehension_check=norm.get("comprehension_check"),
    )


def _llm_stage1_feedback(
    req: SubmitResponseRequest,
    stage: str,
    rows: List[dict],
) -> Optional[Dict[str, str]]:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        return None
    model_name = os.getenv("INQUIRY_STAGE1_SUBMIT_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
    context = _resolve_stage1_context(req)
    session_key = _get_stage1_session_key(req)
    memory = _STAGE1_CONVERSATION_MEMORY.get(session_key, [])
    memory_block = "\n".join(
        [f"{m.get('role', 'user')}: {m.get('content', '')}" for m in memory[-_MAX_MEMORY_TURNS:]]
    ) or "(no previous turns)"
    fewshot = _build_stage1_fewshot_block(rows, limit=6)
    topic = (req.topic or "centripetal acceleration").strip()
    user_answer = (req.response_text or "").strip()
    active_inquiry = (req.current_inquiry_text or "").strip()
    language = _normalize_language(req.language, text_hint=f"{topic} {user_answer}")
    detected_object = _resolve_detected_object_label(req.user_id) or "the chosen object"
    user_prompt = (
        f"Stage: {stage}\n"
        f"Topic: {topic}\n"
        f"Target language: {_language_label(language)}\n"
        f"Detected object to mention: {detected_object}\n"
        f"{context}\n\n"
        f"Active inquiry shown to student (must be used as grading reference):\n{active_inquiry or '(not provided)'}\n\n"
        f"Conversation memory:\n{memory_block}\n\n"
        f"Few-shot Stage 1 examples:\n{fewshot}\n\n"
        f"Current student answer:\n{user_answer}\n\n"
        "You MUST evaluate the student answer against the active inquiry shown to student. "
        "Do not grade against a different inferred question.\n"
        "Return ONLY valid JSON with keys: inquiry, feedback, next_step, matched_answer_type, evaluator.\n"
        "Make response style close to Stage 1 few-shot examples. "
        "IMPORTANT: inquiry, feedback, and next_step must be written in the target language. "
        "The inquiry MUST explicitly mention the detected object name for problem_exploring. "
        "For matched_answer_type='wrong', prefer Socratic guidance style. "
        "The evaluator object MUST be valid JSON with keys: "
        "confidence (0-100 number), decision_tag (short string), evidence_points (array of short strings, max 3). "
        "Keep output concise: feedback should contain at most one guiding question; "
        "next_step should be one short action sentence without a question mark."
    )
    system_msg = (
        "You are a physics tutor for centripetal acceleration (problem exploring stage 1). "
        "Use few-shot examples and conversation memory to generate feedback. "
        "Output JSON only."
    )
    try:
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.4,
            },
            timeout=35,
        )
        resp.raise_for_status()
        content = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "")
        if not content:
            return None
        data = _safe_json_loads_from_llm(content)
        if not data:
            return None
        inquiry = str(data.get("inquiry") or "").strip()
        feedback = str(data.get("feedback") or "").strip()
        next_step = str(data.get("next_step") or "").strip()
        matched = str(data.get("matched_answer_type") or "").strip().lower()
        llm_conf = _extract_llm_confidence(data)
        if not inquiry:
            return None
        inquiry = _enforce_problem_exploring_object(inquiry, detected_object, language=language)
        if not inquiry or inquiry.lower() == "none":
            return None
        decided = _apply_stage1_evaluator_decision(
            llm_matched=matched,
            llm_confidence=llm_conf,
            answer_text=user_answer,
            active_inquiry=active_inquiry,
            detected_object=detected_object,
            language=language,
        )
        matched = decided.get("matched") or matched
        if matched == "wrong":
            rewritten = _llm_rewrite_wrong_feedback_non_reveal(
                api_key=key,
                model_name=model_name,
                language=language,
                topic=topic,
                student_answer=user_answer,
                feedback=feedback,
                next_step=next_step,
            )
            feedback = rewritten.get("feedback") or feedback
            next_step = rewritten.get("next_step") or next_step
        adjusted = _adjust_stage1_feedback_by_coverage(
            answer_text=user_answer,
            language=language,
            matched=matched,
            feedback=feedback,
            next_step=next_step,
        )
        matched = adjusted.get("matched") or matched
        feedback = adjusted.get("feedback") or feedback
        next_step = adjusted.get("next_step") or next_step
        focused = _enforce_stage1_same_question_focus(
            active_inquiry=active_inquiry,
            feedback=feedback,
            next_step=next_step,
            matched=matched,
            language=language,
        )
        feedback = focused.get("feedback") or feedback
        next_step = focused.get("next_step") or next_step
        compact = _compact_feedback_and_next_step(feedback, next_step, language)
        feedback = compact.get("feedback") or feedback
        next_step = compact.get("next_step") or next_step
        # simpan memory per session
        mem = _STAGE1_CONVERSATION_MEMORY.setdefault(session_key, [])
        mem.append({"role": "user", "content": user_answer})
        mem.append(
            {
                "role": "assistant",
                "content": f"inquiry={inquiry} | feedback={feedback} | next_step={next_step}",
            }
        )
        if len(mem) > (_MAX_MEMORY_TURNS * 2):
            _STAGE1_CONVERSATION_MEMORY[session_key] = mem[-(_MAX_MEMORY_TURNS * 2):]
        return {
            "inquiry": inquiry,
            "feedback": feedback,
            "next_step": next_step,
            "matched_answer_type": matched,
        }
    except Exception:
        logger.warning("stage1 submit: LLM feedback failed, returning None", exc_info=True)
        return None


def _normalized_stage(stage: Optional[str]) -> str:
    s = (stage or "problem_finding").strip().lower()
    return s if s in ("problem_finding", "problem_exploring") else "problem_finding"


def _generate_inquiry_internal(req: GenerateInquiryRequest, forced_stage: Optional[str] = None) -> GenerateInquiryResponse:
    """
    Core generator shared by:
    - /inquiry/generate/problem-finding (forced)
    - /inquiry/generate/problem-exploring-stage-1 (forced)
    """
    stage = _normalized_stage(forced_stage or "problem_finding")
    uid = req.user_id or "default"
    logger.info(
        "inquiry generate internal: user_id=%s stage=%s location=%s forced_stage=%s",
        uid,
        stage,
        req.location or "(empty)",
        forced_stage or "(none)",
    )

    profile = get_student_profile(uid) if uid else None
    ability_band = (profile.get("ability_band") if profile else None) or "neutral"
    is_targeted = False
    language = _normalize_language(req.language, text_hint=req.topic)

    # Default untuk problem_finding: pakai location dari request.
    location_for_prompt = (req.location or "").strip() or "your surroundings"
    detected_object_label: Optional[str] = None
    experiment_context: Optional[str] = None
    if stage == "problem_exploring":
        # problem_exploring: konteks objek wajib dari DB (processed_images), bukan request location
        location_for_prompt = "the chosen object"
        detected = get_latest_detected_object(uid)
        if detected and detected.get("label"):
            detected_object_label = (detected.get("label") or "").strip()
            if detected_object_label:
                location_for_prompt = detected_object_label.replace("_", " ")
                logger.info("inquiry problem_exploring: using object from DB: %s", detected_object_label)
        else:
            logger.info("inquiry problem_exploring: no DB object, continuing with generic object label")

    inquiry = generate_inquiry_text(
        stage=stage,
        topic=req.topic or "centripetal acceleration",
        location=location_for_prompt,
        user_id=uid,
        ability_band=ability_band,
        is_targeted=is_targeted,
        experiment_context=experiment_context,
        language=language,
    )
    return GenerateInquiryResponse(
        inquiry=inquiry,
        stage=stage,
        ability_band=ability_band,
        targeting_mode="targeted" if is_targeted else "neutral",
        detected_object=detected_object_label,
    )


@router_inquiry.post("/generate/problem-finding", response_model=GenerateInquiryResponse)
async def generate_inquiry_problem_finding(req: GenerateInquiryRequest):
    """
    New endpoint khusus problem_finding.
    Nilai req.stage diabaikan; stage dipaksa problem_finding.
    """
    try:
        logger.info("POST /inquiry/generate/problem-finding: user_id=%s", req.user_id)
        return _generate_inquiry_internal(req=req, forced_stage="problem_finding")
    except Exception as e:
        logger.exception("inquiry generate problem-finding error: %s", e)
        return GenerateInquiryResponse(inquiry="none", stage="problem_finding")


@router_inquiry.post("/generate/problem-exploring-stage-1", response_model=GenerateInquiryResponse)
async def generate_inquiry_problem_exploring(req: GenerateInquiryProblemExploringStage1Request):
    """
    New endpoint khusus problem_exploring.
    Konteks object name diambil dari DB per user session; request tidak menerima location.
    """
    try:
        logger.info("POST /inquiry/generate/problem-exploring-stage-1: user_id=%s", req.user_id)
        req_internal = GenerateInquiryRequest(
            user_id=req.user_id or "default",
            topic=req.topic or "centripetal acceleration",
            location="",
            language=req.language or "id",
        )
        return _generate_inquiry_internal(req=req_internal, forced_stage="problem_exploring")
    except Exception as e:
        logger.exception("inquiry generate problem-exploring error: %s", e)
        return GenerateInquiryResponse(inquiry="none", stage="problem_exploring")


def _submit_response_internal(req: SubmitResponseRequest, forced_stage: str = "problem_exploring") -> SubmitResponseResponse:
    stage = _normalized_stage(forced_stage)
    language = _normalize_language(req.language, text_hint=f"{req.topic or ''} {req.response_text or ''}")
    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=503, detail="Database connection failed")
    ensure_inquiry_tables(conn)
    conn.close()
    if _is_confusion_intent(req.response_text or "", language):
        increment_questions_answered(req.user_id)
        return _build_confusion_help_response(req, language)
    # Submit response stage-1:
    # Hanya gunakan hasil LLM. Jika gagal, kembalikan None.
    rows = _load_stage1_testset_rows()
    llm_result = _llm_stage1_feedback(req=req, stage=stage, rows=rows)

    increment_questions_answered(req.user_id)
    if llm_result:
        compact = _compact_feedback_and_next_step(
            feedback=(llm_result.get("feedback") or ""),
            next_step=(llm_result.get("next_step") or ""),
            language=language,
        )
        return SubmitResponseResponse(
            feedback=(compact.get("feedback") or None),
            next_step=(compact.get("next_step") or None),
            matched_answer_type=(llm_result.get("matched_answer_type") or None),
        )
    return SubmitResponseResponse(feedback=None, next_step=None, matched_answer_type=None)


def _submit_response_stage2_internal(req: SubmitResponseRequest) -> SubmitResponseResponse:
    """Submit jawaban stage 2 (grafik/sensor): few-shot stage2 + opsional gambar."""
    language = _normalize_language(req.language, text_hint=f"{req.topic or ''} {req.response_text or ''}")
    detected_obj = _resolve_detected_object_label(req.user_id)
    object_matched = _matches_detected_object(req.response_text or "", detected_obj)
    score_pack = _score_inquiry_quality_0_3(
        inquiry_text=req.response_text or "",
        detected_object_label=detected_obj,
        language=language,
    )
    score = int(score_pack.get("score") or 0)
    score_reason = str(score_pack.get("reason") or "").strip()
    if language == "id":
        score_prefix = f"Skor inquiry: {score}/3."
    else:
        score_prefix = f"Inquiry score: {score}/3."
    if score_reason:
        score_prefix = f"{score_prefix} {score_reason}"
    if language == "id":
        score_prefix += f"\nObject DB match: {'ya' if object_matched else 'tidak'}"
    else:
        score_prefix += f"\nDB object match: {'yes' if object_matched else 'no'}"

    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=503, detail="Database connection failed")
    ensure_inquiry_tables(conn)
    conn.close()
    if _is_confusion_intent(req.response_text or "", language):
        increment_questions_answered(req.user_id)
        return _build_confusion_help_response(req, language)
    rows = _load_stage2_testset_rows()
    llm_result = _llm_stage2_feedback(req=req, stage="problem_exploring_stage2", rows=rows)

    increment_questions_answered(req.user_id)
    if llm_result:
        compact = _compact_feedback_and_next_step(
            feedback=(llm_result.get("feedback") or ""),
            next_step=(llm_result.get("next_step") or ""),
            language=language,
        )
        llm_feedback = (compact.get("feedback") or "").strip()
        merged_feedback = score_prefix if not llm_feedback else (score_prefix + "\n\n" + llm_feedback)
        return SubmitResponseResponse(
            feedback=merged_feedback,
            next_step=(compact.get("next_step") or None),
            matched_answer_type=(llm_result.get("matched_answer_type") or None),
        )
    # Jika LLM gagal, tetap kirim skor inquiry agar siswa tetap mendapat evaluasi format pertanyaan.
    return SubmitResponseResponse(feedback=score_prefix, next_step=None, matched_answer_type=None)


@router_inquiry.post("/generate/problem-exploring-stage-2", response_model=GenerateInquiryResponse)
async def generate_inquiry_problem_exploring_stage2(req: GenerateInquiryProblemExploringStage2Request):
    """
    Stage 2: pertanyaan berbasis interpretasi grafik/sensor.
    Opsional: graph_image_base64 (JPEG/PNG, raw base64 atau data URL).
    """
    try:
        logger.info("POST /inquiry/generate/problem-exploring-stage-2: user_id=%s", req.user_id)
        inquiry = _llm_generate_stage2_inquiry(req)
        if not inquiry:
            inquiry = "none"
        return GenerateInquiryResponse(inquiry=inquiry, stage="problem_exploring_stage2")
    except Exception as e:
        logger.exception("inquiry generate stage2 error: %s", e)
        return GenerateInquiryResponse(
            inquiry="none",
            stage="problem_exploring_stage2",
        )


@router_inquiry.post("/generate/problem-exploring-stage-1/submit-response", response_model=SubmitResponseResponse)
async def submit_response_problem_exploring(req: SubmitResponseRequest):
    """
    Submit jawaban khusus alur problem_exploring.
    Stage dipaksa ke problem_exploring.
    """
    return _submit_response_internal(req=req, forced_stage="problem_exploring")


@router_inquiry.post("/generate/problem-exploring-stage-2/submit-response", response_model=SubmitResponseResponse)
async def submit_response_problem_exploring_stage2(req: SubmitResponseRequest):
    """Submit jawaban untuk inquiry stage 2 (kirim ulang graph_image_base64 jika ingin konteks visual)."""
    return _submit_response_stage2_internal(req)
