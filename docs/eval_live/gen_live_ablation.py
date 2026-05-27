#!/usr/bin/env python3
"""
Live multimodal ablation on real student photos (Firestore URLs).

Conditions (4):
  A. zero_shot_no_image              — no image, no recognition (text-only baseline)
  B. zero_shot_image_no_recognition  — image only, no object/location labels
  C. zero_shot_image_recognition     — image + recognition (object + location)
  D. few_shot_cot_recognition        — few-shot + CoT + image + recognition

Output JSONL for eval_inquiry_metrics.py.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
FASTAPI_DIR = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(FASTAPI_DIR))

from gen_ablation_problem_finder import (  # noqa: E402
    SYSTEM_PROMPT_GROUNDED_ID,
    _call_openai_chat,
    _pick_few_shot_examples,
    _system_message,
    _user_prompt_zero_shot_image,
    _user_prompt_zero_shot_no_image,
)

try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv(FASTAPI_DIR / ".env")
except ImportError:
    pass

CONDITIONS = [
    "zero_shot_no_image",
    "zero_shot_image_no_recognition",
    "zero_shot_image_recognition",
    "few_shot_cot_recognition",
]


def _user_prompt_image_no_recognition(lang: str, image_url: Optional[str]) -> List[Dict[str, Any]]:
    if lang == "id":
        text = (
            "Topik: percepatan sentripetal.\n"
            "Anda hanya melihat gambar dari kamera siswa. Anda TIDAK diberi label objek atau lokasi.\n"
            "Buat SATU pertanyaan inquiry umum yang mengajak siswa mengamati gerakan melingkar "
            "pada benda di gambar, TANPA menyebut nama merek/produk tertentu dan TANPA menebak label objek.\n"
            "Keluarkan HANYA satu kalimat pertanyaan."
        )
    else:
        text = (
            "Topic: centripetal acceleration.\n"
            "You only see the student's camera image. You are NOT given object or location labels.\n"
            "Write ONE general inquiry inviting observation of circular motion in the scene, "
            "WITHOUT naming a specific product or guessing the object label.\n"
            "Output ONLY one sentence."
        )
    parts: List[Dict[str, Any]] = []
    if image_url:
        parts.append({"type": "image_url", "image_url": {"url": image_url}})
    parts.append({"type": "text", "text": text})
    return parts


def _user_prompt_few_shot_cot(
    lang: str,
    object_label: str,
    location: str,
    examples: List[str],
    image_url: Optional[str],
) -> List[Dict[str, Any]]:
    bullets = "\n".join(f"- {e}" for e in examples) if examples else "-"
    if lang == "id":
        text = (
            f"Topik: percepatan sentripetal (kelas 11).\n"
            f"Hasil pengenalan objek (recognition): {object_label}.\n"
            f"Lokasi pengamatan: {location}.\n\n"
            f"Contoh pertanyaan inquiry TERBAIK (objek/lokasi lain — tiru POLA spesifiknya, jangan salin kata demi kata):\n{bullets}\n\n"
            f"Rencanakan SECARA INTERNAL (jangan ditampilkan):\n"
            f"  1) Identifikasi bagian {object_label} pada gambar di {location} yang bergerak melingkar.\n"
            f"  2) Tentukan apa yang harus diamati siswa tentang (r), (ω), dan (a) pada objek itu.\n"
            f"  3) Tulis pertanyaan yang WAJIB menyertakan \"{object_label}\" DAN \"{location}\" "
            f"dalam satu kalimat, memakai notasi (r), (ω), dan/atau (a).\n\n"
            f"Lihat gambar. Keluarkan HANYA SATU kalimat pertanyaan inquiry final."
        )
    else:
        text = (
            f"Topic: centripetal acceleration.\n"
            f"Object: {object_label}. Location: {location}.\n"
            f"Examples:\n{bullets}\n\n"
            f"Reason internally, then output ONE inquiry sentence mentioning "
            f"\"{object_label}\" and \"{location}\" with (r), (ω), and/or (a)."
        )
    parts: List[Dict[str, Any]] = []
    if image_url:
        parts.append({"type": "image_url", "image_url": {"url": image_url}})
    parts.append({"type": "text", "text": text})
    return parts


def _build_messages(
    condition: str,
    lang: str,
    object_label: str,
    location: str,
    image_url: Optional[str],
    examples: List[str],
) -> List[Dict[str, Any]]:
    if condition == "zero_shot_no_image":
        return [
            _system_message(lang, grounded=False),
            {"role": "user", "content": _user_prompt_zero_shot_no_image(lang)},
        ]
    if condition == "zero_shot_image_no_recognition":
        return [
            _system_message(lang, grounded=False),
            {"role": "user", "content": _user_prompt_image_no_recognition(lang, image_url)},
        ]
    if condition == "zero_shot_image_recognition":
        return [
            _system_message(lang, grounded=True),
            {"role": "user", "content": _user_prompt_zero_shot_image(lang, object_label, location, image_url)},
        ]
    if condition == "few_shot_cot_recognition":
        return [
            {"role": "system", "content": SYSTEM_PROMPT_GROUNDED_ID},
            {"role": "user", "content": _user_prompt_few_shot_cot(lang, object_label, location, examples, image_url)},
        ]
    raise ValueError(condition)


def _build_context(condition: str, obj: str, loc: str) -> str:
    if condition == "zero_shot_no_image":
        return "Tanpa gambar dan tanpa recognition (baseline teks)."
    if condition == "zero_shot_image_no_recognition":
        return "Gambar siswa (live); tanpa label recognition."
    parts = []
    if obj:
        parts.append(f"Objek: {obj}")
    if loc:
        parts.append(f"Lokasi: {loc}")
    return " | ".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default=str(SCRIPT_DIR / "live_ablation_cases.json"))
    parser.add_argument("-o", "--output", default=str(SCRIPT_DIR / "live_ablation.jsonl"))
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--few-shot-k", type=int, default=5)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=CONDITIONS,
        choices=CONDITIONS,
    )
    args = parser.parse_args()

    try:
        from openai import OpenAI  # type: ignore
    except ImportError:
        print("ERROR: pip install openai", file=sys.stderr)
        return 1

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        print("ERROR: OPENAI_API_KEY missing", file=sys.stderr)
        return 1
    client = OpenAI(api_key=api_key)

    data = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    items = list(data.get("items") or [])
    if args.max_cases:
        items = items[: args.max_cases]
    lang = str(data.get("lang", "id"))
    stage = str(data.get("stage", "problem_finding"))

    pool_for_fs = [{"id": it["id"], "references": it.get("references", [])} for it in items]
    rng = random.Random(args.seed)
    out_path = Path(args.output)
    written = 0

    with out_path.open("w", encoding="utf-8") as f:
        for tc in items:
            tc_id = tc["id"]
            obj = str(tc.get("object_label_id") or tc.get("object_label") or "objek")
            loc = str(tc.get("location") or "sekitar")
            image_url = tc.get("image_url")
            refs = tc.get("references") or []

            for condition in args.conditions:
                examples: List[str] = []
                if condition == "few_shot_cot_recognition":
                    examples = _pick_few_shot_examples(pool_for_fs, tc_id, args.few_shot_k, rng)

                use_image = condition != "zero_shot_no_image"
                img = image_url if use_image else None
                messages = _build_messages(condition, lang, obj, loc, img, examples)
                try:
                    inquiry = _call_openai_chat(
                        client, args.model, messages, args.temperature, 220,
                    )
                except RuntimeError as exc:
                    print(f"FAIL {tc_id}/{condition}: {exc}", file=sys.stderr)
                    continue

                row = {
                    "id": f"{tc_id}-{condition}",
                    "item_group_id": tc_id,
                    "stage": stage,
                    "lang": lang,
                    "prompt_type": condition,
                    "inquiry": inquiry,
                    "context": _build_context(condition, obj, loc),
                    "object_label": tc.get("object_label", ""),
                    "location": loc,
                    "image_url": image_url or "",
                    "image_used": bool(img),
                    "source": "live_firestore_photo",
                }
                if refs:
                    row["reference"] = refs
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
                print(f"OK {tc_id} {condition}: {inquiry[:70]}...", file=sys.stderr)
                time.sleep(0.25)

    print(f"Done. {written} rows -> {out_path}", file=sys.stderr)
    return 0 if written else 2


if __name__ == "__main__":
    sys.exit(main())
