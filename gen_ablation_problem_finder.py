#!/usr/bin/env python3
"""
Ablasi prompt untuk inquiry "problem finding" (centripetal acceleration).
Desain empat kondisi:

  A. zero_shot_no_image   : BASELINE. Tanpa gambar, tanpa hasil recognition,
                            tanpa nama objek/lokasi. Prompt eksplisit melarang
                            menyebut objek atau lokasi tertentu. Output sengaja
                            generik.
  B. zero_shot_image      : Zero-shot + gambar + recognition (objek & lokasi).
                            Multimodal gpt-4o; wajib menyebut nama objek &
                            lokasi.
  C. few_shot             : Few-shot + gambar + recognition. Sama seperti B,
                            tetapi diberi tiga contoh inquiry dari test case
                            LAIN sebagai panduan gaya (tidak bocor reference
                            sendiri).
  D. cot                  : Chain-of-Thought + gambar + recognition. Sama seperti
                            B, tetapi model diminta menalar langkah demi langkah
                            secara internal sebelum menulis satu pertanyaan
                            final.

JSONL output siap dievaluasi oleh `eval_inquiry_metrics.py` (BLEU/ROUGE/BERTScore
+ G-eval). Field `reference` diisi per-item dari test_cases_pf.json sehingga
text-similarity menilai apakah inquiry yang dihasilkan mendekati referensi
spesifik objek + lokasi.

Contoh:
  python gen_ablation_problem_finder.py \
    --test-cases ablation/test_cases_pf.json \
    --output ablation/ablation_pf.jsonl \
    --model gpt-4o --variants 1

Evaluasi & plot:
  python eval_inquiry_metrics.py --input ablation/ablation_pf.jsonl \
    --output-jsonl ablation/ablation_results.jsonl \
    --output-csv ablation/ablation_results.csv \
    --plots-dir ablation/eval_plots --bleu-scale 0-100
  python plot_ablation.py --input ablation/ablation_results.csv \
    --input-jsonl ablation/ablation_results.jsonl \
    --output-dir ablation/eval_plots
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


SCRIPT_DIR = Path(__file__).resolve().parent

try:
    from dotenv import load_dotenv  # type: ignore

    env_path = SCRIPT_DIR / ".env"
    load_dotenv(env_path if env_path.exists() else None)
except ImportError:
    pass


def _resolve_path(p: str) -> Path:
    candidate = Path(p)
    if not candidate.is_absolute():
        candidate = SCRIPT_DIR / candidate
    return candidate


def _encode_image_data_url(path: Path) -> Optional[str]:
    if not path.exists():
        print(f"WARNING: image not found, skipping image: {path}", file=sys.stderr)
        return None
    mime, _ = mimetypes.guess_type(str(path))
    if not mime:
        mime = "image/jpeg"
    b = path.read_bytes()
    return f"data:{mime};base64,{base64.b64encode(b).decode('ascii')}"


SYSTEM_PROMPT_BASELINE_ID = (
    "Anda adalah guru fisika SMA yang memandu siswa Kelas 11 untuk "
    "mengidentifikasi objek di sekitar mereka yang menunjukkan percepatan sentripetal. "
    "Anda BELUM melihat lingkungan siswa sama sekali. "
    "Tulis SATU pertanyaan inquiry singkat dan jelas yang memicu rasa ingin tahu, "
    "TANPA menyebut nama objek tertentu dan TANPA menyebut lokasi tertentu. "
    "Keluarkan HANYA teks pertanyaan, satu kalimat."
)

SYSTEM_PROMPT_GROUNDED_ID = (
    "Anda adalah guru fisika SMA yang memandu siswa Kelas 11 untuk "
    "mengidentifikasi objek di sekitar mereka yang menunjukkan percepatan sentripetal. "
    "Anda telah memperoleh: (1) gambar dari kamera siswa, (2) hasil pengenalan objek (recognition), "
    "dan (3) lokasi pengamatan. "
    "WAJIB: pertanyaan harus MENYEBUT nama objek yang dikenali DAN lokasi yang diberikan secara eksplisit. "
    "WAJIB: gunakan notasi (r) untuk jari-jari, (\u03c9) untuk kecepatan sudut, dan (a) untuk percepatan sentripetal. "
    "Hindari mengungkap rumus a = r\u00b7\u03c9\u00b2. "
    "Tulis SATU pertanyaan inquiry singkat, jelas, dan membangkitkan rasa ingin tahu/pengamatan. "
    "Keluarkan HANYA teks pertanyaan, satu kalimat."
)

SYSTEM_PROMPT_BASELINE_EN = (
    "You are a high school physics teacher guiding a Grade 11 student to "
    "identify everyday objects that demonstrate centripetal acceleration. "
    "You have NOT seen the student's environment. "
    "Write ONE short, clear inquiry question that invites curiosity, WITHOUT naming "
    "any specific object and WITHOUT mentioning any specific location. "
    "Output ONLY the question text, a single sentence."
)

SYSTEM_PROMPT_GROUNDED_EN = (
    "You are a high school physics teacher guiding a Grade 11 student to "
    "identify everyday objects that demonstrate centripetal acceleration. "
    "You have: (1) the student's camera image, (2) an object recognition label, "
    "and (3) the observation location. "
    "MANDATORY: the inquiry MUST mention BOTH the recognized object name AND the location explicitly. "
    "MANDATORY: use the notation (r) for radius, (\u03c9) for angular velocity, and (a) for centripetal acceleration. "
    "Do not reveal the formula a = r\u00b7\u03c9\u00b2. "
    "Output ONLY the question text, a single sentence."
)


def _system_message(lang: str, grounded: bool) -> Dict[str, Any]:
    if lang == "id":
        return {"role": "system", "content": SYSTEM_PROMPT_GROUNDED_ID if grounded else SYSTEM_PROMPT_BASELINE_ID}
    return {"role": "system", "content": SYSTEM_PROMPT_GROUNDED_EN if grounded else SYSTEM_PROMPT_BASELINE_EN}


def _user_prompt_zero_shot_no_image(lang: str) -> List[Dict[str, Any]]:
    if lang == "id":
        text = (
            "Topik: percepatan sentripetal.\n"
            "Buat satu pertanyaan inquiry umum yang mengajak siswa mencari benda di sekitar "
            "yang menunjukkan gerakan melingkar.\n"
            "JANGAN menyebut nama objek atau lokasi tertentu.\n"
            "Keluarkan HANYA satu kalimat pertanyaan."
        )
    else:
        text = (
            "Topic: centripetal acceleration.\n"
            "Generate one general inquiry question asking the student to find any object "
            "around them that exhibits circular motion.\n"
            "DO NOT name any specific object or location.\n"
            "Output ONLY a single inquiry sentence."
        )
    return [{"type": "text", "text": text}]


def _user_prompt_zero_shot_image(
    lang: str, object_label: str, location: str, image_data_url: Optional[str]
) -> List[Dict[str, Any]]:
    if lang == "id":
        text = (
            f"Topik: percepatan sentripetal.\n"
            f"Hasil pengenalan objek (recognition): {object_label}.\n"
            f"Lokasi pengamatan: {location}.\n\n"
            f"Lihat gambar; buat SATU pertanyaan inquiry yang WAJIB:\n"
            f"  - menyebut \"{object_label}\" secara eksplisit,\n"
            f"  - menyebut \"{location}\" secara eksplisit,\n"
            f"  - menggunakan notasi (r), (\u03c9), dan/atau (a),\n"
            f"  - mengajak siswa mengamati gerakan melingkarnya."
        )
    else:
        text = (
            f"Topic: centripetal acceleration.\n"
            f"Detected object: {object_label}. Location: {location}.\n\n"
            f"Look at the image; write ONE inquiry question that MUST:\n"
            f"  - mention \"{object_label}\" explicitly,\n"
            f"  - mention \"{location}\" explicitly,\n"
            f"  - use the notation (r), (\u03c9), and/or (a),\n"
            f"  - invite observation of its circular motion."
        )
    parts: List[Dict[str, Any]] = []
    if image_data_url:
        parts.append({"type": "image_url", "image_url": {"url": image_data_url}})
    parts.append({"type": "text", "text": text})
    return parts


def _user_prompt_few_shot(
    lang: str, object_label: str, location: str, examples: List[str], image_data_url: Optional[str]
) -> List[Dict[str, Any]]:
    bullets = "\n".join(f"- {e}" for e in examples) if examples else "-"
    if lang == "id":
        text = (
            f"Topik: percepatan sentripetal.\n"
            f"Hasil pengenalan objek (recognition): {object_label}.\n"
            f"Lokasi pengamatan: {location}.\n\n"
            f"Contoh GAYA pertanyaan inquiry yang baik (untuk objek/lokasi lain — gunakan polanya saja, jangan disalin):\n"
            f"{bullets}\n\n"
            f"Sekarang lihat gambar; tulis SATU pertanyaan inquiry baru tentang \"{object_label}\" di \"{location}\". "
            f"WAJIB menyebut \"{object_label}\" dan \"{location}\" secara eksplisit serta memakai notasi (r), (\u03c9), dan/atau (a). "
            f"Keluarkan HANYA satu kalimat pertanyaan."
        )
    else:
        text = (
            f"Topic: centripetal acceleration.\n"
            f"Detected object: {object_label}. Location: {location}.\n\n"
            f"Style examples of good inquiry questions (for OTHER objects/locations — use the pattern only, do not copy):\n"
            f"{bullets}\n\n"
            f"Now look at the image; write ONE new inquiry about \"{object_label}\" at \"{location}\". "
            f"MUST mention \"{object_label}\" and \"{location}\" explicitly and use (r), (\u03c9), and/or (a). "
            f"Output ONLY a single inquiry sentence."
        )
    parts: List[Dict[str, Any]] = []
    if image_data_url:
        parts.append({"type": "image_url", "image_url": {"url": image_data_url}})
    parts.append({"type": "text", "text": text})
    return parts


def _user_prompt_cot(
    lang: str, object_label: str, location: str, image_data_url: Optional[str]
) -> List[Dict[str, Any]]:
    if lang == "id":
        text = (
            f"Topik: percepatan sentripetal.\n"
            f"Hasil pengenalan objek (recognition): {object_label}.\n"
            f"Lokasi pengamatan: {location}.\n\n"
            f"Pikirkan langkah demi langkah SECARA INTERNAL (jangan ditampilkan):\n"
            f"  1) Bagian mana dari {object_label} yang menunjukkan gerak melingkar?\n"
            f"  2) Bagaimana jari-jari (r) dan kecepatan sudut (\u03c9) terlihat pada gambar di {location}?\n"
            f"  3) Apa yang sebaiknya diamati siswa untuk memahami percepatan sentripetal (a)?\n"
            f"  4) Susun pertanyaan terbuka yang menyebut \"{object_label}\" dan \"{location}\" secara eksplisit.\n\n"
            f"JANGAN tampilkan langkah berpikirmu. Keluarkan HANYA SATU pertanyaan inquiry final dalam satu kalimat, "
            f"WAJIB menyebut \"{object_label}\" dan \"{location}\" serta memakai (r), (\u03c9), dan/atau (a)."
        )
    else:
        text = (
            f"Topic: centripetal acceleration.\n"
            f"Detected object: {object_label}. Location: {location}.\n\n"
            f"Reason step-by-step INTERNALLY (do not show):\n"
            f"  1) Which part of {object_label} exhibits circular motion?\n"
            f"  2) How are (r) and (\u03c9) visible in the image at {location}?\n"
            f"  3) What should the student observe to understand (a)?\n"
            f"  4) Compose an open-ended question that explicitly mentions \"{object_label}\" and \"{location}\".\n\n"
            f"DO NOT show your reasoning. Output ONLY ONE final inquiry sentence "
            f"that explicitly mentions \"{object_label}\" and \"{location}\" and uses (r), (\u03c9), and/or (a)."
        )
    parts: List[Dict[str, Any]] = []
    if image_data_url:
        parts.append({"type": "image_url", "image_url": {"url": image_data_url}})
    parts.append({"type": "text", "text": text})
    return parts


def _pick_few_shot_examples(
    all_items: List[Dict[str, Any]], current_id: str, k: int, rng: random.Random
) -> List[str]:
    """Ambil k contoh inquiry dari referensi test case LAIN (bukan current_id)."""
    pool: List[str] = []
    for it in all_items:
        if it.get("id") == current_id:
            continue
        refs = it.get("references") or []
        for r in refs:
            if r:
                pool.append(str(r))
    rng.shuffle(pool)
    return pool[:k]


def _build_messages(
    condition: str,
    lang: str,
    object_label: str,
    location: str,
    image_data_url: Optional[str],
    examples: List[str],
) -> List[Dict[str, Any]]:
    if condition == "zero_shot_no_image":
        return [_system_message(lang, grounded=False), {"role": "user", "content": _user_prompt_zero_shot_no_image(lang)}]
    if condition == "zero_shot_image":
        return [_system_message(lang, grounded=True), {"role": "user", "content": _user_prompt_zero_shot_image(lang, object_label, location, image_data_url)}]
    if condition == "few_shot":
        return [_system_message(lang, grounded=True), {"role": "user", "content": _user_prompt_few_shot(lang, object_label, location, examples, image_data_url)}]
    if condition == "cot":
        return [_system_message(lang, grounded=True), {"role": "user", "content": _user_prompt_cot(lang, object_label, location, image_data_url)}]
    raise ValueError(f"Unknown condition: {condition}")


def _call_openai_chat(
    client: Any,
    model: str,
    messages: List[Dict[str, Any]],
    temperature: float,
    max_tokens: int,
    retries: int = 3,
) -> str:
    last_err: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            text = resp.choices[0].message.content.strip()
            return text
        except Exception as exc:  # pragma: no cover
            last_err = exc
            wait_s = min(30, 2 ** (attempt - 1))
            print(
                f"WARNING: LLM call failed ({type(exc).__name__}: {exc}); retry in {wait_s}s",
                file=sys.stderr,
            )
            time.sleep(wait_s)
    raise RuntimeError(f"LLM call failed after {retries} attempts: {last_err}")


def _build_context(condition: str, object_label_local: str, location: str) -> str:
    if condition == "zero_shot_no_image":
        return "Tanpa konteks objek/lokasi (baseline)."
    parts: List[str] = []
    if object_label_local:
        parts.append(f"Objek: {object_label_local}")
    if location:
        parts.append(f"Lokasi: {location}")
    return " | ".join(parts) if parts else ""


def main() -> int:
    parser = argparse.ArgumentParser(description="Ablasi prompt problem-finder via LLM langsung.")
    parser.add_argument(
        "--test-cases",
        default="ablation/test_cases_pf.json",
        help="Path JSON daftar test case (dengan field references per item).",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="ablation/ablation_pf.jsonl",
        help="Path JSONL output.",
    )
    parser.add_argument("--model", default="gpt-4o", help="Model untuk semua kondisi (default: gpt-4o).")
    parser.add_argument(
        "--variants",
        type=int,
        default=1,
        help="Jumlah pengulangan per (test case \u00d7 kondisi) (default: 1).",
    )
    parser.add_argument("--temperature", type=float, default=0.5, help="Sampling temperature (default: 0.5).")
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["zero_shot_no_image", "zero_shot_image", "few_shot", "cot"],
        choices=["zero_shot_no_image", "zero_shot_image", "few_shot", "cot"],
        help="Pilih kondisi yang dijalankan (default: keempat-empatnya).",
    )
    parser.add_argument("--max-tokens", type=int, default=180)
    parser.add_argument("--lang-override", choices=["id", "en"], help="Paksa bahasa output.")
    parser.add_argument("--seed", type=int, default=42, help="Seed untuk pemilihan few-shot examples.")
    parser.add_argument("--few-shot-k", type=int, default=3, help="Jumlah few-shot examples (default: 3).")
    parser.add_argument(
        "--no-image-bcd",
        action="store_true",
        help="Debug: paksa kondisi B/C/D tidak mengirim gambar (text-only).",
    )
    args = parser.parse_args()

    try:
        from openai import OpenAI  # type: ignore
    except ImportError:
        print("ERROR: install 'openai' (>=1.0) -- sudah ada di requirements.txt.", file=sys.stderr)
        return 1

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        print("ERROR: OPENAI_API_KEY belum di-set di .env.", file=sys.stderr)
        return 1
    client = OpenAI(api_key=api_key)

    tc_path = _resolve_path(args.test_cases)
    if not tc_path.exists():
        print(f"ERROR: test cases tidak ditemukan: {tc_path}", file=sys.stderr)
        return 1
    tc_data = json.loads(tc_path.read_text(encoding="utf-8"))
    stage = str(tc_data.get("stage", "problem_finding"))
    lang_default = args.lang_override or str(tc_data.get("lang", "id"))
    items_in: List[Dict[str, Any]] = list(tc_data.get("items") or [])
    if not items_in:
        print("ERROR: test_cases.items kosong.", file=sys.stderr)
        return 1

    out_path = _resolve_path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    written = 0
    with out_path.open("w", encoding="utf-8") as f:
        for tc in items_in:
            tc_id = str(tc["id"])
            object_label = str(tc.get("object_label", "")).strip()
            object_label_id = str(tc.get("object_label_id", object_label)).strip()
            location = str(tc.get("location", "")).strip()
            image_field = tc.get("image")
            image_data_url = None
            if image_field:
                image_data_url = _encode_image_data_url(_resolve_path(image_field))

            item_refs = [str(r) for r in (tc.get("references") or []) if r]

            for condition in args.conditions:
                use_image = (
                    condition in {"zero_shot_image", "few_shot", "cot"} and not args.no_image_bcd
                )
                examples: List[str] = []
                if condition == "few_shot":
                    examples = _pick_few_shot_examples(items_in, tc_id, args.few_shot_k, rng)

                for variant in range(1, args.variants + 1):
                    messages = _build_messages(
                        condition=condition,
                        lang=lang_default,
                        object_label=object_label_id if lang_default == "id" else object_label,
                        location=location,
                        image_data_url=image_data_url if use_image else None,
                        examples=examples,
                    )
                    try:
                        inquiry_text = _call_openai_chat(
                            client=client,
                            model=args.model,
                            messages=messages,
                            temperature=args.temperature,
                            max_tokens=args.max_tokens,
                        )
                    except RuntimeError as exc:
                        print(f"ERROR generating {tc_id}/{condition}/v{variant}: {exc}", file=sys.stderr)
                        continue

                    item_id = (
                        f"{tc_id}-{condition}"
                        if args.variants == 1
                        else f"{tc_id}-{condition}-v{variant}"
                    )
                    row: Dict[str, Any] = {
                        "id": item_id,
                        "item_group_id": tc_id,
                        "stage": stage,
                        "lang": lang_default,
                        "prompt_type": condition,
                        "inquiry": inquiry_text,
                        "context": _build_context(condition, object_label_id, location),
                        "object_label": object_label,
                        "object_label_id": object_label_id,
                        "location": location,
                        "image": str(image_field) if image_field else "",
                        "image_used": bool(use_image and image_data_url),
                        "variant": variant,
                    }
                    if item_refs:
                        row["reference"] = item_refs
                    f.write(json.dumps(row, ensure_ascii=True) + "\n")
                    written += 1
                    img_tag = "[img]" if use_image and image_data_url else "[txt]"
                    print(
                        f"OK {tc_id} {condition} {img_tag} v{variant}: {inquiry_text[:80]}",
                        file=sys.stderr,
                    )

    print(f"\nDone. Wrote {written} item(s) -> {out_path}", file=sys.stderr)
    return 0 if written > 0 else 2


if __name__ == "__main__":
    sys.exit(main())
