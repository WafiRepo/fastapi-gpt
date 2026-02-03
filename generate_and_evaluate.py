#!/usr/bin/env python3
"""
Generate inquiry data with GPT-4 and run evaluation.

This script overwrites the output JSONL file (default inquiry_samples.jsonl).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List

import requests  # type: ignore


FEW_SHOT = [
    {
        "lang": "en",
        "stage": "problem_finding",
        "context": "Classroom with a ceiling fan",
        "difficulty": "easy",
        "inquiry": "Identify a real object around you that shows circular motion and name its center.",
        "candidate": "A ceiling fan rotates around a fixed axis, so its center is the hub.",
        "reference": [
            "A ceiling fan shows circular motion as the blades rotate around the hub.",
            "The fan blades rotate in a circle about the central hub.",
        ],
    },
    {
        "lang": "id",
        "stage": "problem_exploring",
        "context": "Roda sepeda berputar",
        "difficulty": "easy",
        "inquiry": "Jelaskan arah percepatan pada roda berputar dan kaitannya dengan pusat lingkaran.",
        "candidate": "Percepatan mengarah ke pusat roda sehingga benda tetap bergerak melingkar.",
        "reference": [
            "Percepatan sentripetal selalu menuju pusat lingkaran.",
            "Arah percepatan pada gerak melingkar selalu ke pusat.",
        ],
    },
    {
        "lang": "en",
        "stage": "problem_generating",
        "context": "Washer on a string, measured r and w",
        "difficulty": "easy",
        "inquiry": "Using the measured radius and angular speed, ask a question to compute centripetal acceleration.",
        "candidate": "If r = 0.20 m and w = 8 rad/s, what is the centripetal acceleration?",
        "reference": [
            "Given r and w, ask for a using a = r * w^2.",
            "Form a question to compute a from r and w.",
        ],
    },
    {
        "lang": "id",
        "stage": "problem_solving",
        "context": "Komidi putar, r=2.0 m, w=1.5 rad/s",
        "difficulty": "easy",
        "inquiry": "Hitung percepatan sentripetal berdasarkan data r dan w yang diberikan.",
        "candidate": "a = r * w^2 = 2.0 * 2.25 = 4.5 m/s2.",
        "reference": [
            "a = r * w^2 sehingga a = 4.5 m/s2.",
            "Percepatan sentripetalnya 4.5 m/s2 dari a = r * w^2.",
        ],
    },
]


CONTEXTS = {
    "problem_finding": [
        "Classroom with a ceiling fan",
        "Bicycle wheel spinning outside",
        "Washing machine drum in motion",
        "Playground carousel rotating",
        "Desk fan on a table",
        "Blender blade in a kitchen",
        "Yo-yo swinging on a finger",
        "Turntable platter spinning",
        "Toy car tied to a string",
        "Office chair wheel rotating",
    ],
    "problem_exploring": [
        "Rotating fan blades",
        "Spinning bicycle wheel",
        "Washer on a string",
        "Turntable",
        "Rotating toy",
        "Experiment data: a vs r graph",
        "Measured string length",
        "Tension in a string setup",
        "Sensor log with stable period",
        "Rotating disk",
    ],
    "problem_generating": [
        "Fan with measured r and w",
        "Bicycle wheel r and w",
        "Washer on string with a and w",
        "Graph of a vs w",
        "Two rotation cases",
        "Carousel with r and w",
        "Turntable with period T",
        "Measurement uncertainty in r",
        "Constant w with two radii",
        "Sensor data with a and r",
    ],
    "problem_solving": [
        "Given r and w",
        "Given a and w",
        "Turntable with r and T",
        "Wheel with r and w",
        "Two cases for comparison",
        "Given a and r",
        "Constant w with two radii",
        "Carousel with r and w",
        "Uncertainty in r",
        "Given r and w for v",
    ],
}


DIFFICULTY_CYCLE = ["easy", "intermediate", "advanced", "easy", "intermediate", "advanced", "easy", "easy", "intermediate", "advanced"]


def call_openai(model: str, payload: Dict[str, Any], api_key: str) -> Dict[str, Any]:
    system_msg = (
        "You are an educational designer for physics. "
        "Return JSON only with keys: inquiry, candidate, reference."
    )
    prompt = {
        "task": "Generate an inquiry prompt and example candidate + reference answers.",
        "topic": "centripetal acceleration / circular motion",
        "requirements": [
            "1 concise inquiry prompt (1-2 sentences)",
            "candidate is a plausible response",
            "reference is a list of 2 gold responses",
            "use the provided context",
            "align with the stage goal",
        ],
        "input": payload,
        "few_shot": FEW_SHOT,
    }

    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=True)},
            ],
            "temperature": 0.2,
        },
        timeout=60,
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        raise RuntimeError("Invalid JSON from model.")


def generate_items(model: str, output_path: str, count_per_stage: int) -> None:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    stages = list(CONTEXTS.keys())
    langs = ["en", "id"]

    with open(output_path, "w", encoding="utf-8") as f:
        idx = 1
        for stage in stages:
            contexts = CONTEXTS[stage][:count_per_stage]
            for lang in langs:
                for i, context in enumerate(contexts):
                    difficulty = DIFFICULTY_CYCLE[i % len(DIFFICULTY_CYCLE)] if stage in ("problem_generating", "problem_solving") else "easy"
                    payload = {
                        "id": f"{stage[:2].upper()}-{lang.upper()}-{idx:03d}",
                        "lang": lang,
                        "stage": stage,
                        "context": context if lang == "en" else context,  # context can be localized later
                        "difficulty": difficulty,
                    }
                    # Retry loop
                    for attempt in range(4):
                        try:
                            data = call_openai(model, payload, api_key)
                            item = {
                                "id": payload["id"],
                                "lang": payload["lang"],
                                "stage": payload["stage"],
                                "inquiry": data.get("inquiry", ""),
                                "candidate": data.get("candidate", ""),
                                "reference": data.get("reference", []),
                                "context": payload["context"],
                                "difficulty": payload["difficulty"],
                            }
                            f.write(json.dumps(item, ensure_ascii=True) + "\n")
                            idx += 1
                            break
                        except Exception as exc:
                            if attempt == 3:
                                raise
                            time.sleep(2 ** attempt)


def run_eval(eval_script: str, input_path: str, bleu_scale: str, plots_dir: str) -> int:
    cmd = [
        sys.executable,
        eval_script,
        "--input",
        input_path,
        "--bleu-scale",
        bleu_scale,
        "--plots-dir",
        plots_dir,
    ]
    return subprocess.call(cmd)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate inquiry data and evaluate.")
    parser.add_argument("--output", default="fastapi-gpt/data/inquiry_samples.jsonl", help="Output JSONL path.")
    parser.add_argument("--count-per-stage", type=int, default=10, help="Items per stage per language.")
    parser.add_argument("--model", default="gpt-4", help="Model for inquiry generation.")
    parser.add_argument("--bleu-scale", default="0-100", choices=["0-1", "0-100"], help="BLEU scale.")
    parser.add_argument("--plots-dir", default="eval_plots", help="PNG output directory.")
    parser.add_argument("--no-eval", action="store_true", help="Skip evaluation.")
    args = parser.parse_args()

    generate_items(args.model, args.output, args.count_per_stage)

    if not args.no_eval:
        eval_script = os.path.join(os.path.dirname(__file__), "eval_inquiry_metrics.py")
        return run_eval(eval_script, args.output, args.bleu_scale, args.plots_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
