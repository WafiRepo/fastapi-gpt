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
import csv
import json
import os
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


# Few-shot examples untuk generation
FEW_SHOT_EXAMPLES = [
    {
        "stage": "problem_finding",
        "topic": "centripetal acceleration",
        "input": "Think about daily life objects that move in circles.",
        "output": {"inquiry": "What everyday activity around you involves something moving in a circular path?"}
    },
    {
        "stage": "problem_finding",
        "topic": "centripetal acceleration",
        "input": "Imagine the string force suddenly disappears while an object is moving in a circle.",
        "output": {"inquiry": "What do you think happens to the object's motion if the force keeping it moving in a circle suddenly disappears?"}
    },
    {
        "stage": "problem_exploring",
        "topic": "centripetal acceleration",
        "input": "A ball is spun on a string at constant speed.",
        "output": {"inquiry": "What happens to the ball's motion when you analyze the relationship between speed and the required force?"}
    },
]

# Few-shot examples untuk full generation (inquiry + reference)
# Hanya untuk problem_finding dan problem_exploring
FEW_SHOT_FULL = [
    {
        "lang": "en",
        "stage": "problem_finding",
        "context": "Classroom with a ceiling fan",
        "inquiry": "Identify a real object around you that shows circular motion and name its center.",
        "reference": [
            "A ceiling fan shows circular motion as the blades rotate around the hub.",
            "The fan blades rotate in a circle about the central hub.",
            "What objects in your environment rotate around a fixed point?",
        ],
    },
    {
        "lang": "en",
        "stage": "problem_exploring",
        "context": "Spinning bicycle wheel",
        "inquiry": "What happens to the wheel's motion when you analyze the relationship between radius and centripetal acceleration?",
        "reference": [
            "How does changing the radius affect the centripetal acceleration of the spinning wheel?",
            "What relationship can you observe between the wheel's radius and its circular motion?",
            "Explore how the radius influences the forces acting on the spinning wheel.",
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
}

TOPIC = "centripetal acceleration"
AUDIENCE = "Grade 11"

# Path ke folder prompts
PROMPTS_DIR = Path(__file__).parent / "data" / "prompts"


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
    
    # Stage descriptions untuk dinamis (tidak digunakan, template sudah memiliki deskripsi)
    # stage_descriptions = {
    #     "problem_finding": "invites noticing/observing/curiosity, not calculation",
    #     "problem_exploring": "focuses on exploring and analyzing objects through hands-on experiments",
    # }
    
    # Replace placeholders
    prompt = template.replace("{STAGE}", stage)
    prompt = prompt.replace("{TOPIC}", topic)
    prompt = prompt.replace("{AUDIENCE}", audience)
    prompt = prompt.replace("{INPUT_TEXT}", input_text)
    
    # Replace stage description in template if needed
    # Template sudah memiliki semua stage descriptions, jadi tidak perlu replace
    
    return prompt


def build_few_shot_prompt(stage: str, topic: str, input_text: str) -> str:
    """Build few-shot prompt dari template file."""
    template = load_prompt_template("few_shot")
    
    # Replace placeholders
    prompt = template.replace("{STAGE}", stage)
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
    api_key: str,
    num_references: int = 3
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
    else:
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
    for attempt in range(4):
        try:
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
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0.2,
                },
                timeout=60,
            )
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


def generate_inquiry_only(
    model: str,
    payload: Dict[str, Any],
    api_key: str,
    prompt_type: str = "zero_shot"
) -> str:
    """
    Generate inquiry saja (tanpa reference).
    Reference sudah di-generate sebelumnya dan akan digunakan untuk semua strategies.
    Returns: inquiry question string.
    """
    stage = payload["stage"]
    topic = payload.get("topic", TOPIC)
    audience = payload.get("audience", AUDIENCE)
    context = payload.get("context", "")
    
    # Build prompt berdasarkan strategy
    if prompt_type == "zero_shot":
        user_prompt = build_zero_shot_prompt(stage, topic, audience, context)
    elif prompt_type == "few_shot":
        user_prompt = build_few_shot_prompt(stage, topic, context)
    elif prompt_type == "cot":
        user_prompt = build_cot_prompt(stage, topic, audience, context)
    elif prompt_type == "draft_critique_revise":
        user_prompt = build_draft_critique_revise_prompt(stage, topic, audience, context)
    else:
        raise ValueError(f"Unknown prompt type: {prompt_type}")
    
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
    else:
        system_msg = (
            "You are an educational designer for physics. "
            "Return JSON only with key: inquiry. "
            "The 'inquiry' is the generated inquiry question for students. "
            "Do NOT include 'reference' in your response, only 'inquiry'."
        )
    
    # Retry loop
    for attempt in range(4):
        try:
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
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0.2,
                },
                timeout=60,
            )
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
    num_references: int = 3
) -> None:
    """
    Generate items dengan semua prompt strategies untuk perbandingan.
    Reference di-generate sekali per stage (bukan per context).
    
    Args:
        num_references: Jumlah reference questions per stage (default: 3)
    """
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        print("\n" + "=" * 60)
        print("ERROR: OPENAI_API_KEY is not set!")
        print("=" * 60)
        print("\nCara mengatasi:")
        print("1. Buat file .env di folder project dengan isi:")
        print("   OPENAI_API_KEY=your-api-key-here")
        print("\n2. Atau set environment variable:")
        print("   Windows PowerShell:")
        print("   $env:OPENAI_API_KEY='your-api-key-here'")
        print("\n   Windows CMD:")
        print("   set OPENAI_API_KEY=your-api-key-here")
        print("\n   Linux/Mac:")
        print("   export OPENAI_API_KEY='your-api-key-here'")
        print("=" * 60 + "\n")
        raise RuntimeError("OPENAI_API_KEY is not set. Please set it in .env file or as environment variable.")
    
    stages = list(CONTEXTS.keys())
    langs = ["en"]  # Only English
    
    # Total items = reference generations + inquiry generations
    # Reference: sekali per stage (bukan per context)
    # Inquiry: sekali per (stage, context, strategy)
    total_references = len(stages) * len(langs)  # Sekali per stage
    total_inquiries = len(stages) * len(langs) * count_per_stage * len(prompt_strategies)
    total_items = total_references + total_inquiries
    current_item = 0
    
    # Dictionary untuk menyimpan reference per stage
    stage_references: Dict[str, List[str]] = {}
    
    with open(output_path, "w", encoding="utf-8") as f:
        idx = 1
        for stage in stages:
            contexts = CONTEXTS[stage][:count_per_stage]
            
            # Generate reference SEKALI per stage (bukan per context)
            for lang in langs:
                current_item += 1
                print(f"[{current_item}/{total_items}] Generating reference for {stage} {lang}...", end=" ", flush=True)
                try:
                    reference = generate_reference_only(
                        model=model,
                        stage=stage,
                        topic=TOPIC,
                        audience=AUDIENCE,
                        api_key=api_key,
                        num_references=num_references
                    )
                    stage_references[f"{stage}_{lang}"] = reference
                    print("[OK]")
                except Exception as exc:
                    print(f"[ERROR] {exc}")
                    print(f"[WARNING] Skipping stage {stage} due to reference generation failure")
                    stage_references[f"{stage}_{lang}"] = []  # Empty reference
                    continue
            
            # Generate inquiry untuk semua contexts dalam stage ini menggunakan reference yang sama
            for lang in langs:
                reference = stage_references.get(f"{stage}_{lang}", [])
                
                for i, context in enumerate(contexts):
                    # Generate inquiry dengan setiap prompt strategy menggunakan reference yang sama
                    for prompt_type in prompt_strategies:
                        current_item += 1
                        print(f"[{current_item}/{total_items}] Generating {stage} {lang} {prompt_type} context {i+1}...", end=" ", flush=True)
                        
                        payload = {
                            "id": f"{stage[:2].upper()}-{lang.upper()}-{idx:03d}-{prompt_type[:2]}",
                            "lang": lang,
                            "stage": stage,
                            "context": context,
                            "topic": TOPIC,
                            "audience": AUDIENCE,
                            "prompt_type": prompt_type
                        }
                        
                        try:
                            # Generate inquiry saja (reference sudah di-generate per stage)
                            inquiry = generate_inquiry_only(model, payload, api_key, prompt_type)
                            
                            # Validate that inquiry exists
                            if not inquiry:
                                raise ValueError("Generated inquiry is empty")
                            
                            item = {
                                "id": payload["id"],
                                "lang": payload["lang"],
                                "stage": payload["stage"],
                                "inquiry": inquiry,
                                "reference": reference,  # Reference sama untuk semua context dalam stage
                                "context": payload["context"],
                                "prompt_type": prompt_type
                            }
                            f.write(json.dumps(item, ensure_ascii=True) + "\n")
                            f.flush()
                            idx += 1
                            print("[OK]")
                        except Exception as exc:
                            print(f"[ERROR] {exc}")
                            time.sleep(1)
    
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
            "lang": item.get("lang", ""),
            "stage": item.get("stage", ""),
            "prompt_type": item.get("prompt_type", ""),
            "inquiry": item.get("inquiry", ""),
            "reference": reference_str,
            "context": item.get("context", ""),
        }
        csv_rows.append(row)
    
    # Write CSV with error handling
    fieldnames = ["id", "lang", "stage", "prompt_type", "inquiry", "reference", "context"]
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
    """
    def mean(values: List[float]) -> float:
        if not values:
            return 0.0
        return sum(values) / len(values)
    
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
        
        # Extract metrics
        metrics = {
            "bleu1": [],
            "bleu4": [],
            "rouge1": [],
            "rouge2": [],
            "rougeL": [],
            "bertscore_f1": [],
            "geval_overall": [],
        }
        
        for r in strategy_results:
            for metric in metrics.keys():
                val = r.get(metric)
                if isinstance(val, (int, float)):
                    metrics[metric].append(float(val))
        
        comparison[strategy] = {
            "bleu1_mean": mean(metrics["bleu1"]),
            "bleu4_mean": mean(metrics["bleu4"]),
            "rouge1_mean": mean(metrics["rouge1"]),
            "rouge2_mean": mean(metrics["rouge2"]),
            "rougeL_mean": mean(metrics["rougeL"]),
            "bertscore_f1_mean": mean(metrics["bertscore_f1"]),
            "geval_overall_mean": mean(metrics["geval_overall"]),
            "count": len(strategy_results)
        }
    
    return comparison


def run_eval(eval_script: str, input_path: str, bleu_scale: str, plots_dir: str, output_dir: str) -> int:
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
    ]
    return subprocess.call(cmd)


def complete_workflow(
    model: str = "gpt-4",
    count_per_stage: int = 10,
    prompt_strategies: List[str] = ["zero_shot", "few_shot", "cot", "draft_critique_revise"],
    output_dir: str = "data",
    eval_plots_dir: str = "eval_plots",
    bleu_scale: str = "0-100",
    skip_generation: bool = False,
    input_jsonl: Optional[str] = None,
    num_references: int = 3,
) -> Dict[str, Any]:
    """
    Workflow lengkap: Generation → Evaluation → Comparison
    
    Args:
        num_references: Jumlah reference questions per (stage, context) (default: 3)
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(eval_plots_dir, exist_ok=True)
    
    if skip_generation:
        if not input_jsonl:
            raise ValueError("--input-jsonl required when --skip-generation is set")
        output_jsonl = input_jsonl
    else:
        # 1. GENERATION
        output_jsonl = os.path.join(output_dir, "inquiry_samples_all_strategies.jsonl")
        print("=" * 60)
        print("Step 1: Generating inquiry samples with all prompt strategies...")
        print("=" * 60)
        generate_items_comparison(
            model=model,
            output_path=output_jsonl,
            count_per_stage=count_per_stage,
            prompt_strategies=prompt_strategies,
            num_references=num_references
        )
        print(f"\n✓ Generated samples saved to {output_jsonl}\n")
    
    # 2. EVALUATION
    print("=" * 60)
    print("Step 2: Evaluating with multiple metrics...")
    print("=" * 60)
    eval_results_csv = os.path.join(output_dir, "evaluation_results_all.csv")
    eval_results_jsonl = os.path.join(output_dir, "evaluation_results_all.jsonl")
    
    eval_script = os.path.join(os.path.dirname(__file__), "eval_inquiry_metrics.py")
    eval_exit_code = run_eval(
        eval_script=eval_script,
        input_path=output_jsonl,
        bleu_scale=bleu_scale,
        plots_dir=eval_plots_dir,
        output_dir=output_dir
    )
    
    if eval_exit_code != 0:
        print(f"⚠ Evaluation exited with code {eval_exit_code}")
        return {"error": "Evaluation failed"}
    
    print(f"\n✓ Evaluation results saved to:")
    print(f"  - {eval_results_csv}")
    print(f"  - {eval_results_jsonl}\n")
    
    # 3. COMPARISON
    print("=" * 60)
    print("Step 3: Comparing prompt strategies...")
    print("=" * 60)
    
    # Load evaluation results
    eval_results_path = eval_results_jsonl
    if not os.path.exists(eval_results_path):
        # Try CSV and convert
        print("⚠ JSONL results not found, skipping comparison")
        return {
            "generated_samples": output_jsonl,
            "evaluation_results": eval_results_csv,
        }
    
    comparison = compare_prompt_strategies(eval_results_path)
    
    comparison_file = os.path.join(output_dir, "prompt_strategy_comparison.json")
    with open(comparison_file, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=True)
    
    print(f"\n✓ Strategy comparison saved to {comparison_file}")
    print("\n" + "=" * 60)
    print("Comparison Results:")
    print("=" * 60)
    print(json.dumps(comparison, indent=2, ensure_ascii=True))
    print("=" * 60 + "\n")
    
    return {
        "generated_samples": output_jsonl,
        "evaluation_results": eval_results_csv,
        "evaluation_results_jsonl": eval_results_jsonl,
        "comparison": comparison_file,
        "comparison_data": comparison
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Complete workflow: Generate with multiple prompt strategies and evaluate"
    )
    parser.add_argument("--model", default="gpt-4", help="GPT model to use")
    parser.add_argument("--count-per-stage", type=int, default=10, help="Items per stage per language")
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["zero_shot", "few_shot", "cot", "draft_critique_revise"],
        choices=["zero_shot", "few_shot", "cot", "draft_critique_revise"],
        help="Prompt strategies to use"
    )
    parser.add_argument("--output-dir", default="data", help="Output directory")
    parser.add_argument("--plots-dir", default="eval_plots", help="Plots directory")
    parser.add_argument("--bleu-scale", default="0-100", choices=["0-1", "0-100"], help="BLEU scale")
    parser.add_argument("--num-references", type=int, default=3, help="Number of reference questions per (stage, context). Default: 3. Recommended: 3-5 for optimal reliability.")
    parser.add_argument("--skip-generation", action="store_true", help="Skip generation, only evaluate")
    parser.add_argument("--input-jsonl", help="Input JSONL for evaluation (if skipping generation)")
    parser.add_argument("--no-eval", action="store_true", help="Skip evaluation")
    
    args = parser.parse_args()
    
    if args.no_eval:
        # Only generation
        output_jsonl = os.path.join(args.output_dir, "inquiry_samples_all_strategies.jsonl")
        os.makedirs(args.output_dir, exist_ok=True)
        print("Generating inquiry samples...")
        generate_items_comparison(
            model=args.model,
            output_path=output_jsonl,
            count_per_stage=args.count_per_stage,
            prompt_strategies=args.strategies,
            num_references=args.num_references
        )
        print(f"\n✓ Generated samples saved to {output_jsonl}")
        return 0
    
    # Complete workflow
    result =         complete_workflow(
            model=args.model,
            count_per_stage=args.count_per_stage,
            prompt_strategies=args.strategies,
            output_dir=args.output_dir,
            eval_plots_dir=args.plots_dir,
            bleu_scale=args.bleu_scale,
            skip_generation=args.skip_generation,
            input_jsonl=args.input_jsonl,
            num_references=args.num_references,
        )
    
    if "error" in result:
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
