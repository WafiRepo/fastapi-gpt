#!/usr/bin/env python3
"""
Script untuk membuat plot dari hasil evaluasi yang sudah ada (CSV atau JSONL).
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError as e:
    print(f"Error: Missing required library. Install with: pip install pandas matplotlib numpy")
    sys.exit(1)


def load_results(file_path: str) -> List[Dict[str, Any]]:
    """Load results from CSV or JSONL file."""
    if file_path.endswith('.jsonl'):
        items = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                items.append(json.loads(line))
        return items
    elif file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
        return df.to_dict('records')
    else:
        raise ValueError(f"Unsupported file format: {file_path}")


def create_combined_comparison_plot(results: List[Dict[str, Any]], output_path: str) -> None:
    """
    Membuat combined comparison plot seperti comparison_plot_test.png
    Menampilkan perbandingan metrics berdasarkan prompt_type dan stage.
    """
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Check required columns
    if 'stage' not in df.columns or 'prompt_type' not in df.columns:
        print("Warning: 'stage' or 'prompt_type' not found. Skipping comparison plot.")
        return
    
    # Auto-detect metrics (exclude non-metric columns)
    exclude_cols = ['stage', 'prompt_type', 'id', 'lang', 'context']
    metrics = [col for col in df.columns if col not in exclude_cols]
    
    # Filter only numeric metrics
    available_metrics = []
    for m in metrics:
        if m in df.columns:
            # Check if column has numeric values
            numeric_values = pd.to_numeric(df[m], errors='coerce').dropna()
            if len(numeric_values) > 0:
                available_metrics.append(m)
    
    if not available_metrics:
        print("Error: No metrics found in data")
        return
    
    print(f"Creating comparison plot with metrics: {available_metrics}")
    
    # Group by stage dan prompt_type, calculate mean
    grouped = df.groupby(['stage', 'prompt_type'])[available_metrics].mean().reset_index()
    
    stages = sorted(grouped['stage'].unique())
    prompt_types = sorted(grouped['prompt_type'].unique())
    
    prompt_labels = {
        'zero_shot': 'Zero-Shot',
        'few_shot': 'Few-Shot',
        'cot': 'CoT',
        'draft_critique_revise': 'Draft-Critique-Revise'
    }
    
    stage_labels = {
        'problem_finding': 'Problem Finding',
        'problem_exploring': 'Problem Exploring'
    }
    
    # Create figure
    n_metrics = len(available_metrics)
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    if n_metrics == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for idx, metric in enumerate(available_metrics):
        ax = axes[idx]
        
        x = np.arange(len(stages))
        width = 0.2
        multiplier = 0
        
        for prompt_type in prompt_types:
            values = []
            for stage in stages:
                row = grouped[(grouped['stage'] == stage) & (grouped['prompt_type'] == prompt_type)]
                if not row.empty:
                    val = row[metric].values[0]
                    # Handle NaN
                    if pd.isna(val):
                        values.append(0)
                    else:
                        values.append(float(val))
                else:
                    values.append(0)
            
            offset = width * multiplier
            bars = ax.bar(x + offset, values, width, label=prompt_labels.get(prompt_type, prompt_type),
                         color=colors[multiplier % len(colors)], alpha=0.8, edgecolor='black', linewidth=0.5)
            
            # Value labels
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}',
                           ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            multiplier += 1
        
        ax.set_xlabel('Stage', fontsize=11, fontweight='bold')
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=10)
        ax.set_title(f'{metric.replace("_", " ").title()}', fontsize=12, fontweight='bold')
        ax.set_xticks(x + width * (len(prompt_types) - 1) / 2)
        ax.set_xticklabels([stage_labels.get(s, s) for s in stages], fontsize=10)
        ax.legend(loc='best', fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax.set_ylim(0, 1)
    
    # Hide unused subplots
    for idx in range(len(available_metrics), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Evaluation Metrics Comparison: Prompt Strategies vs Stages', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Created: {output_path}")
    plt.close()


def create_plots_from_results(results: List[Dict[str, Any]], output_dir: str) -> None:
    """Create plots from evaluation results."""
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

    # Individual metric curves
    for key in metric_keys:
        values: List[Optional[float]] = []
        for row in results:
            val = row.get(key)
            if isinstance(val, (int, float)) and not (isinstance(val, float) and np.isnan(val)):
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
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        filename = os.path.join(output_dir, f"{key}_curve.png")
        plt.savefig(filename, dpi=150)
        plt.close()
        print(f"Created: {filename}")

    # Combined plot
    bleu_keys = ["bleu1", "bleu4"]
    other_keys = ["rouge1", "rouge2", "rougeL", "bertscore_f1", "geval_overall"]

    def series_for(keys: List[str]) -> Dict[str, List[float]]:
        out: Dict[str, List[float]] = {}
        for key in keys:
            values: List[Optional[float]] = []
            for row in results:
                val = row.get(key)
                if isinstance(val, (int, float)) and not (isinstance(val, float) and np.isnan(val)):
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
        print(f"Created: {combined_path}")

    # Summary plot
    metrics = [
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
    
    summary_values = []
    summary_labels = []
    for metric in metrics:
        values = []
        for row in results:
            val = row.get(metric)
            if isinstance(val, (int, float)) and not (isinstance(val, float) and np.isnan(val)):
                values.append(float(val))
        if values:
            avg = sum(values) / len(values)
            summary_values.append(avg)
            summary_labels.append(metric.replace("_", " ").title())

    if summary_values:
        plt.figure(figsize=(10, 4))
        plt.plot(summary_labels, summary_values, marker="o", linewidth=1)
        plt.title("Summary metrics (line plot)")
        plt.ylabel("Score")
        plt.ylim(0, 1)
        plt.xticks(rotation=20, ha="right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        summary_path = os.path.join(output_dir, "summary_metrics.png")
        plt.savefig(summary_path, dpi=150)
        plt.close()
        print(f"Created: {summary_path}")
    
    # Create combined comparison plot
    comparison_path = os.path.join(output_dir, "comparison_plot.png")
    create_combined_comparison_plot(results, comparison_path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create plots from evaluation results (CSV or JSONL)"
    )
    parser.add_argument("--input", required=True, help="Input CSV or JSONL file with evaluation results")
    parser.add_argument("--output-dir", default="eval_plots", help="Output directory for plots")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: File not found: {args.input}")
        return 1
    
    try:
        print(f"Loading results from: {args.input}")
        results = load_results(args.input)
        print(f"Loaded {len(results)} results")
        
        print(f"\nCreating plots in: {args.output_dir}")
        create_plots_from_results(results, args.output_dir)
        
        print("\nAll plots created successfully!")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
