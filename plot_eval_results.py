#!/usr/bin/env python3
"""
Script untuk membuat plot evaluasi dari Excel file.
Menampilkan perbandingan metrics berdasarkan prompt_type dan stage dalam 1 plot.
"""

import argparse
import os
import sys
from typing import Dict, List, Any

try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError as e:
    print(f"Error: Missing required library. Install with: pip install pandas matplotlib numpy openpyxl")
    sys.exit(1)


def load_excel_data(file_path: str) -> pd.DataFrame:
    """Load data dari Excel file."""
    try:
        df = pd.read_excel(file_path)
        print(f"Loaded {len(df)} rows from {file_path}")
        return df
    except Exception as e:
        print(f"Error loading Excel file: {e}")
        sys.exit(1)


def create_comparison_plot(df: pd.DataFrame, output_path: str, metrics: List[str] = None) -> None:
    """
    Membuat plot perbandingan metrics berdasarkan prompt_type dan stage.
    
    Args:
        df: DataFrame dengan kolom stage, prompt_type, dan metrics
        output_path: Path untuk menyimpan plot
        metrics: List metrics yang akan diplot (jika None, akan auto-detect)
    """
    # Auto-detect metrics (exclude stage dan prompt_type)
    if metrics is None:
        exclude_cols = ['stage', 'prompt_type', 'id', 'lang', 'context']
        metrics = [col for col in df.columns if col not in exclude_cols]
    
    # Filter hanya metrics yang ada di dataframe
    available_metrics = [m for m in metrics if m in df.columns]
    
    if not available_metrics:
        print("Error: No metrics found in data")
        return
    
    print(f"Plotting metrics: {available_metrics}")
    
    # Group by stage dan prompt_type, calculate mean
    grouped = df.groupby(['stage', 'prompt_type'])[available_metrics].mean().reset_index()
    
    # Prepare data untuk plotting
    stages = grouped['stage'].unique()
    prompt_types = grouped['prompt_type'].unique()
    
    # Mapping untuk label yang lebih readable
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
    
    # Create figure dengan subplots
    n_metrics = len(available_metrics)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
    if n_metrics == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    # Color palette
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # Plot setiap metric
    for idx, metric in enumerate(available_metrics):
        ax = axes[idx]
        
        # Prepare data untuk grouped bar chart
        x = np.arange(len(stages))
        width = 0.2  # Width of bars
        multiplier = 0
        
        for prompt_type in prompt_types:
            values = []
            for stage in stages:
                row = grouped[(grouped['stage'] == stage) & (grouped['prompt_type'] == prompt_type)]
                if not row.empty:
                    values.append(row[metric].values[0])
                else:
                    values.append(0)
            
            offset = width * multiplier
            bars = ax.bar(x + offset, values, width, label=prompt_labels.get(prompt_type, prompt_type), 
                         color=colors[multiplier % len(colors)], alpha=0.8)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=8)
            
            multiplier += 1
        
        ax.set_xlabel('Stage', fontsize=12, fontweight='bold')
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=11)
        ax.set_title(f'{metric.replace("_", " ").title()}', fontsize=13, fontweight='bold')
        ax.set_xticks(x + width * (len(prompt_types) - 1) / 2)
        ax.set_xticklabels([stage_labels.get(s, s) for s in stages], fontsize=10)
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1)
    
    # Hide unused subplots
    for idx in range(len(available_metrics), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    plt.close()


def create_combined_plot(df: pd.DataFrame, output_path: str, metrics: List[str] = None) -> None:
    """
    Membuat 1 plot besar dengan semua metrics dalam subplot.
    """
    # Auto-detect metrics
    if metrics is None:
        exclude_cols = ['stage', 'prompt_type', 'id', 'lang', 'context']
        metrics = [col for col in df.columns if col not in exclude_cols]
    
    available_metrics = [m for m in metrics if m in df.columns]
    
    if not available_metrics:
        print("Error: No metrics found in data")
        return
    
    # Group by stage dan prompt_type
    grouped = df.groupby(['stage', 'prompt_type'])[available_metrics].mean().reset_index()
    
    stages = grouped['stage'].unique()
    prompt_types = grouped['prompt_type'].unique()
    
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
                    values.append(row[metric].values[0])
                else:
                    values.append(0)
            
            offset = width * multiplier
            bars = ax.bar(x + offset, values, width, label=prompt_labels.get(prompt_type, prompt_type),
                         color=colors[multiplier % len(colors)], alpha=0.8, edgecolor='black', linewidth=0.5)
            
            # Value labels
            for bar in bars:
                height = bar.get_height()
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
    print(f"Combined plot saved to {output_path}")
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create comparison plots from Excel evaluation results"
    )
    parser.add_argument("--input", required=True, help="Input Excel file path")
    parser.add_argument("--output", default="eval_plots/comparison_plot.png", help="Output plot path")
    parser.add_argument(
        "--metrics",
        nargs="+",
        help="Specific metrics to plot (default: all available metrics)"
    )
    parser.add_argument(
        "--combined",
        action="store_true",
        help="Create one combined plot with all metrics"
    )
    
    args = parser.parse_args()
    
    # Load data
    df = load_excel_data(args.input)
    
    # Validate required columns
    if 'stage' not in df.columns or 'prompt_type' not in df.columns:
        print("Error: Excel file must contain 'stage' and 'prompt_type' columns")
        return 1
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Create plot
    if args.combined:
        create_combined_plot(df, args.output, args.metrics)
    else:
        create_comparison_plot(df, args.output, args.metrics)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
