#!/usr/bin/env python3
"""Plot live multimodal ablation (4 conditions, BLEU/ROUGE/BERTScore only)."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional

CONDITION_ORDER = [
    "zero_shot_no_image",
    "zero_shot_image_no_recognition",
    "zero_shot_image_recognition",
    "few_shot_cot_recognition",
]
CONDITION_LABELS = {
    "zero_shot_no_image": "ZS\n(no img/rec)",
    "zero_shot_image_no_recognition": "ZS+img\n(no rec)",
    "zero_shot_image_recognition": "ZS+img+rec",
    "few_shot_cot_recognition": "FS+CoT\n+img+rec",
}
CONDITION_COLORS = {
    "zero_shot_no_image": "#6B7280",
    "zero_shot_image_no_recognition": "#9CA3AF",
    "zero_shot_image_recognition": "#4C78A8",
    "few_shot_cot_recognition": "#54A24B",
}
METRICS = ["bleu1", "rougeL", "bertscore_f1"]
METRIC_LABELS = ["BLEU-1", "ROUGE-L", "BERTScore-F1"]
BASELINE = "zero_shot_no_image"


def _to_float(x: Any) -> Optional[float]:
    if x is None or x == "":
        return None
    try:
        f = float(x)
        return None if math.isnan(f) or math.isinf(f) else f
    except (TypeError, ValueError):
        return None


def _normalize_row(row: Dict[str, Any], bleu_scale: str) -> Dict[str, Any]:
    out = dict(row)
    bleu1 = _to_float(row.get("bleu1"))
    if bleu1 is not None and bleu_scale == "0-100":
        bleu1 = bleu1 / 100.0
    out["bleu1"] = bleu1
    out["rougeL"] = _to_float(row.get("rougeL"))
    out["bertscore_f1"] = _to_float(row.get("bertscore_f1"))
    return out


def _summarize(by_cond: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for cond in CONDITION_ORDER:
        grp = by_cond.get(cond, [])

        def m(key: str) -> float:
            vals = [r[key] for r in grp if r.get(key) is not None]
            return mean(vals) if vals else 0.0

        def s(key: str) -> float:
            vals = [r[key] for r in grp if r.get(key) is not None]
            return pstdev(vals) if len(vals) > 1 else 0.0

        bleu_m, rouge_m, bert_m = m("bleu1"), m("rougeL"), m("bertscore_f1")
        has_bert = bert_m > 0
        composite = (bleu_m + rouge_m + bert_m) / 3.0 if has_bert else (bleu_m + rouge_m) / 2.0
        rows.append({
            "prompt_type": cond,
            "n": len(grp),
            "bleu1_mean": bleu_m,
            "bleu1_std": s("bleu1"),
            "rougeL_mean": rouge_m,
            "rougeL_std": s("rougeL"),
            "bertscore_f1_mean": bert_m,
            "bertscore_f1_std": s("bertscore_f1"),
            "composite_mean": composite,
            "has_bertscore": has_bert,
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--input-jsonl", default="")
    parser.add_argument("--output-dir", default="eval_plots_live")
    parser.add_argument("--bleu-scale", default="0-100", choices=["0-1", "0-100"])
    args = parser.parse_args()

    rows: List[Dict[str, Any]] = []
    with open(args.input, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(_normalize_row(r, args.bleu_scale))

    by_cond: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if r.get("prompt_type"):
            by_cond[str(r["prompt_type"])].append(r)

    summary_rows = _summarize(by_cond)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    try:
        import matplotlib.pyplot as plt  # type: ignore
        import numpy as np  # type: ignore

        by = {r["prompt_type"]: r for r in summary_rows}

        # Figure 1: grouped bar — x = conditions, grouped by metric
        fig, ax = plt.subplots(figsize=(12, 5.5))
        x = np.arange(len(CONDITION_ORDER))
        has_bert = any(by[c].get("bertscore_f1_mean", 0) > 0 for c in CONDITION_ORDER)
        metrics = list(METRICS) if has_bert else ["bleu1", "rougeL"]
        metric_labels = list(METRIC_LABELS) if has_bert else ["BLEU-1", "ROUGE-L"]
        n_met = len(metrics)
        width = 0.22 if has_bert else 0.35
        for i, (mk, ml) in enumerate(zip(metrics, metric_labels)):
            means = [by[c][f"{mk}_mean"] for c in CONDITION_ORDER]
            stds = [by[c][f"{mk}_std"] for c in CONDITION_ORDER]
            offset = (i - (n_met - 1) / 2) * width
            ax.bar(x + offset, means, width, yerr=stds, label=ml, capsize=2)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [CONDITION_LABELS[c] for c in CONDITION_ORDER],
            fontsize=9,
            rotation=0,
            ha="center",
        )
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score (0–1)")
        ax.set_title(
            "Live multimodal ablation — "
            + ("BLEU-1, ROUGE-L, BERTScore-F1" if has_bert else "BLEU-1, ROUGE-L")
        )
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "grouped_bar.png", dpi=150)
        plt.close()

        # Figure 2: delta vs baseline — x = metrics (no overlap)
        base = by.get(BASELINE)
        if base:
            fig, ax = plt.subplots(figsize=(9, 4.5))
            compare_conds = [c for c in CONDITION_ORDER if c != BASELINE]
            width = 0.24
            delta_metrics = (
                ["bleu1_mean", "rougeL_mean", "bertscore_f1_mean"]
                if any(by[c].get("bertscore_f1_mean", 0) > 0 for c in CONDITION_ORDER)
                else ["bleu1_mean", "rougeL_mean"]
            )
            delta_labels = (
                METRIC_LABELS
                if len(delta_metrics) == 3
                else ["BLEU-1", "ROUGE-L"]
            )
            x_m = np.arange(len(delta_labels))
            for i, cond in enumerate(compare_conds):
                sr = by[cond]
                deltas = [sr[k] - base[k] for k in delta_metrics]
                offset = (i - (len(compare_conds) - 1) / 2) * width
                ax.bar(
                    x_m + offset,
                    deltas,
                    width,
                    label=CONDITION_LABELS[cond].replace("\n", " "),
                    color=CONDITION_COLORS.get(cond, "#333"),
                )
            ax.axhline(0, color="black", lw=0.8)
            ax.set_xticks(x_m)
            ax.set_xticklabels(delta_labels, fontsize=10)
            ax.set_ylabel("Δ vs baseline")
            ax.set_title("Improvement over ZS (no image, no recognition)")
            ax.legend(loc="best", fontsize=8)
            ax.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            plt.savefig(out_dir / "delta_vs_baseline.png", dpi=150)
            plt.close()

        # Figure 3: line plot per condition (all metrics)
        fig, ax = plt.subplots(figsize=(10, 4.5))
        profile_labels = METRIC_LABELS if has_bert else ["BLEU-1", "ROUGE-L"]
        for cond in CONDITION_ORDER:
            sr = by[cond]
            vals = (
                [sr["bleu1_mean"], sr["rougeL_mean"], sr["bertscore_f1_mean"]]
                if has_bert
                else [sr["bleu1_mean"], sr["rougeL_mean"]]
            )
            ax.plot(profile_labels, vals, marker="o", label=CONDITION_LABELS[cond].replace("\n", " "))
        ax.set_ylim(0, 1.05)
        ax.set_title("Metric profile by condition")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "metric_profile.png", dpi=150)
        plt.close()

        # Figure 4: per-case BERTScore C vs B (optional)
        if args.input_jsonl:
            by_group: Dict[str, Dict[str, float]] = defaultdict(dict)
            with open(args.input_jsonl, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    gid = str(r.get("item_group_id", ""))
                    pt = str(r.get("prompt_type", ""))
                    b = _to_float(r.get("bertscore_f1"))
                    if gid and b is not None:
                        by_group[gid][pt] = b
            diffs = []
            for gid in sorted(by_group.keys()):
                d = by_group[gid]
                if "few_shot_cot_recognition" in d and "zero_shot_image_recognition" in d:
                    diffs.append(d["few_shot_cot_recognition"] - d["zero_shot_image_recognition"])
            if diffs:
                fig, ax = plt.subplots(figsize=(12, 3.5))
                ax.bar(range(len(diffs)), diffs, color=["#54A24B" if x >= 0 else "#9CA3AF" for x in diffs])
                ax.axhline(0, color="black", lw=0.8)
                ax.set_title("Per-photo Δ BERTScore-F1: Few-shot+CoT − ZS+img+rec")
                ax.set_ylabel("Δ BERTScore-F1")
                plt.tight_layout()
                plt.savefig(out_dir / "pair_diff_bertscore.png", dpi=150)
                plt.close()

    except Exception as exc:
        print(f"Plot warning: {exc}", file=sys.stderr)

    print(f"Wrote {out_dir / 'summary.csv'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
