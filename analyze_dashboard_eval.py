#!/usr/bin/env python3
"""
Gabungkan dashboard_eval.jsonl + dashboard_results.jsonl, hitung ringkasan per stage,
korelasi skor manual PE2 vs G-eval, dan tulis plot + CSV untuk paper.

Contoh:
  python analyze_dashboard_eval.py \\
    --input-jsonl docs/eval_live/dashboard_eval.jsonl \\
    --results-jsonl docs/eval_live/dashboard_results.jsonl \\
    --output-dir docs/eval_live
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _stage_label(stage: str, substage: str = "") -> str:
    stage = (stage or "").strip()
    sub = (substage or "").strip()
    if sub.lower() in ("", "nan", "none"):
        return stage
    return f"{stage}_{sub}"


def merge_and_summarize(
    input_rows: List[Dict[str, Any]],
    result_rows: List[Dict[str, Any]],
    output_dir: Path,
) -> None:
    res_by_id = {r["id"]: r for r in result_rows}
    merged: List[Dict[str, Any]] = []
    for item in input_rows:
        rid = item["id"]
        r = res_by_id.get(rid, {})
        row = {**item}
        for k in ("bleu1", "rougeL", "bertscore_f1", "geval_overall", "geval_rubric_scores", "geval_rationale"):
            if k in r:
                row[k] = r[k]
        if isinstance(row.get("geval_overall"), (int, float)):
            row["geval_norm"] = float(row["geval_overall"]) / 5.0
        merged.append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    merged_path = output_dir / "dashboard_merged.csv"
    pd.DataFrame(merged).to_csv(merged_path, index=False, encoding="utf-8-sig")

    # Summary by stage
    df = pd.DataFrame(merged)
    if df.empty:
        print("No merged rows.", file=sys.stderr)
        return

    def _row_stage_key(r: pd.Series) -> str:
        sub = r.get("substage", "")
        if pd.isna(sub):
            sub = ""
        return _stage_label(str(r.get("stage", "")), str(sub))

    df["stage_key"] = df.apply(_row_stage_key, axis=1)

    def agg_col(col: str) -> Dict[str, Any]:
        if col not in df.columns:
            return {"mean": None, "std": None, "n": 0}
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        if vals.empty:
            return {"mean": None, "std": None, "n": 0}
        return {
            "mean": round(float(vals.mean()), 4),
            "std": round(float(vals.std(ddof=0)), 4) if len(vals) > 1 else 0.0,
            "n": int(len(vals)),
        }

    def _mean_col(frame: pd.DataFrame, col: str) -> Optional[float]:
        if col not in frame.columns:
            return None
        vals = pd.to_numeric(frame[col], errors="coerce").dropna()
        if vals.empty:
            return None
        return round(float(vals.mean()), 4)

    summary_rows: List[Dict[str, Any]] = []
    for stage_key, grp in df.groupby("stage_key"):
        summary_rows.append({
            "stage_key": stage_key,
            "n": len(grp),
            "bleu1_mean": _mean_col(grp, "bleu1"),
            "rougeL_mean": _mean_col(grp, "rougeL"),
            "bertscore_f1_mean": _mean_col(grp, "bertscore_f1"),
            "geval_mean": _mean_col(grp, "geval_overall"),
            "geval_norm_mean": _mean_col(grp, "geval_norm"),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / "summary_by_stage.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    # Manual vs G-eval (PE2 with manual_inquiry_score_num)
    pe2 = df[df["stage_key"] == "problem_exploring_stage2"].copy()
    pe2["manual_num"] = pd.to_numeric(pe2.get("manual_inquiry_score_num"), errors="coerce")
    pe2["geval"] = pd.to_numeric(pe2.get("geval_overall"), errors="coerce")
    valid = pe2.dropna(subset=["manual_num", "geval"])
    corr_rows: List[Dict[str, Any]] = []
    if len(valid) >= 3:
        corr = valid["manual_num"].corr(valid["geval"])
        corr_rows.append({
            "metric": "pearson_r_manual_vs_geval",
            "value": round(float(corr), 4),
            "n": len(valid),
            "note": "PE2 items with Skor inquiry X/3 in Dashboard feedback",
        })
    else:
        corr_rows.append({
            "metric": "pearson_r_manual_vs_geval",
            "value": None,
            "n": len(valid),
            "note": "Insufficient paired PE2 manual scores",
        })

    # Agreement: manual >=2 vs geval >=4
    if len(valid) > 0:
        valid = valid.copy()
        valid["manual_pass"] = valid["manual_num"] >= 2
        valid["geval_pass"] = valid["geval"] >= 4.0
        agree = (valid["manual_pass"] == valid["geval_pass"]).mean()
        corr_rows.append({
            "metric": "agreement_pass_rate_manual_geval",
            "value": round(float(agree), 4),
            "n": len(valid),
            "note": "Pass = manual>=2/3 AND geval>=4/5",
        })

    pd.DataFrame(corr_rows).to_csv(output_dir / "manual_geval_correlation.csv", index=False)

    # Plots
    try:
        import matplotlib.pyplot as plt  # type: ignore

        if not summary_df.empty and "geval_norm_mean" in summary_df.columns:
            fig, ax = plt.subplots(figsize=(9, 4.5))
            labels = summary_df["stage_key"].tolist()
            vals = summary_df["geval_norm_mean"].fillna(0).tolist()
            ax.bar(labels, vals, color="#4C78A8")
            ax.set_ylim(0, 1.05)
            ax.set_ylabel("G-eval (normalized 0-1)")
            ax.set_title("Live student experiment — mean G-eval by stage")
            plt.xticks(rotation=25, ha="right")
            plt.tight_layout()
            plt.savefig(output_dir / "geval_by_stage.png", dpi=150)
            plt.close()

        if len(valid) >= 3:
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.scatter(valid["manual_num"], valid["geval"], alpha=0.7)
            ax.set_xlabel("Manual inquiry score (numerator /3)")
            ax.set_ylabel("G-eval overall (1-5)")
            ax.set_title("PE2: manual rubric vs G-eval")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / "manual_vs_geval_pe2.png", dpi=150)
            plt.close()
    except Exception as exc:
        print(f"WARNING: plots skipped: {exc}", file=sys.stderr)

    print(f"Wrote {merged_path}", file=sys.stderr)
    print(f"Wrote {summary_path}", file=sys.stderr)
    print(f"Wrote {output_dir / 'manual_geval_correlation.csv'}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze dashboard evaluation results.")
    parser.add_argument("--input-jsonl", type=Path, default=SCRIPT_DIR / "docs/eval_live/dashboard_eval.jsonl")
    parser.add_argument("--results-jsonl", type=Path, default=SCRIPT_DIR / "docs/eval_live/dashboard_results.jsonl")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "docs/eval_live")
    args = parser.parse_args()

    if not args.input_jsonl.exists() or not args.results_jsonl.exists():
        print("ERROR: input or results JSONL missing. Run export + eval first.", file=sys.stderr)
        return 1

    merge_and_summarize(_load_jsonl(args.input_jsonl), _load_jsonl(args.results_jsonl), args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
