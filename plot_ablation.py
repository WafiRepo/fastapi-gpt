#!/usr/bin/env python3
"""
Plot perbandingan hasil ablasi prompt problem-finder.

Empat kondisi:
  - zero_shot_no_image (baseline, tanpa gambar/recognition)
  - zero_shot_image    (zero-shot + gambar + recognition)
  - few_shot           (few-shot + gambar + recognition)
  - cot                (CoT + gambar + recognition)

Output:
  - summary.csv         : mean & std per kondisi (BLEU-1, ROUGE-L, BERTScore-F1, G-eval(norm))
  - grouped_bar.png     : grouped bar metrik per kondisi
  - geval_rubric.png    : rubric G-eval per dimensi (butuh --input-jsonl)
  - pair_diff.png       : selisih G-eval per test case vs baseline
  - delta_vs_baseline.png : delta mean per metrik (B/C/D - A)

Contoh:
  python plot_ablation.py \
    --input ablation/ablation_results.csv \
    --input-jsonl ablation/ablation_results.jsonl \
    --output-dir ablation/eval_plots --bleu-scale 0-100
"""

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


CONDITION_ORDER = ["zero_shot_no_image", "zero_shot_image", "few_shot", "cot"]
CONDITION_LABELS = {
    "zero_shot_no_image": "ZS (no img/rec)\nbaseline",
    "zero_shot_image": "ZS\n+ img + rec",
    "few_shot": "Few-shot\n+ img + rec",
    "cot": "CoT\n+ img + rec",
}
CONDITION_COLORS = {
    "zero_shot_no_image": "#9CA3AF",
    "zero_shot_image": "#4C78A8",
    "few_shot": "#54A24B",
    "cot": "#B279A2",
}
METRIC_ORDER = ["bleu1", "rougeL", "bertscore_f1", "geval_norm"]
METRIC_LABELS = {
    "bleu1": "BLEU-1",
    "rougeL": "ROUGE-L",
    "bertscore_f1": "BERTScore-F1",
    "geval_norm": "G-eval (0-1)",
}
METRIC_BAR_COLORS = ["#4C78A8", "#F58518", "#54A24B", "#B279A2"]


def _to_float(x: Any) -> Optional[float]:
    if x is None or x == "":
        return None
    try:
        f = float(x)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


def _normalize_row(row: Dict[str, Any], bleu_scale: str) -> Dict[str, Any]:
    out: Dict[str, Any] = dict(row)
    bleu1 = _to_float(row.get("bleu1"))
    if bleu1 is not None and bleu_scale == "0-100":
        bleu1 = bleu1 / 100.0
    out["bleu1"] = bleu1
    out["bleu4"] = _to_float(row.get("bleu4"))
    out["rouge1"] = _to_float(row.get("rouge1"))
    out["rouge2"] = _to_float(row.get("rouge2"))
    out["rougeL"] = _to_float(row.get("rougeL"))
    out["bertscore_precision"] = _to_float(row.get("bertscore_precision"))
    out["bertscore_recall"] = _to_float(row.get("bertscore_recall"))
    out["bertscore_f1"] = _to_float(row.get("bertscore_f1"))
    g = _to_float(row.get("geval_overall"))
    out["geval_overall"] = g
    out["geval_norm"] = g / 5.0 if g is not None else None
    rubric_raw = row.get("geval_rubric_scores")
    rubric: Dict[str, float] = {}
    if isinstance(rubric_raw, dict):
        for k, v in rubric_raw.items():
            try:
                rubric[str(k)] = float(v)
            except (TypeError, ValueError):
                continue
    elif isinstance(rubric_raw, str) and rubric_raw.strip():
        try:
            parsed = json.loads(rubric_raw)
        except json.JSONDecodeError:
            try:
                fixed = (
                    rubric_raw.replace("'", '"')
                    .replace("None", "null")
                    .replace("True", "true")
                    .replace("False", "false")
                )
                parsed = json.loads(fixed)
            except Exception:
                parsed = {}
        if isinstance(parsed, dict):
            for k, v in parsed.items():
                try:
                    rubric[str(k)] = float(v)
                except (TypeError, ValueError):
                    continue
    out["geval_rubric_scores"] = rubric
    return out


def _read_csv(path: Path, bleu_scale: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(_normalize_row(dict(r), bleu_scale))
    return rows


def _read_jsonl(path: Path, bleu_scale: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(_normalize_row(json.loads(line), bleu_scale))
            except json.JSONDecodeError:
                continue
    return rows


def _merge_rubric_from_jsonl(rows: List[Dict[str, Any]], jsonl_rows: List[Dict[str, Any]]) -> None:
    if not jsonl_rows:
        return
    idx: Dict[str, Dict[str, float]] = {}
    for r in jsonl_rows:
        rid = str(r.get("id", ""))
        if rid:
            rubric = r.get("geval_rubric_scores") or {}
            if isinstance(rubric, dict) and rubric:
                idx[rid] = rubric
    for r in rows:
        rid = str(r.get("id", ""))
        if rid in idx and not r.get("geval_rubric_scores"):
            r["geval_rubric_scores"] = idx[rid]


def _group_by_condition(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    g: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        cond = str(r.get("prompt_type", "")).strip()
        if cond in CONDITION_ORDER:
            g[cond].append(r)
    return g


def _mean_metric(items: List[Dict[str, Any]], key: str) -> float:
    vals = [it[key] for it in items if isinstance(it.get(key), (int, float))]
    return mean(vals) if vals else 0.0


def _std_metric(items: List[Dict[str, Any]], key: str) -> float:
    vals = [it[key] for it in items if isinstance(it.get(key), (int, float))]
    return pstdev(vals) if len(vals) > 1 else 0.0


def _mean_rubric(items: List[Dict[str, Any]], dim: str) -> float:
    vals: List[float] = []
    for it in items:
        rub = it.get("geval_rubric_scores") or {}
        v = rub.get(dim)
        if isinstance(v, (int, float)):
            vals.append(float(v))
    return mean(vals) if vals else 0.0


def _all_rubric_dims(rows: List[Dict[str, Any]]) -> List[str]:
    seen: List[str] = []
    for r in rows:
        rub = r.get("geval_rubric_scores") or {}
        for k in rub.keys():
            if k not in seen:
                seen.append(str(k))
    return seen


def _write_summary_csv(out_path: Path, grouped: Dict[str, List[Dict[str, Any]]]) -> None:
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        header = ["prompt_type", "n"]
        for m in METRIC_ORDER:
            header += [f"{m}_mean", f"{m}_std"]
        w.writerow(header)
        for cond in CONDITION_ORDER:
            items = grouped.get(cond, [])
            row = [cond, len(items)]
            for m in METRIC_ORDER:
                row += [f"{_mean_metric(items, m):.4f}", f"{_std_metric(items, m):.4f}"]
            w.writerow(row)


def _plot_grouped_bar(grouped: Dict[str, List[Dict[str, Any]]], output_dir: Path) -> Path:
    import matplotlib.pyplot as plt  # type: ignore

    conds = [c for c in CONDITION_ORDER if c in grouped]
    labels = [CONDITION_LABELS[c] for c in conds]
    x = list(range(len(conds)))
    width = 0.18

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for i, m in enumerate(METRIC_ORDER):
        vals = [_mean_metric(grouped[c], m) for c in conds]
        stds = [_std_metric(grouped[c], m) for c in conds]
        offset = (i - 1.5) * width
        bars = ax.bar(
            [xi + offset for xi in x],
            vals,
            width=width,
            yerr=stds,
            capsize=3,
            label=METRIC_LABELS[m],
            color=METRIC_BAR_COLORS[i % len(METRIC_BAR_COLORS)],
        )
        for b, v in zip(bars, vals):
            ax.text(
                b.get_x() + b.get_width() / 2.0,
                v + 0.02,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score (0-1)")
    ax.set_title(
        "Pengaruh input gambar + recognition pada kualitas inquiry\n(mean \u00b1 std; 10 test case per kondisi)"
    )
    ax.axvline(0.5, color="black", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    plt.tight_layout()
    out = output_dir / "grouped_bar.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def _plot_rubric(
    grouped: Dict[str, List[Dict[str, Any]]], rows: List[Dict[str, Any]], output_dir: Path
) -> Optional[Path]:
    import matplotlib.pyplot as plt  # type: ignore

    dims = _all_rubric_dims(rows)
    if not dims:
        return None
    conds = [c for c in CONDITION_ORDER if c in grouped]
    if not conds:
        return None

    x = list(range(len(dims)))
    width = 0.8 / max(1, len(conds))
    fig, ax = plt.subplots(figsize=(max(8.5, 1.4 * len(dims)), 5.2))
    for i, cond in enumerate(conds):
        vals = [_mean_rubric(grouped[cond], d) for d in dims]
        offset = (i - (len(conds) - 1) / 2.0) * width
        bars = ax.bar(
            [xi + offset for xi in x],
            vals,
            width=width,
            label=CONDITION_LABELS[cond].replace("\n", " "),
            color=CONDITION_COLORS[cond],
        )
        for b, v in zip(bars, vals):
            ax.text(
                b.get_x() + b.get_width() / 2.0,
                v + 0.05,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
    ax.set_xticks(x)
    ax.set_xticklabels([d.replace(" ", "\n") for d in dims], fontsize=9)
    ax.set_ylim(0, 5.4)
    ax.set_ylabel("G-eval rubric score (1-5)")
    ax.set_title("Rubric G-eval per dimensi per kondisi prompt")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(ncol=len(conds), loc="upper center", bbox_to_anchor=(0.5, -0.2))
    plt.tight_layout()
    out = output_dir / "geval_rubric.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def _plot_pair_diff(grouped: Dict[str, List[Dict[str, Any]]], output_dir: Path) -> Optional[Path]:
    import matplotlib.pyplot as plt  # type: ignore

    base = grouped.get("zero_shot_no_image")
    if not base:
        return None
    base_map: Dict[str, float] = {}
    for it in base:
        gid = str(it.get("item_group_id") or it.get("id"))
        g = it.get("geval_norm")
        if isinstance(g, (int, float)):
            base_map[gid] = float(g)
    if not base_map:
        return None

    others = [c for c in ["zero_shot_image", "few_shot", "cot"] if c in grouped]
    case_ids = sorted(base_map.keys())
    if not others or not case_ids:
        return None

    x = list(range(len(case_ids)))
    width = 0.8 / max(1, len(others))
    fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(case_ids) + 4), 5.0))
    for i, cond in enumerate(others):
        cond_map: Dict[str, float] = {}
        for it in grouped[cond]:
            gid = str(it.get("item_group_id") or it.get("id"))
            g = it.get("geval_norm")
            if isinstance(g, (int, float)):
                cond_map[gid] = float(g)
        diffs = [(cond_map.get(cid, 0.0) - base_map.get(cid, 0.0)) for cid in case_ids]
        offset = (i - (len(others) - 1) / 2.0) * width
        ax.bar(
            [xi + offset for xi in x],
            diffs,
            width=width,
            label=CONDITION_LABELS[cond].replace("\n", " "),
            color=CONDITION_COLORS[cond],
        )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(case_ids, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("\u0394 G-eval (cond - ZS no img/rec)")
    ax.set_title("Selisih G-eval per test case vs baseline (lebih tinggi = lebih baik)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(ncol=len(others), loc="upper center", bbox_to_anchor=(0.5, -0.22))
    plt.tight_layout()
    out = output_dir / "pair_diff.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def _plot_delta_vs_baseline(grouped: Dict[str, List[Dict[str, Any]]], output_dir: Path) -> Optional[Path]:
    import matplotlib.pyplot as plt  # type: ignore

    if "zero_shot_no_image" not in grouped:
        return None
    others = [c for c in ["zero_shot_image", "few_shot", "cot"] if c in grouped]
    if not others:
        return None

    x = list(range(len(METRIC_ORDER)))
    width = 0.8 / max(1, len(others))
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    for i, cond in enumerate(others):
        deltas: List[float] = []
        for m in METRIC_ORDER:
            base_mean = _mean_metric(grouped["zero_shot_no_image"], m)
            deltas.append(_mean_metric(grouped[cond], m) - base_mean)
        offset = (i - (len(others) - 1) / 2.0) * width
        bars = ax.bar(
            [xi + offset for xi in x],
            deltas,
            width=width,
            label=CONDITION_LABELS[cond].replace("\n", " "),
            color=CONDITION_COLORS[cond],
        )
        for b, v in zip(bars, deltas):
            ax.text(
                b.get_x() + b.get_width() / 2.0,
                v + (0.005 if v >= 0 else -0.015),
                f"{v:+.2f}",
                ha="center",
                va="bottom" if v >= 0 else "top",
                fontsize=8,
            )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS[m] for m in METRIC_ORDER])
    ax.set_ylabel("\u0394 vs baseline (ZS no img/rec)")
    ax.set_title("Peningkatan rata-rata terhadap baseline tanpa gambar + recognition")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(ncol=len(others), loc="upper center", bbox_to_anchor=(0.5, -0.12))
    plt.tight_layout()
    out = output_dir / "delta_vs_baseline.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot perbandingan ablasi prompt problem-finder.")
    parser.add_argument("--input", required=True, help="CSV hasil eval_inquiry_metrics.py")
    parser.add_argument(
        "--input-jsonl",
        default=None,
        help="(Opsional) JSONL hasil evaluator untuk membaca geval_rubric_scores.",
    )
    parser.add_argument("--output-dir", default="ablation/eval_plots", help="Folder output plot.")
    parser.add_argument(
        "--bleu-scale",
        choices=["0-1", "0-100"],
        default="0-1",
        help="Skala BLEU pada CSV input.",
    )
    args = parser.parse_args()

    in_path = Path(args.input).expanduser().resolve()
    if not in_path.exists():
        print(f"ERROR: input tidak ditemukan: {in_path}", file=sys.stderr)
        return 1

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_csv(in_path, args.bleu_scale)
    if args.input_jsonl:
        jsonl_path = Path(args.input_jsonl).expanduser().resolve()
        if jsonl_path.exists():
            jsonl_rows = _read_jsonl(jsonl_path, args.bleu_scale)
            _merge_rubric_from_jsonl(rows, jsonl_rows)

    if not rows:
        print("ERROR: CSV kosong.", file=sys.stderr)
        return 1

    grouped = _group_by_condition(rows)
    if not grouped:
        print("ERROR: tidak ada kolom prompt_type yang cocok dengan kondisi ablasi.", file=sys.stderr)
        return 1

    summary_csv = out_dir / "summary.csv"
    _write_summary_csv(summary_csv, grouped)

    bar_path = _plot_grouped_bar(grouped, out_dir)
    rubric_path = _plot_rubric(grouped, rows, out_dir)
    pair_path = _plot_pair_diff(grouped, out_dir)
    delta_path = _plot_delta_vs_baseline(grouped, out_dir)

    print(f"Wrote: {summary_csv}")
    print(f"Wrote: {bar_path}")
    if rubric_path:
        print(f"Wrote: {rubric_path}")
    if pair_path:
        print(f"Wrote: {pair_path}")
    if delta_path:
        print(f"Wrote: {delta_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
