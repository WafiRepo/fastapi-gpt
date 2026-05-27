#!/usr/bin/env python3
"""
Generate English PDF: Live student experiment evaluation (Dashboard CSV).

Reads:
  docs/eval_live/summary_by_stage.csv
  docs/eval_live/manual_geval_correlation.csv
  docs/eval_live/geval_by_stage.png
  docs/eval_live/manual_vs_geval_pe2.png (optional)

Writes:
  docs/eval_live/Dashboard_Live_Eval_Report.pdf
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

SCRIPT_DIR = Path(__file__).resolve().parent
EVAL_DIR = SCRIPT_DIR
OUT_PDF = EVAL_DIR / "Dashboard_Live_Eval_Report.pdf"


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _table_from_rows(rows: List[Dict[str, Any]], cols: List[str]) -> Table:
    data = [cols]
    for r in rows:
        data.append([str(r.get(c, "")) for c in cols])
    t = Table(data, repeatRows=1)
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4C78A8")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    return t


def build_pdf() -> None:
    styles = getSampleStyleSheet()
    body = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        alignment=TA_JUSTIFY,
    )
    h1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=16, spaceAfter=12)
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=12, spaceAfter=8)

    summary = _read_csv(EVAL_DIR / "summary_by_stage.csv")
    corr = _read_csv(EVAL_DIR / "manual_geval_correlation.csv")

    doc = SimpleDocTemplate(
        str(OUT_PDF),
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )
    story: List[Any] = []

    story.append(Paragraph("Live Student Experiment Evaluation", h1))
    story.append(
        Paragraph(
            "This report documents automatic evaluation of inquiry outputs collected "
            "from the P-Magiv Dashboard (centripetal acceleration module). Student "
            "responses were converted to JSONL, scored with BLEU, ROUGE-L, BERTScore "
            "(where reference text exists), and G-eval (LLM-as-judge with a five-dimension "
            "rubric). Problem Exploring stage 2 items include manual inquiry scores "
            "extracted from teacher feedback in the Dashboard CSV.",
            body,
        )
    )
    story.append(Spacer(1, 0.4 * cm))

    story.append(Paragraph("1. Data source and pipeline", h2))
    story.append(
        Paragraph(
            "<b>Source:</b> <i>Dashboard P-Magiv V2.csv</i> — student emails, object "
            "recognition, Problem Finding (PF), Problem Exploring stages 1–2 (PE1/PE2), "
            "and Problem Generating (PG) at easy/intermediate/advanced difficulty.<br/>"
            "<b>Export:</b> <code>export_dashboard_csv_for_eval.py</code> → "
            "<code>dashboard_eval.jsonl</code>.<br/>"
            "<b>Metrics:</b> <code>eval_inquiry_metrics.py</code> → "
            "<code>dashboard_results.jsonl</code>.<br/>"
            "<b>Analysis:</b> <code>analyze_dashboard_eval.py</code> merges inputs with "
            "scores and compares PE2 manual rubric (X/3) to G-eval.",
            body,
        )
    )
    story.append(Spacer(1, 0.3 * cm))

    story.append(Paragraph("2. Aggregate scores by stage", h2))
    if summary:
        cols = [
            "stage_key",
            "n",
            "bleu1_mean",
            "rougeL_mean",
            "bertscore_f1_mean",
            "geval_mean",
            "geval_norm_mean",
        ]
        story.append(_table_from_rows(summary, cols))
    else:
        story.append(Paragraph("(Run analyze_dashboard_eval.py after evaluation.)", body))
    story.append(Spacer(1, 0.4 * cm))

    plot_geval = EVAL_DIR / "geval_by_stage.png"
    if plot_geval.exists():
        story.append(Paragraph("Figure 1. Mean G-eval (normalized) by stage", h2))
        story.append(Image(str(plot_geval), width=14 * cm, height=7 * cm))
        story.append(Spacer(1, 0.3 * cm))

    story.append(Paragraph("3. Manual vs automatic rubric (PE2)", h2))
    if corr:
        story.append(_table_from_rows(corr, ["metric", "value", "n", "note"]))
    plot_corr = EVAL_DIR / "manual_vs_geval_pe2.png"
    if plot_corr.exists():
        story.append(Spacer(1, 0.2 * cm))
        story.append(Paragraph("Figure 2. PE2 manual inquiry score vs G-eval", h2))
        story.append(Image(str(plot_corr), width=10 * cm, height=8 * cm))

    story.append(PageBreak())
    story.append(Paragraph("4. Comparison with ablation study", h2))
    story.append(
        Paragraph(
            "The controlled ablation (n=10 per condition, synthetic test cases with "
            "object-specific references) showed that multimodal grounded prompts (B/C/D) "
            "substantially outperform a text-only baseline (A) on overlap metrics and G-eval. "
            "Live student data spans all inquiry stages and real classroom variance; lower "
            "BLEU/ROUGE on PF/PE is expected when references are generic templates rather "
            "than per-student gold answers. G-eval is the primary comparable signal because "
            "it judges inquiry quality from context without requiring exact wording match.",
            body,
        )
    )
    story.append(Spacer(1, 0.3 * cm))
    story.append(
        Paragraph(
            "<b>Interpretation guidelines:</b> (1) Use G-eval and rubric dimensions for "
            "cross-stage quality trends. (2) Treat BLEU/ROUGE/BERTScore on PF/PE as "
            "reference-alignment diagnostics only when references are specific. "
            "(3) For PE2, report Pearson correlation and pass-rate agreement between "
            "manual X/3 scores and G-eval ≥4/5. (4) PG items without references rely "
            "solely on G-eval.",
            body,
        )
    )

    doc.build(story)
    print(f"Wrote {OUT_PDF}", file=sys.stderr)


if __name__ == "__main__":
    build_pdf()
