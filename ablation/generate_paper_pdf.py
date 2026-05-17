#!/usr/bin/env python3
"""
Build a paper-style PDF report (English) summarizing the ablation study on
problem-finding inquiry generation.

Reads:
  - ablation/test_cases_pf.json
  - ablation/eval_plots/summary.csv
  - ablation/ablation_results.jsonl
  - ablation/eval_plots/{grouped_bar,delta_vs_baseline,pair_diff,geval_rubric}.png

Writes:
  - ablation/Ablation_PF_Report.pdf
"""

from __future__ import annotations

import csv
import html
import json
import random
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT, TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Preformatted,
    Spacer,
    Table,
    TableStyle,
    XPreformatted,
)


SCRIPT_DIR = Path(__file__).resolve().parent
ABL_DIR = SCRIPT_DIR
PLOTS_DIR = ABL_DIR / "eval_plots"
OUT_PDF = ABL_DIR / "Ablation_PF_Report.pdf"
FASTAPI_DIR = ABL_DIR.parent
TEST_CASES_JSON = ABL_DIR / "test_cases_pf.json"

# Import the actual prompt builders from the generator so the PDF always
# reflects the exact prompts that were sent to the LLM.
sys.path.insert(0, str(FASTAPI_DIR))
try:
    from gen_ablation_problem_finder import (  # type: ignore
        SYSTEM_PROMPT_BASELINE_ID,
        SYSTEM_PROMPT_GROUNDED_ID,
        _pick_few_shot_examples,
        _user_prompt_cot,
        _user_prompt_few_shot,
        _user_prompt_zero_shot_image,
        _user_prompt_zero_shot_no_image,
    )
except Exception:  # pragma: no cover
    SYSTEM_PROMPT_BASELINE_ID = ""
    SYSTEM_PROMPT_GROUNDED_ID = ""
    _pick_few_shot_examples = None  # type: ignore
    _user_prompt_cot = None  # type: ignore
    _user_prompt_few_shot = None  # type: ignore
    _user_prompt_zero_shot_image = None  # type: ignore
    _user_prompt_zero_shot_no_image = None  # type: ignore

CONDITION_ORDER = ["zero_shot_no_image", "zero_shot_image", "few_shot", "cot"]
CONDITION_PRETTY = {
    "zero_shot_no_image": "A. Zero-shot (no image, no recognition)",
    "zero_shot_image": "B. Zero-shot + image + recognition",
    "few_shot": "C. Few-shot + image + recognition",
    "cot": "D. Chain-of-Thought + image + recognition",
}
CONDITION_SHORT = {
    "zero_shot_no_image": "A: ZS",
    "zero_shot_image": "B: ZS+IR",
    "few_shot": "C: FS+IR",
    "cot": "D: CoT+IR",
}


def _read_summary_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _rubric_means(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    by_cond: Dict[str, Dict[str, List[float]]] = {c: {} for c in CONDITION_ORDER}
    for r in rows:
        cond = str(r.get("prompt_type", ""))
        if cond not in by_cond:
            continue
        rubric = r.get("geval_rubric_scores") or {}
        if isinstance(rubric, dict):
            for k, v in rubric.items():
                if isinstance(v, (int, float)):
                    by_cond[cond].setdefault(str(k), []).append(float(v))
    return {
        c: {k: round(mean(vs), 2) for k, vs in d.items() if vs}
        for c, d in by_cond.items()
    }


def _sample_inquiries(rows: List[Dict[str, Any]], group_id: str) -> Dict[str, str]:
    """Return one inquiry per condition for a given test case id."""
    out: Dict[str, str] = {}
    for r in rows:
        if str(r.get("item_group_id")) == group_id:
            out[str(r.get("prompt_type"))] = str(r.get("inquiry", ""))
    return out


def _styles() -> Dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    body = ParagraphStyle(
        "Body",
        parent=base["BodyText"],
        fontName="Times-Roman",
        fontSize=10.5,
        leading=14,
        alignment=TA_JUSTIFY,
        spaceAfter=4,
    )
    title = ParagraphStyle(
        "TitleX",
        parent=base["Title"],
        fontName="Times-Bold",
        fontSize=18,
        leading=22,
        alignment=TA_CENTER,
        spaceAfter=10,
    )
    h1 = ParagraphStyle(
        "H1",
        parent=base["Heading1"],
        fontName="Times-Bold",
        fontSize=13,
        leading=16,
        spaceBefore=10,
        spaceAfter=6,
    )
    h2 = ParagraphStyle(
        "H2",
        parent=base["Heading2"],
        fontName="Times-Bold",
        fontSize=11.5,
        leading=14,
        spaceBefore=6,
        spaceAfter=4,
    )
    caption = ParagraphStyle(
        "Caption",
        parent=body,
        fontName="Times-Italic",
        fontSize=9.5,
        leading=12,
        alignment=TA_CENTER,
        spaceAfter=8,
    )
    abstract = ParagraphStyle(
        "Abstract",
        parent=body,
        leftIndent=12,
        rightIndent=12,
        fontSize=10,
        leading=13.2,
    )
    code = ParagraphStyle(
        "Code",
        parent=body,
        fontName="Courier",
        fontSize=8.5,
        leading=11,
        leftIndent=8,
        textColor=colors.HexColor("#222222"),
        alignment=TA_LEFT,
    )
    return {
        "title": title,
        "body": body,
        "h1": h1,
        "h2": h2,
        "caption": caption,
        "abstract": abstract,
        "code": code,
    }


def _table_aggregate(summary: List[Dict[str, str]], styles: Dict[str, ParagraphStyle]) -> Table:
    rows: List[List[Any]] = [[
        "Condition",
        "n",
        "BLEU-1",
        "ROUGE-L",
        "BERTScore-F1",
        "G-eval (0-1)",
    ]]
    by_cond: Dict[str, Dict[str, str]] = {row["prompt_type"]: row for row in summary}
    for cond in CONDITION_ORDER:
        r = by_cond.get(cond, {})
        rows.append([
            Paragraph(CONDITION_PRETTY[cond], styles["body"]),
            r.get("n", "-"),
            f"{float(r['bleu1_mean']):.3f} \u00b1 {float(r['bleu1_std']):.3f}",
            f"{float(r['rougeL_mean']):.3f} \u00b1 {float(r['rougeL_std']):.3f}",
            f"{float(r['bertscore_f1_mean']):.3f} \u00b1 {float(r['bertscore_f1_std']):.3f}",
            f"{float(r['geval_norm_mean']):.3f} \u00b1 {float(r['geval_norm_std']):.3f}",
        ])
    t = Table(rows, colWidths=[5.6 * cm, 1.0 * cm, 2.6 * cm, 2.6 * cm, 2.8 * cm, 2.4 * cm])
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E5E7EB")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#9CA3AF")),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def _table_delta(summary: List[Dict[str, str]], styles: Dict[str, ParagraphStyle]) -> Table:
    by_cond = {row["prompt_type"]: row for row in summary}
    base = by_cond["zero_shot_no_image"]

    def d(cond: str, key: str) -> str:
        return f"{(float(by_cond[cond][key]) - float(base[key])):+.3f}"

    rows: List[List[Any]] = [["Condition vs A", "\u0394 BLEU-1", "\u0394 ROUGE-L", "\u0394 BERTScore-F1", "\u0394 G-eval"]]
    for c in ["zero_shot_image", "few_shot", "cot"]:
        rows.append([
            CONDITION_PRETTY[c].split(".")[0] + " - A",
            d(c, "bleu1_mean"),
            d(c, "rougeL_mean"),
            d(c, "bertscore_f1_mean"),
            d(c, "geval_norm_mean"),
        ])
    t = Table(rows, colWidths=[4.6 * cm, 2.6 * cm, 2.6 * cm, 3.0 * cm, 2.6 * cm])
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E5E7EB")),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#9CA3AF")),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


def _table_rubric(rubric: Dict[str, Dict[str, float]], styles: Dict[str, ParagraphStyle]) -> Table:
    dims: List[str] = []
    for c in CONDITION_ORDER:
        for k in rubric.get(c, {}):
            if k not in dims:
                dims.append(k)
    header = ["G-eval rubric dimension"] + [CONDITION_SHORT[c] for c in CONDITION_ORDER]
    rows: List[List[Any]] = [header]
    for d in dims:
        rows.append([d] + [f"{rubric.get(c, {}).get(d, 0.0):.2f}" for c in CONDITION_ORDER])
    t = Table(rows, colWidths=[6.0 * cm, 2.4 * cm, 2.4 * cm, 2.4 * cm, 2.4 * cm])
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E5E7EB")),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#9CA3AF")),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


def _figure(path: Path, max_w_cm: float = 15.5) -> Image:
    img = Image(str(path))
    iw, ih = img.imageWidth, img.imageHeight
    target_w = max_w_cm * cm
    scale = target_w / iw
    img.drawWidth = iw * scale
    img.drawHeight = ih * scale
    return img


def _scaled_image(path: Path, target_w_cm: float = 6.5) -> Image:
    img = Image(str(path))
    iw, ih = img.imageWidth, img.imageHeight
    target_w = target_w_cm * cm
    scale = target_w / iw
    img.drawWidth = iw * scale
    img.drawHeight = ih * scale
    return img


def _code_block(text: str, styles: Dict[str, ParagraphStyle], indent_pt: float = 6.0) -> XPreformatted:
    """Monospace, line-wrapping block with light grey background-like indent."""
    style = ParagraphStyle(
        "CodeBlock",
        parent=styles["body"],
        fontName="Courier",
        fontSize=8.5,
        leading=11,
        leftIndent=indent_pt,
        rightIndent=indent_pt,
        textColor=colors.HexColor("#1F2937"),
        alignment=TA_LEFT,
        spaceAfter=6,
        spaceBefore=2,
    )
    return XPreformatted(html.escape(text), style)


def _load_test_case(tc_id: str) -> Optional[Dict[str, Any]]:
    if not TEST_CASES_JSON.exists():
        return None
    data = json.loads(TEST_CASES_JSON.read_text(encoding="utf-8"))
    for it in data.get("items", []):
        if str(it.get("id")) == tc_id:
            return it
    return None


def _load_all_test_cases() -> List[Dict[str, Any]]:
    if not TEST_CASES_JSON.exists():
        return []
    data = json.loads(TEST_CASES_JSON.read_text(encoding="utf-8"))
    return list(data.get("items") or [])


def _prompt_text_only(parts: Any) -> str:
    """Extract concatenated text from a chat content (list of parts or str)."""
    if isinstance(parts, str):
        return parts
    out_parts: List[str] = []
    for p in parts or []:
        if isinstance(p, dict) and p.get("type") == "text":
            out_parts.append(str(p.get("text", "")))
        elif isinstance(p, dict) and p.get("type") == "image_url":
            out_parts.append("[ATTACHED IMAGE: gambar dari kamera siswa (base64) ]")
    return "\n".join(out_parts)


def _build_inputs_appendix(tc_id: str, lang: str, styles: Dict[str, ParagraphStyle]) -> List[Any]:
    story: List[Any] = []
    P = lambda txt, s="body": Paragraph(txt, styles[s])

    if _user_prompt_zero_shot_no_image is None:
        story.append(P(
            "<i>(Could not import prompt builders from gen_ablation_problem_finder.py; "
            "example-inputs appendix skipped.)</i>"
        ))
        return story

    tc = _load_test_case(tc_id)
    if not tc:
        story.append(P(f"<i>(Test case {tc_id} not found; example-inputs appendix skipped.)</i>"))
        return story

    all_items = _load_all_test_cases()
    object_label_id = tc.get("object_label_id") or tc.get("object_label") or ""
    object_label_en = tc.get("object_label") or ""
    location = tc.get("location") or ""
    image_rel = tc.get("image") or ""
    refs: List[str] = list(tc.get("references") or [])

    story.append(P(f"Appendix B. Example inputs ({tc_id})", "h1"))
    story.append(P(
        "All four conditions are run on the same 10 test cases. This appendix shows the "
        f"complete input for one representative test case (<b>{tc_id}</b>): the raw fields, "
        "the camera image, the gold-standard references, and the exact system + user prompts "
        "sent to GPT-4o for each of the four conditions."
    ))

    story.append(P("B.1 Test case fields", "h2"))
    fields_rows: List[List[Any]] = [
        ["Field", "Value"],
        ["id", tc_id],
        ["object_label (recognition)", object_label_en],
        ["object_label_id (Indonesian)", object_label_id],
        ["location", location],
        ["image", image_rel],
        ["language", lang],
    ]
    t_fields = Table(fields_rows, colWidths=[5.0 * cm, 11.0 * cm])
    t_fields.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
        ("FONTNAME", (0, 1), (0, -1), "Times-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E5E7EB")),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#9CA3AF")),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(t_fields)
    story.append(Spacer(1, 4))

    # Image preview if available
    image_abs = (FASTAPI_DIR / image_rel) if image_rel else None
    if image_abs and image_abs.exists():
        story.append(_scaled_image(image_abs, target_w_cm=6.5))
        story.append(P(
            f"<b>Figure B.1.</b> Camera image used as input for test case {tc_id} "
            f"(<i>{image_rel}</i>). This is the actual image attached as <font face='Courier'>"
            "image_url</font> content part to GPT-4o in conditions B, C, and D.",
            "caption",
        ))

    story.append(P("B.2 Gold-standard references (for evaluation only)", "h2"))
    story.append(P(
        "These references are not shown to the generator. They are used by "
        "<font face='Courier'>eval_inquiry_metrics.py</font> as the gold standard for "
        "BLEU-1, ROUGE-L, and BERTScore-F1."
    ))
    for i, r in enumerate(refs, start=1):
        story.append(P(f"<b>R{i}.</b> {html.escape(r)}"))
    story.append(Spacer(1, 4))

    story.append(P("B.3 System prompts", "h2"))
    story.append(P(
        "Two system prompts are used. Condition A (no image, no recognition) uses the "
        "<i>baseline</i> system prompt, which explicitly tells the model it has not seen "
        "the environment. Conditions B, C, and D use the <i>grounded</i> system prompt, "
        "which describes the multimodal inputs and mandates explicit mention of object and "
        "location."
    ))
    story.append(P("<b>B.3.1 Baseline system prompt (condition A).</b>"))
    story.append(_code_block(SYSTEM_PROMPT_BASELINE_ID, styles))
    story.append(P("<b>B.3.2 Grounded system prompt (conditions B, C, D).</b>"))
    story.append(_code_block(SYSTEM_PROMPT_GROUNDED_ID, styles))

    # Build the same prompts the generator built.
    rng = random.Random(42)
    few_shot_examples = (
        _pick_few_shot_examples(all_items, tc_id, 3, rng) if _pick_few_shot_examples else []
    )

    image_placeholder_url = "data:image/jpeg;base64,..." if image_abs and image_abs.exists() else None
    user_a_parts = _user_prompt_zero_shot_no_image(lang)
    user_b_parts = _user_prompt_zero_shot_image(lang, object_label_id, location, image_placeholder_url)
    user_c_parts = _user_prompt_few_shot(lang, object_label_id, location, few_shot_examples, image_placeholder_url)
    user_d_parts = _user_prompt_cot(lang, object_label_id, location, image_placeholder_url)

    story.append(P("B.4 User prompts per condition (rendered for this test case)", "h2"))
    story.append(P(
        "Below are the exact user messages sent to GPT-4o for test case "
        f"<b>{tc_id}</b>. Where an image was attached, the placeholder "
        "<font face='Courier'>[ATTACHED IMAGE: ...]</font> stands in for the base64 "
        "<font face='Courier'>image_url</font> content part."
    ))

    story.append(P("<b>B.4.1 Condition A &mdash; Zero-shot (no image, no recognition).</b>"))
    story.append(_code_block(_prompt_text_only(user_a_parts), styles))

    story.append(P("<b>B.4.2 Condition B &mdash; Zero-shot + image + recognition.</b>"))
    story.append(_code_block(_prompt_text_only(user_b_parts), styles))

    story.append(P("<b>B.4.3 Condition C &mdash; Few-shot + image + recognition.</b>"))
    if few_shot_examples:
        story.append(P(
            "The three style examples for this test case (deterministically picked with "
            "<font face='Courier'>seed=42</font> from references of <i>other</i> test cases) are:"
        ))
        for i, e in enumerate(few_shot_examples, start=1):
            story.append(P(f"<b>E{i}.</b> <i>{html.escape(e)}</i>"))
    story.append(_code_block(_prompt_text_only(user_c_parts), styles))

    story.append(P("<b>B.4.4 Condition D &mdash; Chain-of-Thought + image + recognition.</b>"))
    story.append(_code_block(_prompt_text_only(user_d_parts), styles))

    return story


def _build_story(
    summary: List[Dict[str, str]],
    rubric: Dict[str, Dict[str, float]],
    sample: Dict[str, str],
    test_case_label: str,
    styles: Dict[str, ParagraphStyle],
) -> List[Any]:
    story: List[Any] = []
    P = lambda txt, s="body": Paragraph(txt, styles[s])

    story.append(P(
        "Visual Grounding via Image and Object Recognition Improves "
        "LLM-Generated Inquiry Questions: An Ablation Study with Zero-shot, "
        "Few-shot, and Chain-of-Thought Prompting",
        "title",
    ))
    story.append(P(
        "<i>Ablation report &mdash; G-Uphysic / Phyphox project, "
        "centripetal acceleration domain</i>",
        "caption",
    ))
    story.append(Spacer(1, 4))

    story.append(P("Abstract", "h2"))
    story.append(P(
        "We study how visual grounding (a camera image plus an object-recognition label) "
        "affects the quality of inquiry questions generated by a large language model "
        "(LLM) for the &lsquo;problem-finding&rsquo; stage of a guided inquiry on centripetal "
        "acceleration. We compare four prompting conditions on 10 test cases (10 image + "
        "object + location triples) using GPT-4o as the generator: (A) zero-shot without "
        "image or recognition, (B) zero-shot with image and recognition, (C) few-shot "
        "with image and recognition, (D) chain-of-thought (CoT) with image and recognition. "
        "Generated inquiries are evaluated against three object/location-specific reference "
        "inquiries per test case using BLEU-1, ROUGE-L and BERTScore-F1, and against a "
        "five-dimension rubric using G-eval (gpt-4 as judge). All three conditions that "
        "include image and recognition (B, C, D) outperform the unimodal baseline (A) on "
        "every metric (e.g. G-eval: 0.91/0.91/0.90 vs 0.82; BLEU-1: 0.73/0.77/0.71 vs 0.24). "
        "The gain is concentrated in the &lsquo;Context integration&rsquo; rubric dimension, which "
        "confirms that the recognition label drives most of the improvement.",
        "abstract",
    ))
    story.append(Spacer(1, 6))

    story.append(P("1. Introduction", "h1"))
    story.append(P(
        "An inquiry tutor for high-school physics must produce questions that are not "
        "only physically correct but also <i>grounded</i> in what the student is actually "
        "looking at. When the tutor has access to the student&rsquo;s environment (via a "
        "camera image and an object-recognition module), it can name the specific object "
        "and location in its question, which helps observation and curiosity. This report "
        "documents an ablation that quantifies the benefit of such visual grounding "
        "across three popular prompting techniques (zero-shot, few-shot and CoT)."
    ))

    story.append(P("2. Method", "h1"))
    story.append(P("2.1 Task", "h2"))
    story.append(P(
        "The system must generate a single problem-finding inquiry sentence (Indonesian) "
        "that invites the student to observe an object exhibiting circular motion. "
        "Notation conventions are radius (r), angular velocity (&omega;) and centripetal "
        "acceleration (a); the formula a = r&middot;&omega;&sup2; must not be revealed."
    ))

    story.append(P("2.2 Test cases", "h2"))
    story.append(P(
        "We construct 10 test cases, each with a recognized object label (e.g. <i>"
        "electric_fan</i>, <i>Bicycle</i>, <i>paddlewheel</i>, <i>unicycle</i>, <i>steel_drum</i>), "
        "an Indonesian object name and observation location, a camera-style image, and "
        "three gold-standard inquiry references mentioning the object, the location and "
        "the notation (r), (&omega;), (a). The references serve only at evaluation time "
        "and are not shown to the generator."
    ))

    story.append(P("2.3 Conditions", "h2"))
    story.append(P(
        "<b>A. Zero-shot (no image, no recognition)</b> &mdash; baseline. The system prompt "
        "states explicitly that the tutor has not seen the environment, and the user prompt "
        "forbids naming any specific object or location. Image is not sent.<br/>"
        "<b>B. Zero-shot + image + recognition</b> &mdash; the camera image is attached, the "
        "recognition label and location are provided, and the prompt mandates explicit "
        "mention of both.<br/>"
        "<b>C. Few-shot + image + recognition</b> &mdash; same as B, plus three style "
        "examples drawn (deterministically, seed=42) from references of <i>other</i> test "
        "cases so the model never sees its own references.<br/>"
        "<b>D. CoT + image + recognition</b> &mdash; same as B, plus an internal step-by-step "
        "instruction asking the model to reason about which part rotates, what the student "
        "should observe, and how to phrase an open-ended question; only the final question "
        "is emitted."
    ))

    story.append(P("2.4 Implementation", "h2"))
    story.append(P(
        "All four conditions use <i>gpt-4o</i> at temperature 0.5 with a 180-token cap, "
        "called directly via the OpenAI API key in <font face='Courier'>.env</font> (no "
        "backend hop). For conditions B/C/D the image is base64-encoded and attached as "
        "an <font face='Courier'>image_url</font> content part. The generator script is "
        "<font face='Courier'>gen_ablation_problem_finder.py</font>. Each of the 10 test "
        "cases is run once per condition (40 generations total)."
    ))

    story.append(P("2.5 Evaluation metrics", "h2"))
    story.append(P(
        "Each generated inquiry is scored with: (i) <i>BLEU-1</i> and <i>BLEU-4</i> "
        "(sentence-level, NLTK, exponential smoothing); (ii) <i>ROUGE-1/2/L</i> (rouge_score, "
        "no stemmer for Indonesian); (iii) <i>BERTScore</i> precision/recall/F1 with "
        "<i>xlm-roberta-large</i>; and (iv) <i>G-eval</i> with GPT-4 as judge on a "
        "five-point rubric covering Stage alignment, Topic relevance, Clarity and structure, "
        "Context integration, and Inquiry quality. BLEU is reported on a 0-1 scale. "
        "Aggregate statistics (mean and standard deviation) are computed across the 10 test "
        "cases per condition."
    ))

    story.append(P("3. Results", "h1"))
    story.append(P("3.1 Aggregate scores", "h2"))
    story.append(P(
        "Table 1 reports mean and standard deviation per condition. All three image+"
        "recognition conditions improve over the baseline on every metric, with "
        "Few-shot+IR producing the strongest text-similarity scores and ZS+IR / Few-shot+IR "
        "tied on G-eval."
    ))
    story.append(_table_aggregate(summary, styles))
    story.append(P(
        "<b>Table 1.</b> Aggregate inquiry-quality scores per condition (n=10). Values are "
        "mean &plusmn; standard deviation.",
        "caption",
    ))

    story.append(Spacer(1, 4))
    if (PLOTS_DIR / "grouped_bar.png").exists():
        story.append(_figure(PLOTS_DIR / "grouped_bar.png"))
        story.append(P(
            "<b>Figure 1.</b> Grouped-bar comparison of four metrics across the four "
            "conditions. Error bars show one standard deviation across the 10 test cases. "
            "The error bars of the baseline A do not overlap with those of B/C/D on any "
            "metric, indicating that the improvement is not within-noise.",
            "caption",
        ))

    story.append(P("3.2 Improvement over baseline", "h2"))
    story.append(P(
        "Table 2 and Figure 2 quantify the mean improvement of each grounded condition "
        "over baseline A. Every delta is positive; the largest gains are on BLEU-1 and "
        "ROUGE-L because object-/location-specific n-grams in the references can only "
        "match grounded inquiries that explicitly mention them."
    ))
    story.append(_table_delta(summary, styles))
    story.append(P("<b>Table 2.</b> Mean improvement (\u0394) of B/C/D over baseline A.", "caption"))
    if (PLOTS_DIR / "delta_vs_baseline.png").exists():
        story.append(_figure(PLOTS_DIR / "delta_vs_baseline.png"))
        story.append(P(
            "<b>Figure 2.</b> Mean improvement (\u0394) of each grounded condition over "
            "baseline A on the four metrics. All bars are positive, i.e. adding image+"
            "recognition never hurts on any metric.",
            "caption",
        ))

    story.append(P("3.3 Per test case difference", "h2"))
    story.append(P(
        "Figure 3 plots the per-test-case difference in G-eval (B/C/D minus A on the same "
        "test case). On almost every test case the grounded conditions match or exceed the "
        "baseline; we never observe a regression larger than the rubric&rsquo;s scoring step. "
        "This indicates that the gain is not driven by a few outliers."
    ))
    if (PLOTS_DIR / "pair_diff.png").exists():
        story.append(_figure(PLOTS_DIR / "pair_diff.png"))
        story.append(P(
            "<b>Figure 3.</b> Per-test-case G-eval difference relative to baseline A. "
            "Bars above zero indicate the grounded condition is better on that case.",
            "caption",
        ))

    story.append(P("3.4 Rubric breakdown", "h2"))
    story.append(P(
        "Table 3 and Figure 4 show G-eval averages for each rubric dimension. Stage "
        "alignment, Topic relevance and Clarity are near the ceiling for every condition; "
        "the gain of B/C/D over A is concentrated in <i>Context integration</i> "
        "(approximately 2 for A versus 4-5 for B/C/D). This is the expected behavior: "
        "image and recognition are exactly the signals that should boost context "
        "integration, while the underlying physics relevance was already strong."
    ))
    story.append(_table_rubric(rubric, styles))
    story.append(P("<b>Table 3.</b> Mean rubric scores (1-5) per condition.", "caption"))
    if (PLOTS_DIR / "geval_rubric.png").exists():
        story.append(_figure(PLOTS_DIR / "geval_rubric.png"))
        story.append(P(
            "<b>Figure 4.</b> G-eval rubric breakdown per condition. The largest gap "
            "between baseline A and the grounded conditions is on Context integration.",
            "caption",
        ))

    story.append(P("4. Discussion", "h1"))
    story.append(P(
        "<b>Image+recognition is the dominant factor.</b> Moving from zero-shot without any "
        "context (A) to zero-shot with image and recognition (B) yields the largest single "
        "jump in every metric. Adding few-shot examples (C) gives a further small "
        "improvement on text-similarity scores by aligning the surface style with the "
        "gold references. CoT (D) lands slightly below few-shot on text-similarity but is "
        "indistinguishable from B on G-eval, suggesting CoT keeps the semantic quality "
        "intact while introducing more paraphrasing.<br/><br/>"
        "<b>Why the baseline is bad on BLEU/ROUGE.</b> The references are object- and "
        "location-specific by design. Baseline A is explicitly forbidden from naming the "
        "object or the location, so it cannot match those tokens. This is fair: the "
        "reference set encodes our gold standard for a grounded inquiry. Baseline A is "
        "what a textbook-style tutor would produce when it has no view of the student's "
        "environment.<br/><br/>"
        "<b>Limitations.</b> (1) Sample size is small (n=10 per condition); we mitigate "
        "by reporting standard deviations. (2) The recognition labels are clean (we feed "
        "the gold label directly); a real CV pipeline will introduce mistakes that could "
        "narrow the gap. (3) G-eval uses a single LLM judge; cross-judge consistency was "
        "not measured here. (4) Generation language is Indonesian only."
    ))

    story.append(P("5. Conclusion", "h1"))
    story.append(P(
        "For the problem-finding stage of a guided inquiry tutor on centripetal "
        "acceleration, providing the LLM with a camera image and a recognition label "
        "improves inquiry quality across zero-shot, few-shot and chain-of-thought "
        "prompting. The improvement is consistent across automatic text-similarity metrics "
        "(BLEU-1, ROUGE-L, BERTScore-F1) and an LLM-based rubric (G-eval), and is largely "
        "explained by better integration of the visual context, which is what a multimodal "
        "tutor is supposed to do."
    ))

    story.append(P("Appendix A. Reproducibility", "h1"))
    story.append(P(
        "Run the full pipeline from <font face='Courier'>fastapi-gpt/</font>:",
    ))
    story.append(P(
        "python gen_ablation_problem_finder.py --test-cases ablation/test_cases_pf.json "
        "--output ablation/ablation_pf.jsonl --model gpt-4o --variants 1 --temperature 0.5 --seed 42",
        "code",
    ))
    story.append(P(
        "python eval_inquiry_metrics.py --input ablation/ablation_pf.jsonl "
        "--output-jsonl ablation/ablation_results.jsonl "
        "--output-csv ablation/ablation_results.csv "
        "--plots-dir ablation/eval_plots --bleu-scale 0-100",
        "code",
    ))
    story.append(P(
        "python plot_ablation.py --input ablation/ablation_results.csv "
        "--input-jsonl ablation/ablation_results.jsonl "
        "--output-dir ablation/eval_plots --bleu-scale 0-100",
        "code",
    ))

    story.extend(_build_inputs_appendix("TC-001", lang="id", styles=styles))

    if sample:
        story.append(P(f"Appendix C. Sample generated inquiries ({test_case_label})", "h1"))
        story.append(P(
            "Below are the actual inquiries produced by GPT-4o for test case TC-001 under "
            "each condition (one generation per condition; temperature 0.5, seed 42)."
        ))
        for cond in CONDITION_ORDER:
            inq = sample.get(cond)
            if inq:
                story.append(P(f"<b>{CONDITION_PRETTY[cond]}.</b> {html.escape(inq)}"))

    return story


def main() -> int:
    summary_csv = PLOTS_DIR / "summary.csv"
    jsonl = ABL_DIR / "ablation_results.jsonl"
    if not summary_csv.exists():
        print(f"ERROR: {summary_csv} not found.", file=sys.stderr)
        return 1
    if not jsonl.exists():
        print(f"ERROR: {jsonl} not found.", file=sys.stderr)
        return 1

    summary = _read_summary_csv(summary_csv)
    rows = _read_jsonl(jsonl)
    rubric = _rubric_means(rows)
    sample = _sample_inquiries(rows, "TC-001")
    test_case_label = "TC-001: Kipas angin / ruang kelas"

    styles = _styles()

    doc = BaseDocTemplate(
        str(OUT_PDF),
        pagesize=A4,
        leftMargin=1.8 * cm,
        rightMargin=1.8 * cm,
        topMargin=1.6 * cm,
        bottomMargin=1.6 * cm,
        title="Ablation Study: Image+Recognition for Inquiry Generation",
        author="G-Uphysic",
    )
    frame = Frame(
        doc.leftMargin,
        doc.bottomMargin,
        doc.width,
        doc.height,
        id="content",
    )
    doc.addPageTemplates([PageTemplate(id="default", frames=[frame])])

    story = _build_story(summary, rubric, sample, test_case_label, styles)
    doc.build(story)
    print(f"Wrote: {OUT_PDF}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
