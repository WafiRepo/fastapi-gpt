"""
Path ke PDF bank soal untuk few-shot prompting.
Set env FEW_SHOT_QUESTIONS_PDF ke file absolut, atau letakkan few_shot_questions.pdf di folder fastapi-gpt.
"""
import os
from typing import Optional


def resolve_few_shot_pdf_path(script_dir: str, logger) -> Optional[str]:
    env = (os.getenv("FEW_SHOT_QUESTIONS_PDF") or "").strip()
    if env:
        p = os.path.abspath(os.path.expanduser(env))
    else:
        p = os.path.join(script_dir, "few_shot_questions.pdf")
    if os.path.isfile(p):
        return p
    logger.warning(
        "Few-shot PDF tidak ditemukan di %s — melanjutkan tanpa contoh dari PDF.",
        p,
    )
    return None
