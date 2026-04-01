from __future__ import annotations

import re


def clean_model_text(text: str) -> str:
    if text is None:
        return ""

    text = text.strip()
    text = text.replace("```latex", "").replace("```", "").strip()

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        text = lines[0]

    text = re.sub(r"^(latex\s*:\s*)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^(answer\s*:\s*)", "", text, flags=re.IGNORECASE)
    return text.strip()


def normalize_crohme_latex(text: str) -> str:
    if text is None:
        return ""

    text = text.strip()
    text = text.replace("\\n", " ")
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def canonicalize_latex_for_metrics(text: str) -> str:
    text = normalize_crohme_latex(text)
    text = re.sub(r"\s+", "", text)
    return text


def wrap_latex_in_math_mode(text: str) -> str:
    text = normalize_crohme_latex(text)
    if not text:
        return ""

    if (
        (text.startswith("$$") and text.endswith("$$"))
        or (text.startswith("$") and text.endswith("$"))
        or (text.startswith(r"\(") and text.endswith(r"\)"))
        or (text.startswith(r"\[") and text.endswith(r"\]"))
        or text.startswith(r"\boxed{")
    ):
        return text

    return f"${text}$"


def tokenize_latex_for_bleu(text: str) -> list[str]:
    canonical = canonicalize_latex_for_metrics(text)
    if not canonical:
        return []

    return re.findall(r"\\[A-Za-z]+|\\.|[A-Za-z]+|\d+|.", canonical)
