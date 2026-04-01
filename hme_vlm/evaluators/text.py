from __future__ import annotations

from collections import Counter
import math

from hme_vlm.normalization import canonicalize_latex_for_metrics, tokenize_latex_for_bleu


def exact_match(gold: str, pred: str) -> bool:
    return canonicalize_latex_for_metrics(gold) == canonicalize_latex_for_metrics(pred)


def levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (char_a != char_b)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


def character_error_rate(gold: str, pred: str) -> float:
    gold_text = canonicalize_latex_for_metrics(gold)
    pred_text = canonicalize_latex_for_metrics(pred)
    if not gold_text:
        return 0.0 if not pred_text else 1.0
    return levenshtein_distance(gold_text, pred_text) / len(gold_text)


def edit_score(gold: str, pred: str) -> float:
    gold_text = canonicalize_latex_for_metrics(gold)
    pred_text = canonicalize_latex_for_metrics(pred)
    denom = max(len(gold_text), len(pred_text), 1)
    return 1.0 - (levenshtein_distance(gold_text, pred_text) / denom)


def bleu4(gold: str, pred: str) -> float:
    reference = tokenize_latex_for_bleu(gold)
    candidate = tokenize_latex_for_bleu(pred)

    if not reference and not candidate:
        return 1.0
    if not candidate:
        return 0.0

    precisions: list[float] = []
    for n in range(1, 5):
        if len(candidate) < n:
            precisions.append(0.0)
            continue

        candidate_counts = Counter(tuple(candidate[i : i + n]) for i in range(len(candidate) - n + 1))
        reference_counts = Counter(tuple(reference[i : i + n]) for i in range(len(reference) - n + 1))

        overlap = 0
        total = 0
        for gram, count in candidate_counts.items():
            overlap += min(count, reference_counts.get(gram, 0))
            total += count

        precisions.append((overlap + 1.0) / (total + 1.0))

    if any(value <= 0.0 for value in precisions):
        return 0.0

    if len(candidate) > len(reference):
        brevity_penalty = 1.0
    else:
        brevity_penalty = math.exp(1.0 - (len(reference) / max(len(candidate), 1)))

    score = brevity_penalty * math.exp(sum(math.log(value) for value in precisions) / 4.0)
    return float(score)
