"""Dependency-free retrieval metrics for RAG evaluations."""

import math
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


def normalize_text(value: str) -> str:
    """Normalize whitespace while preserving the source wording."""
    return " ".join((value or "").replace("\u00a0", " ").split())


def source_matches(actual_source: str, expected_source: str) -> bool:
    """Match full paths and file names without depending on the workspace root."""
    if not actual_source or not expected_source:
        return False
    actual = Path(actual_source.replace("\\", "/")).as_posix()
    expected = Path(expected_source.replace("\\", "/")).as_posix()
    if actual == expected or actual.endswith("/" + expected):
        return True
    return "/" not in expected and Path(actual).name == expected


def candidate_source(candidate: Dict[str, Any]) -> str:
    metadata = candidate.get("metadata") or {}
    return str(
        candidate.get("source")
        or metadata.get("_source")
        or metadata.get("_file_name")
        or ""
    )


def group_relevant_units(
    relevant: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Collapse evidence snippets from the same section into one ranking unit."""
    grouped: "OrderedDict[Tuple[str, str, str], Dict[str, Any]]" = OrderedDict()
    for item in relevant:
        source = str(item.get("source") or "")
        h2 = str(item.get("h2") or "")
        h3 = str(item.get("h3") or "")
        key = (source, h2, h3)
        unit = grouped.setdefault(
            key,
            {
                "source": source,
                "h2": h2,
                "h3": h3,
                "grade": 0,
                "evidence_quotes": [],
            },
        )
        unit["grade"] = max(unit["grade"], int(item.get("grade", 1)))
        unit["evidence_quotes"].append(str(item.get("evidence_quote") or ""))
    return list(grouped.values())


def matched_unit_indexes(
    candidate: Dict[str, Any],
    units: Sequence[Dict[str, Any]],
) -> List[int]:
    """Return relevant unit indexes whose evidence is contained in a candidate."""
    content = normalize_text(str(candidate.get("content") or ""))
    source = candidate_source(candidate)
    matched: List[int] = []
    for index, unit in enumerate(units):
        if not source_matches(source, str(unit.get("source") or "")):
            continue
        evidence_quotes = [
            normalize_text(str(evidence))
            for evidence in unit.get("evidence_quotes") or []
        ]
        if any(evidence and evidence in content for evidence in evidence_quotes):
            matched.append(index)
    return matched


def dcg(grades: Iterable[int]) -> float:
    return sum((2**grade - 1) / math.log2(rank + 2) for rank, grade in enumerate(grades))


def score_query(
    row: Dict[str, Any],
    candidates: Sequence[Dict[str, Any]],
    k: int,
) -> Dict[str, Any]:
    """Score one query against an ordered candidate list."""
    if k <= 0:
        raise ValueError("k must be a positive integer")
    top_k = list(candidates[:k])
    relevant = list(row.get("relevant") or [])
    is_negative = not relevant
    units = group_relevant_units(relevant)

    if is_negative:
        returned = 1.0 if top_k else 0.0
        return {
            "qid": row["qid"],
            "category": row.get("category", ""),
            "negative": True,
            "relevant_count": 0,
            "relevant_unit_count": 0,
            "retrieved_count": len(top_k),
            "recall": None,
            "hit_rate": None,
            "mrr": None,
            "ndcg": None,
            "doc_recall": None,
            "empty_result": 1.0 - returned,
            "no_answer_return": returned,
            "first_relevant_rank": None,
        }

    hit_indexes: Set[int] = set()
    first_relevant_rank: Optional[int] = None
    candidate_grades: List[int] = []
    doc_hit = False

    for rank, candidate in enumerate(top_k, start=1):
        matched = matched_unit_indexes(candidate, units)
        new_matches = [index for index in matched if index not in hit_indexes]
        if new_matches:
            hit_indexes.update(matched)
            if first_relevant_rank is None:
                first_relevant_rank = rank
            candidate_grades.append(
                max(int(units[i].get("grade", 1)) for i in new_matches)
            )
            doc_hit = True
        else:
            candidate_grades.append(0)

    ideal_grades = sorted((int(unit.get("grade", 1)) for unit in units), reverse=True)
    ideal_dcg = dcg(ideal_grades[:k])
    recall = len(hit_indexes) / len(units)
    hit_rate = 1.0 if hit_indexes else 0.0
    mrr = 1.0 / first_relevant_rank if first_relevant_rank else 0.0
    ndcg = dcg(candidate_grades) / ideal_dcg if ideal_dcg else 0.0

    return {
        "qid": row["qid"],
        "category": row.get("category", ""),
        "negative": False,
        "relevant_count": len(relevant),
        "relevant_unit_count": len(units),
        "retrieved_count": len(top_k),
        "recall": recall,
        "hit_rate": hit_rate,
        "mrr": mrr,
        "ndcg": ndcg,
        "doc_recall": 1.0 if doc_hit else 0.0,
        "empty_result": 1.0 if not top_k else 0.0,
        "no_answer_return": None,
        "first_relevant_rank": first_relevant_rank,
    }


def average(values: Iterable[Optional[float]]) -> Optional[float]:
    present = [float(value) for value in values if value is not None]
    return sum(present) / len(present) if present else None


def aggregate_scores(per_query: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    positives = [row for row in per_query if not row["negative"]]
    negatives = [row for row in per_query if row["negative"]]
    return {
        "positive_queries": len(positives),
        "negative_queries": len(negatives),
        "doc_recall": average(row["doc_recall"] for row in positives),
        "chunk_recall": average(row["recall"] for row in positives),
        "hit_rate": average(row["hit_rate"] for row in positives),
        "mrr": average(row["mrr"] for row in positives),
        "ndcg": average(row["ndcg"] for row in positives),
        "positive_empty_result_rate": average(row["empty_result"] for row in positives),
        "no_answer_return_rate": average(row["no_answer_return"] for row in negatives),
    }


def group_by_category(per_query: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in per_query:
        grouped.setdefault(str(row.get("category") or "unknown"), []).append(row)
    return {category: aggregate_scores(rows) for category, rows in sorted(grouped.items())}
