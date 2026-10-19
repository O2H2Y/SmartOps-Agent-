"""RAG retrieval evaluation CLI.

Examples:
    python -m eval.rag_eval stats
    python -m eval.rag_eval retrieve --top-k 10 --index-dir aiops-docs
    python -m eval.rag_eval score --results eval/results/retrieval_xxx.jsonl --k 3
"""

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from eval.metrics import aggregate_scores, group_by_category, normalize_text, score_query

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "eval" / "rag_eval_dataset.jsonl"
DEFAULT_CORPUS_DIR = ROOT / "aiops-docs"
DEFAULT_RESULTS_DIR = ROOT / "eval" / "results"


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_markdown_sections(path: Path) -> Dict[str, str]:
    sections: Dict[str, List[str]] = {}
    current: Optional[str] = None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("## "):
                current = line[3:].strip()
                sections.setdefault(current, [])
            elif current is not None:
                sections[current].append(line)
    return {name: "".join(lines) for name, lines in sections.items()}


def parse_markdown_subsections(section_text: str) -> Dict[str, str]:
    subsections: Dict[str, List[str]] = {}
    current: Optional[str] = None
    for line in section_text.splitlines(keepends=True):
        if line.startswith("### "):
            current = line[4:].strip()
            subsections.setdefault(current, [line])
        elif current is not None:
            subsections[current].append(line)
    return {name: "".join(lines) for name, lines in subsections.items()}


def resolve_source_path(source: str, corpus_dir: Path) -> Path:
    source_path = Path(source)
    if source_path.is_absolute():
        return source_path
    parts = source_path.parts
    if parts and parts[0] == corpus_dir.name:
        source_path = Path(*parts[1:])
    candidate = corpus_dir / source_path
    return candidate if candidate.exists() else ROOT / source


def validate_dataset(
    dataset: Sequence[Dict[str, Any]],
    corpus_dir: Path,
) -> List[str]:
    errors: List[str] = []
    seen_qids = set()
    for row in dataset:
        qid = str(row.get("qid") or "")
        if not qid:
            errors.append("row without qid")
            continue
        if qid in seen_qids:
            errors.append(f"{qid}: duplicate qid")
        seen_qids.add(qid)
        if not str(row.get("query") or "").strip():
            errors.append(f"{qid}: empty query")
        relevant = row.get("relevant")
        if not isinstance(relevant, list):
            errors.append(f"{qid}: relevant must be a list")
            continue
        if row.get("category") == "negative" and relevant:
            errors.append(f"{qid}: negative query must have no relevant items")
        if row.get("category") != "negative" and not relevant:
            errors.append(f"{qid}: positive query must have relevant items")

        section_cache: Dict[Path, Dict[str, str]] = {}
        for index, item in enumerate(relevant):
            source = resolve_source_path(str(item.get("source") or ""), corpus_dir)
            evidence = normalize_text(str(item.get("evidence_quote") or ""))
            h2 = str(item.get("h2") or "").strip()
            h3 = str(item.get("h3") or "").strip()
            grade = item.get("grade")
            if (
                isinstance(grade, bool)
                or not isinstance(grade, int)
                or grade <= 0
            ):
                errors.append(f"{qid}[{index}]: grade must be a positive integer")
            if not source.is_file():
                errors.append(f"{qid}[{index}]: source not found: {source}")
                continue
            if source not in section_cache:
                section_cache[source] = parse_markdown_sections(source)
            sections = section_cache[source]
            if h2 not in sections:
                errors.append(f"{qid}[{index}]: h2 not found: {h2}")
                continue
            target_text = sections[h2]
            if h3:
                subsections = parse_markdown_subsections(target_text)
                if h3 not in subsections:
                    errors.append(f"{qid}[{index}]: h3 not found in '{h2}': {h3}")
                    continue
                target_text = subsections[h3]
            if not evidence:
                errors.append(f"{qid}[{index}]: empty evidence_quote")
            elif evidence not in normalize_text(target_text):
                location = f"{h2} / {h3}" if h3 else h2
                errors.append(f"{qid}[{index}]: evidence not in '{location}'")
    return errors


def dataset_stats(dataset: Sequence[Dict[str, Any]], corpus_dir: Path) -> Dict[str, Any]:
    category_counts: Dict[str, int] = {}
    intent_counts: Dict[str, int] = {}
    grade_counts: Dict[str, int] = {}
    negative_type_counts: Dict[str, int] = {}
    category_relevant_links: Dict[str, int] = {}
    relevant_links = 0
    sources = set()
    sections = set()
    subsections = set()
    for row in dataset:
        category = str(row.get("category") or "unknown")
        category_counts[category] = category_counts.get(category, 0) + 1
        intent = str(row.get("intent") or "unknown")
        intent_counts[intent] = intent_counts.get(intent, 0) + 1
        negative_type = row.get("negative_type")
        if negative_type:
            negative_type_counts[str(negative_type)] = (
                negative_type_counts.get(str(negative_type), 0) + 1
            )
        for item in row.get("relevant") or []:
            relevant_links += 1
            category_relevant_links[category] = category_relevant_links.get(category, 0) + 1
            source = str(item.get("source") or "")
            sources.add(source)
            sections.add((source, str(item.get("h2") or "")))
            if item.get("h3"):
                subsections.add(
                    (source, str(item.get("h2") or ""), str(item.get("h3") or ""))
                )
            grade = str(item.get("grade", 1))
            grade_counts[grade] = grade_counts.get(grade, 0) + 1

    corpus_files = sorted(corpus_dir.glob("*.md")) + sorted(corpus_dir.glob("*.txt"))
    total_chars = sum(len(path.read_text(encoding="utf-8")) for path in corpus_files)
    total_bytes = sum(path.stat().st_size for path in corpus_files)
    total_h2 = 0
    for path in corpus_files:
        text = path.read_text(encoding="utf-8")
        total_h2 += len(re.findall(r"(?m)^## ", text))

    positive_queries = sum(1 for row in dataset if row.get("category") != "negative")
    negative_queries = sum(1 for row in dataset if row.get("category") == "negative")
    return {
        "corpus_files": len(corpus_files),
        "corpus_chars": total_chars,
        "corpus_bytes": total_bytes,
        "corpus_h2_sections": total_h2,
        "queries": len(dataset),
        "positive_queries": positive_queries,
        "negative_queries": negative_queries,
        "relevant_links": relevant_links,
        "category_relevant_links": dict(sorted(category_relevant_links.items())),
        "covered_sections": len(sections),
        "covered_subsections": len(subsections),
        "grade_counts": dict(sorted(grade_counts.items())),
        "negative_type_counts": dict(sorted(negative_type_counts.items())),
        "categories": dict(sorted(category_counts.items())),
        "intent_types": len(intent_counts),
        "covered_sources": len(sources),
    }


def load_dataset(path: Path, corpus_dir: Path) -> List[Dict[str, Any]]:
    dataset = read_jsonl(path)
    errors = validate_dataset(dataset, corpus_dir)
    if errors:
        formatted = "\n".join(f"- {error}" for error in errors)
        raise ValueError(f"dataset validation failed:\n{formatted}")
    return dataset


def load_candidates(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    candidates: Dict[str, List[Dict[str, Any]]] = {}
    for row in read_jsonl(path):
        qid = str(row.get("qid") or "")
        if not qid:
            raise ValueError(f"{path}: result row without qid")
        if qid in candidates:
            raise ValueError(f"{path}: duplicate qid: {qid}")
        candidates[qid] = list(row.get("candidates") or [])
    return candidates


def score_dataset(
    dataset: Sequence[Dict[str, Any]],
    candidates_by_qid: Dict[str, List[Dict[str, Any]]],
    k: int,
) -> Dict[str, Any]:
    per_query = []
    missing = []
    for row in dataset:
        qid = row["qid"]
        if qid not in candidates_by_qid:
            missing.append(qid)
            continue
        per_query.append(score_query(row, candidates_by_qid[qid], k))
    if missing:
        preview = ", ".join(missing[:10])
        suffix = "..." if len(missing) > 10 else ""
        raise ValueError(f"missing retrieval results for {len(missing)} queries: {preview}{suffix}")
    summary = aggregate_scores(per_query)
    summary["k"] = k
    summary["evaluated_queries"] = len(per_query)
    summary["missing_result_queries"] = missing
    return {
        "summary": summary,
        "by_category": group_by_category(per_query),
        "per_query": per_query,
    }


def _load_retrieval_dependencies():
    try:
        from app.config import config
        from app.services.vector_store_manager import vector_store_manager
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "retrieval requires the project's Python 3.11+ environment and dependencies; "
            "install with `uv sync` or `pip install -e .`"
        ) from exc
    return config, vector_store_manager


def _candidate_from_pair(rank: int, pair: Tuple[Any, float]) -> Dict[str, Any]:
    document, score = pair
    return {
        "rank": rank,
        "content": document.page_content,
        "score": float(score),
        "metadata": dict(document.metadata),
    }


def run_retrieval(
    dataset: Sequence[Dict[str, Any]],
    top_k: int,
    output: Path,
    index_dir: Optional[Path],
) -> Dict[str, Any]:
    config, vector_store_manager = _load_retrieval_dependencies()
    if index_dir is not None:
        from app.services.vector_index_service import vector_index_service

        result = vector_index_service.index_directory(str(index_dir))
        if not result.success:
            raise RuntimeError(f"indexing failed: {result.to_dict()}")

    vector_store = vector_store_manager.get_vector_store()
    rows = []
    for row in dataset:
        pairs = vector_store.similarity_search_with_score(row["query"], k=top_k)
        candidates = [
            _candidate_from_pair(rank, pair)
            for rank, pair in enumerate(pairs, start=1)
        ]
        rows.append(
            {
                "qid": row["qid"],
                "query": row["query"],
                "category": row.get("category", ""),
                "candidates": candidates,
            }
        )

    write_jsonl(output, rows)
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": str(DEFAULT_DATASET),
        "output": str(output),
        "queries": len(dataset),
        "top_k": top_k,
        "rag_top_k": config.rag_top_k,
        "embedding_model": config.dashscope_embedding_model,
        "metric_type": "L2",
        "index_type": "IVF_FLAT",
        "nlist": 128,
        "indexed_directory": str(index_dir) if index_dir else None,
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {"trace": str(output), "manifest": str(manifest_path), **manifest}


def build_parser() -> argparse.ArgumentParser:
    def positive_int(value: str) -> int:
        parsed = int(value)
        if parsed <= 0:
            raise argparse.ArgumentTypeError("value must be a positive integer")
        return parsed

    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval quality")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    subparsers = parser.add_subparsers(dest="command", required=True)

    stats = subparsers.add_parser(
        "stats",
        help="validate the dataset and print corpus/query statistics",
    )
    stats.add_argument("--output", type=Path, default=None)

    retrieve = subparsers.add_parser("retrieve", help="run retrieval and write a trace")
    retrieve.add_argument("--top-k", type=positive_int, default=10)
    retrieve.add_argument("--index-dir", type=Path, default=None)
    retrieve.add_argument("--output", type=Path, default=None)

    score = subparsers.add_parser("score", help="calculate metrics from a retrieval trace")
    score.add_argument("--results", type=Path, required=True)
    score.add_argument("--k", type=positive_int, default=3)
    score.add_argument("--output", type=Path, default=None)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        dataset = load_dataset(args.dataset, args.corpus_dir)
        if args.command == "stats":
            serialized = json.dumps(
                dataset_stats(dataset, args.corpus_dir),
                ensure_ascii=False,
                indent=2,
            )
            print(serialized)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(serialized + "\n", encoding="utf-8")
            return 0

        if args.command == "retrieve":
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output = args.output or DEFAULT_RESULTS_DIR / f"retrieval_{timestamp}.jsonl"
            result = run_retrieval(dataset, args.top_k, output, args.index_dir)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0

        if args.command == "score":
            result = score_dataset(dataset, load_candidates(args.results), args.k)
            serialized = json.dumps(result, ensure_ascii=False, indent=2)
            print(serialized)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(serialized + "\n", encoding="utf-8")
            return 0
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}")
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
