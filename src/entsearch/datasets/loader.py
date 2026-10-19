"""Loaders for real corpora.

Generic JSONL (an Enterprise RAG Bench-style export):
    documents.jsonl  {"id", "text", "title"?, "source"?, "metadata"?}
    queries.jsonl    {"id", "text", "relevant_doc_ids"? | "relevant_chunk_ids"?, "answer_spans"?,
                      "answerable"?}

BEIR (SciFact, FiQA, NFCorpus, ... — `corpus.jsonl`, `queries.jsonl`, `qrels/<split>.tsv`), the
standard way to put numbers on a retriever. BEIR judges *documents*, so cases carry
`relevant_doc_ids` and are scored by ranking documents by their best chunk; graded judgements
(NFCorpus, TREC-COVID) keep their grades, and nDCG uses them as gains, as trec_eval does.

Duplicate document ids are not checked here: `SearchEngine.index` refuses them, one place.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from entsearch.domain import Document, Query, QueryCase


def _jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_documents(path: str | Path) -> list[Document]:
    return [
        Document(
            id=str(obj["id"]),
            text=obj["text"],
            title=obj.get("title", ""),
            source=obj.get("source", ""),
            metadata=obj.get("metadata", {}),
        )
        for obj in _jsonl(Path(path))
    ]


def load_cases(path: str | Path) -> list[QueryCase]:
    return [
        QueryCase(
            query=Query(id=str(obj["id"]), text=obj["text"]),
            relevant_doc_ids={str(d) for d in obj.get("relevant_doc_ids", [])},
            relevant_chunk_ids=set(obj.get("relevant_chunk_ids", [])),
            answer_spans=obj.get("answer_spans", []),
            answerable=obj.get("answerable", True),
        )
        for obj in _jsonl(Path(path))
    ]


def load_beir(directory: str | Path, split: str = "test") -> tuple[list[Document], list[QueryCase]]:
    """A BEIR dataset directory -> (documents, cases with document-level relevance). Only
    queries with at least one positive judgement in `split` become cases."""
    root = Path(directory)
    docs = [
        Document(id=str(o["_id"]), title=o.get("title", ""), text=o.get("text", ""), source="beir")
        for o in _jsonl(root / "corpus.jsonl")
    ]
    queries = {str(o["_id"]): str(o["text"]) for o in _jsonl(root / "queries.jsonl")}
    grades: dict[str, dict[str, int]] = {}
    with (root / "qrels" / f"{split}.tsv").open() as f:
        rows = csv.reader(f, delimiter="\t")
        next(rows, None)  # header: query-id, corpus-id, score
        for qid, did, score in rows:
            if int(score) > 0:
                grades.setdefault(qid, {})[did] = int(score)
    cases = [
        QueryCase(query=Query(id=qid, text=queries[qid]), relevant_doc_ids=set(g), grades=g)
        for qid, g in sorted(grades.items())
        if qid in queries
    ]
    return docs, cases
