"""Corpora. `synthetic` is a tiny fictional enterprise wiki with gold relevance; `loader` reads a
generic JSONL benchmark (documents + queries) such as an Enterprise RAG Bench export."""

from entsearch.datasets import loader, synthetic

__all__ = ["loader", "synthetic"]
