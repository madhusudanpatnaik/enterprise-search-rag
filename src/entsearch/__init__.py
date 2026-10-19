"""Enterprise Search — hybrid retrieval with verified citations.

The talk's point is that enterprise RAG is not "embed, top-k, prompt". It is: chunk deliberately,
retrieve two ways (lexical + semantic), fuse, re-rank with a cross-encoder, and then *refuse to
answer* anything the retrieved text does not support. This package is that pipeline, with the
retrieval math and the citation gate written in real code and measured (recall, latency P95/P99).

The production models (sentence-transformers encoder, cross-encoder reranker, an LLM generator)
sit behind small interfaces; a deterministic stack (feature-hashing vectors, a hand-written BM25,
an extractive answerer) implements the same interfaces so the whole engine runs with no model
download and the tests are deterministic.
"""

from entsearch.domain import Answer, Chunk, Document, Query, Retrieved

__all__ = ["Answer", "Chunk", "Document", "Query", "Retrieved"]
