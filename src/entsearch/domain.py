"""Shared types. Pure data — no retrieval logic, no model calls."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Document(BaseModel):
    id: str
    text: str
    title: str = ""
    source: str = ""
    # ACLs/space/owner would live here in a real deployment; the filter is applied before ranking.
    metadata: dict[str, str] = Field(default_factory=dict)


class Chunk(BaseModel):
    id: str  # "<doc_id>:<index>"
    doc_id: str
    index: int
    text: str
    start: int  # char offset in the source document
    end: int
    title: str = ""  # the document's title: context for every sentence ("Legacy VPN: Its ...")

    @staticmethod
    def make(doc_id: str, index: int, text: str, start: int, end: int, title: str = "") -> Chunk:
        return Chunk(
            id=f"{doc_id}:{index}",
            doc_id=doc_id,
            index=index,
            text=text,
            start=start,
            end=end,
            title=title,
        )

    @property
    def indexed_text(self) -> str:
        """What the indexes and rerankers see: the title, then the passage. A chunk cut from the
        middle of a document often never names its subject; the title does."""
        return f"{self.title}\n{self.text}" if self.title else self.text


class Query(BaseModel):
    id: str
    text: str


class Scored(BaseModel):
    """A chunk id with a score from one retriever — the internal currency of retrieval/fusion."""

    chunk_id: str
    score: float


class Retrieved(BaseModel):
    chunk: Chunk
    score: float
    source: str = ""  # "dense" | "bm25" | "hybrid" | "rerank"


class Citation(BaseModel):
    chunk_id: str
    quote: str = ""


class Answer(BaseModel):
    query_id: str
    text: str  # what the user gets: only the claims the gate verified (or the abstention)
    draft: str = ""  # what the generator wrote, before the gate
    citations: list[Citation] = Field(default_factory=list)
    supported: bool = True  # no unsupported claim (an abstention makes no claim)
    abstained: bool = False  # the engine or the generator declined to answer
    refusal_reason: str = ""
    retrieved: list[Retrieved] = Field(default_factory=list)
    timings_ms: dict[str, float] = Field(default_factory=dict)


class QueryCase(BaseModel):
    """One evaluation case: a query, the chunks that should be retrieved, and reference spans."""

    query: Query
    relevant_chunk_ids: set[str] = Field(default_factory=set)
    # Benchmarks judge documents, not chunks (BEIR qrels): when set, retrieval is scored by ranking
    # documents by their best chunk, and `relevant_chunk_ids` is ignored.
    relevant_doc_ids: set[str] = Field(default_factory=set)
    grades: dict[str, int] = Field(default_factory=dict)  # graded qrels (BEIR): id -> grade
    answer_spans: list[str] = Field(default_factory=list)
    answerable: bool = True  # False: the corpus does not cover it; the right answer is to abstain
