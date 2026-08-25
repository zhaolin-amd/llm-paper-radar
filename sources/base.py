from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

SourceName = Literal[
    "arxiv",
    "arxiv_authors",
    "hf_daily",
    "openreview",
]

# arXiv throttles clients with no/blank User-Agent more aggressively, and its
# API etiquette asks for a contact address. A UA carrying mailto: is treated as
# a well-behaved client. Shared by every export.arxiv.org caller.
ARXIV_USER_AGENT = "llm-paper-radar/1.0 (mailto:zhaolin@amd.com)"


class SourceRecord(BaseModel):
    name: SourceName
    fetched_at: datetime
    extras: dict = Field(default_factory=dict)


class Paper(BaseModel):
    id: str
    title: str
    authors: list[str]
    abstract: str
    url: str
    pdf_url: str | None
    published_at: datetime
    primary_category: str
    categories: list[str]
    code_url: str | None = None
    # Populated by `pipeline.enrich_github` for papers with a known `code_url`.
    # Shape: {"stars": int, "created_at": datetime, "fetched_at": datetime}.
    # Used only as a soft signal in `render.heat_score`; never gated on.
    code_meta: dict | None = None

    sources: list[SourceRecord]

    relevance_score: int | None = None
    relevance_reason: str | None = None
    relevance_breakdown: dict | None = None
    summary: str | None = None
    highlights: list[str] = Field(default_factory=list)
    # Up to 3 prior methods the paper directly compares against / builds on,
    # extracted from the abstract by the summarize step. Each entry:
    # {"name": str, "relation": str, "arxiv_id": str | None}.
    related_methods: list[dict] = Field(default_factory=list)
    # English-language siblings. Populated by summarize in the same LLM call.
    # name/arxiv_id stay language-agnostic; only `relation` differs from
    # related_methods. Absence (None / []) means the day was summarized before
    # bilingual output was added — render falls back to skipping the _en file.
    summary_en: str | None = None
    highlights_en: list[str] = Field(default_factory=list)
    related_methods_en: list[dict] = Field(default_factory=list)
    # English translation of relevance_reason (filter step writes the Chinese
    # original; summarize translates it as part of the same bilingual call so
    # we don't pay a second filter-LLM round-trip). The two free-form Chinese
    # fields inside relevance_breakdown (calibration_cost, inference_perf)
    # are translated in-place and stored as `<key>_en` siblings of that dict
    # rather than adding a parallel breakdown_en.
    relevance_reason_en: str | None = None
    seen_before: bool = False
    # Author -> institution, extracted from the paper's own PDF first page by
    # `pipeline.affiliations`. Populated in-memory at render time only, for
    # papers in the final-surfaced set — never written back to
    # data/summarized/*.json.
    author_affiliations: dict[str, str] | None = None


class Source(ABC):
    """Base class. Each concrete source implements `fetch()` returning Papers."""

    name: SourceName

    @abstractmethod
    async def fetch(self, target_date: datetime) -> list[Paper]:
        """Fetch papers for the given date (UTC)."""
        ...
