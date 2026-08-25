"""Pydantic config model loaded from config.yaml; consumed by every pipeline stage."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class ArxivConfig(BaseModel):
    enabled: bool = True
    categories: list[str] = ["cs.CL", "cs.LG", "cs.AR"]


class SimpleSourceConfig(BaseModel):
    enabled: bool = True


class WatchedAuthor(BaseModel):
    name: str
    affiliation: str = ""


class ArxivAuthorsConfig(BaseModel):
    enabled: bool = True
    window_days: int = 7
    # Restrict author search to these arXiv categories. Defaults cover the
    # compression/quant/diffusion space; widen if you want everything.
    categories: list[str] = ["cs.LG", "cs.CL", "cs.CV", "cs.AR", "stat.ML"]
    authors: list[WatchedAuthor] = []


class OpenReviewConfig(BaseModel):
    enabled: bool = True
    window_days: int = 7
    # Venue invitation prefixes — the fetcher appends `/-/Submission`. Strings
    # may contain a `{year}` placeholder, auto-expanded to the current and next
    # calendar year. Active set (kept in sync with config.yaml):
    #   ICLR.cc/{year}/Conference
    #   ICML.cc/{year}/Conference
    #   NeurIPS.cc/{year}/Conference
    #   MLSys.org/{year}/Conference
    #   AAAI.org/{year}/Conference
    #   aclweb.org/ACL/{year}/Conference
    #   EMNLP/{year}/Conference
    venues: list[str] = []


class SourcesConfig(BaseModel):
    arxiv: ArxivConfig = ArxivConfig()
    hf_daily: SimpleSourceConfig = SimpleSourceConfig()
    arxiv_authors: ArxivAuthorsConfig = ArxivAuthorsConfig()
    openreview: OpenReviewConfig = OpenReviewConfig()


class KeywordRule(BaseModel):
    pattern: str
    weight: int


class PrefilterConfig(BaseModel):
    """Cheap keyword prefilter that runs before the LLM judge. Papers with
    no whitelist hits AND >= `max_blacklist_hits` blacklist matches are
    hard-gated locally without burning a Sonnet call.

    Word-boundary matching is applied — `QuIP` will not match inside
    `equipping`, `MIT` will not match inside `Amit`."""

    enabled: bool = True
    whitelist: list[KeywordRule] = Field(default_factory=list)
    blacklist: list[KeywordRule] = Field(default_factory=list)
    max_blacklist_hits: int = 2


class FilterConfig(BaseModel):
    model: str = "claude-sonnet-4-6"
    concurrency: int = 50
    prefilter: PrefilterConfig = PrefilterConfig()


class SummarizeConfig(BaseModel):
    model: str = "claude-opus-4-7"
    concurrency: int = 20


class AffiliationConfig(BaseModel):
    """Author-institution extraction from a paper's own PDF first page.
    Only runs against the final-surfaced set (watched + passed hard_gate),
    never the full scored/filtered pool — see pipeline/affiliations.py."""

    enabled: bool = True
    model: str = "claude-sonnet-4-6"
    concurrency: int = 3


class RenderConfig(BaseModel):
    truncate_after: int = 10
    # New bucket enum: [ptq, qat, low_bits, kv_cache, pruning_distill, diffusion, survey]
    topic_caps: dict[str, int] = Field(
        default_factory=lambda: {
            "ptq": 8,
            "low_bits": 5,
            "qat": 5,
            "kv_cache": 5,
            "pruning_distill": 3,
            "diffusion": 3,
            "survey": 3,
            "_default": 2,
        }
    )


class DedupeConfig(BaseModel):
    cross_day_strategy: Literal["strict", "lenient"] = "lenient"
    source_priority: list[str] = []


class Config(BaseModel):
    sources: SourcesConfig = SourcesConfig()
    filter: FilterConfig = FilterConfig()
    summarize: SummarizeConfig = SummarizeConfig()
    affiliation: AffiliationConfig = AffiliationConfig()
    render: RenderConfig = RenderConfig()
    dedupe: DedupeConfig = DedupeConfig()


def load_config(path: Path = Path("config.yaml")) -> Config:
    data = yaml.safe_load(Path(path).read_text())
    return Config(**data)
