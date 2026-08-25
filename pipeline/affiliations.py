"""Author-institution extraction from a paper's own PDF first page.

Third-party indexers (OpenAlex, Semantic Scholar) only have affiliation
data once a paper is matched to a formally published venue record —
useless for same-day/week preprints. The only day-of source is the PDF
itself (authors self-report affiliation in the title-page author block).
This only runs against the final-surfaced set (watched + passed
hard_gate), never the full scored/filtered pool, to bound PDF-fetch + LLM
cost. Results are cached indefinitely in `data/affiliations.json` keyed by
arxiv id — a paper's title-page affiliations don't change after
publication.
"""

from __future__ import annotations

import asyncio
import io
import json
from pathlib import Path

import httpx
from pypdf import PdfReader

from pipeline.config import AffiliationConfig
from pipeline.llm_client import LLMClient
from sources.base import ARXIV_USER_AGENT, Paper

CACHE_PATH = Path("data/affiliations.json")

_SYSTEM_PROMPT = """You are given an arXiv paper's known author list and the raw \
extracted text of its PDF first page. Identify each author's institution/\
affiliation as stated on the page (author block, footnotes, or \
correspondence line).

Return JSON only: {"affiliations": {"<author name exactly as given>": \
"<short institution name>", ...}}. Only include authors you can confidently \
map; omit the rest. Keep institution names short (e.g. "MIT", "Google \
DeepMind", "Tsinghua University"), not full mailing addresses."""


def _load_cache() -> dict[str, dict[str, str]]:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text())
    return {}


def _save_cache(cache: dict[str, dict[str, str]]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=2, ensure_ascii=False, sort_keys=True))


async def _fetch_first_page_text(pdf_url: str, http: httpx.AsyncClient) -> str | None:
    try:
        resp = await http.get(pdf_url)
        resp.raise_for_status()
        reader = PdfReader(io.BytesIO(resp.content))
        if not reader.pages:
            return None
        return reader.pages[0].extract_text() or None
    except Exception:
        return None


async def _extract_one(
    paper: Paper, http: httpx.AsyncClient, llm: LLMClient, sem: asyncio.Semaphore
) -> dict[str, str]:
    if not paper.pdf_url:
        return {}
    async with sem:
        text = await _fetch_first_page_text(paper.pdf_url, http)
        if not text:
            return {}
        user = (
            f"Authors: {', '.join(paper.authors)}\n\n"
            f"First-page text:\n{text[:6000]}"
        )
        try:
            r = await llm.call_json(_SYSTEM_PROMPT, user, max_tokens=500)
            aff = r.get("affiliations") or {}
            return {k: v for k, v in aff.items() if isinstance(k, str) and isinstance(v, str)}
        except Exception as e:
            print(f"affiliations: {paper.id} failed: {e}")
            return {}


async def enrich_affiliations(
    papers: list[Paper], cfg: AffiliationConfig
) -> dict[str, dict[str, str]]:
    """Populate `paper.author_affiliations` in place for every paper passed
    in, using an on-disk cache keyed by arxiv id. Only fetches for cache
    misses. Returns the (possibly updated) cache."""
    cache = _load_cache()
    if cfg.enabled:
        targets = [p for p in papers if p.id not in cache]
        if targets:
            sem = asyncio.Semaphore(cfg.concurrency)
            llm = LLMClient(model=cfg.model)
            async with httpx.AsyncClient(
                timeout=30.0,
                follow_redirects=True,
                headers={"User-Agent": ARXIV_USER_AGENT},
            ) as http:
                results = await asyncio.gather(
                    *[_extract_one(p, http, llm, sem) for p in targets]
                )
            for p, r in zip(targets, results, strict=True):
                cache[p.id] = r
            _save_cache(cache)
    for p in papers:
        p.author_affiliations = cache.get(p.id) or None
    return cache
