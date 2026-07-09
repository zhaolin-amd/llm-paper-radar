"""Fetch submissions from OpenReview venues (ICLR / ICML / MLSys / AAAI / ACL / EMNLP / NeurIPS / etc.).

Each configured venue invitation (e.g. `ICLR.cc/2026/Conference/-/Submission`)
is paginated newest-first; notes whose creation date falls inside the rolling
window are turned into Paper objects with `id = "or-<note_id>"` and the
forum URL as the canonical link. OpenReview submissions usually don't expose
an arXiv id during review, so we keep the OpenReview id as the primary key
and let dedupe treat them as separate entries from any arXiv duplicates.

Venue strings in config may contain a `{year}` placeholder; `_expand_year_templates`
expands each template into the current and next calendar year (so a venue's
two-year CFP window is always covered without yearly config edits). Venues
that don't exist yet — e.g. next year's conference before its CFP opens — are
silently skipped by `fetch()` when the API returns an error. Only the
`/-/Submission` invitation is queried, so reviews, official comments,
rebuttals, and meta-reviews (which live under separate invitations like
`/-/Official_Review`, `/-/Official_Comment`, `/-/Rebuttal`) are never pulled.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import httpx

from sources._openreview_auth import openreview_auth_headers
from sources.base import Paper, Source, SourceRecord


class OpenReviewSource(Source):
    name = "openreview"
    BASE_URL = "https://api2.openreview.net/notes"
    PAGE_SIZE = 1000  # API max per request

    def __init__(self, venues: list[str], window_days: int = 7, max_pages: int = 5):
        self.venues = venues
        self.window_days = window_days
        self.max_pages = max_pages

    async def fetch(self, target_date: datetime) -> list[Paper]:
        window_start = target_date - timedelta(days=self.window_days)
        window_end = target_date + timedelta(hours=1)
        out: list[Paper] = []
        expanded = _expand_year_templates(self.venues, target_date.year)
        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
            client.headers.update(await openreview_auth_headers(client))
            for venue in expanded:
                invitation = f"{venue}/-/Submission"
                try:
                    notes = await self._fetch_venue(client, invitation, window_start)
                except Exception as e:
                    print(f"openreview: skip {venue}: {type(e).__name__}: {e}")
                    continue
                for n in notes:
                    pub = _note_pub(n)
                    if pub is None or not (window_start <= pub <= window_end):
                        continue
                    paper = _note_to_paper(n, venue, pub)
                    if paper is not None:
                        out.append(paper)
                await asyncio.sleep(1.0)
        return out

    async def _fetch_venue(
        self,
        client: httpx.AsyncClient,
        invitation: str,
        window_start: datetime,
    ) -> list[dict]:
        all_notes: list[dict] = []
        offset = 0
        for _ in range(self.max_pages):
            url = (
                f"{self.BASE_URL}?invitation={invitation}"
                f"&offset={offset}&limit={self.PAGE_SIZE}"
                f"&sort=cdate:desc"
            )
            resp = None
            last_exc: Exception | None = None
            for attempt in range(3):
                try:
                    resp = await client.get(url)
                    if resp.status_code != 429:
                        last_exc = None
                        break
                    wait = 5 * (2**attempt)
                    print(f"openreview: 429 on {invitation} offset={offset}, sleep {wait}s")
                except (httpx.TimeoutException, httpx.TransportError) as e:
                    last_exc = e
                    wait = 5 * (2**attempt)
                    print(
                        f"openreview: {type(e).__name__} on {invitation} offset={offset},"
                        f" sleep {wait}s"
                    )
                await asyncio.sleep(wait)
            if last_exc is not None:
                raise last_exc
            assert resp is not None
            resp.raise_for_status()
            notes = resp.json().get("notes", [])
            if not notes:
                break
            all_notes.extend(notes)
            oldest = _note_pub(notes[-1])
            if oldest is not None and oldest < window_start:
                break
            offset += self.PAGE_SIZE
            await asyncio.sleep(1.0)
        return all_notes


def _expand_year_templates(venues: list[str], current_year: int) -> list[str]:
    """Expand `{year}` placeholders against the current year and the next year.

    A given conference's CFP window straddles two calendar years (e.g. in 2026
    OpenReview has both ICLR 2026 papers being reviewed and ICLR 2027 papers
    just opened). Templating means the config doesn't need a yearly edit.
    Venues without `{year}` are passed through unchanged. Order is preserved;
    duplicates are dropped. The fetcher silently skips venues that don't exist.
    """
    out: list[str] = []
    seen: set[str] = set()
    for tmpl in venues:
        if "{year}" in tmpl:
            candidates = [tmpl.format(year=current_year), tmpl.format(year=current_year + 1)]
        else:
            candidates = [tmpl]
        for v in candidates:
            if v not in seen:
                seen.add(v)
                out.append(v)
    return out


# Hosts that publish multiple conferences from the same root, so the actual
# conference name is the *second* path segment, not the host. Example:
# `aclweb.org/ACL/...` and `aclweb.org/EMNLP/...` both live under `aclweb.org`.
_GENERIC_HOSTS = {"aclweb.org"}


def _venue_label(venue: str) -> str:
    """Short lowercase label for a venue, used as the paper's category tag.

    Handles the three naming patterns OpenReview venues use in the wild:

    - `ICLR.cc/2026/Conference`        → `iclr`
    - `MLSys.org/2026/Conference`      → `mlsys`
    - `EMNLP/2025/Conference`          → `emnlp`     (no `.org`/`.cc`)
    - `aclweb.org/ACL/2026/Conference` → `acl`       (host is a generic org)
    """
    head = venue.split("/", 1)[0]
    if head in _GENERIC_HOSTS:
        parts = venue.split("/")
        if len(parts) >= 2 and parts[1]:
            return parts[1].lower()
    return head.split(".", 1)[0].lower()


def _content_value(content: dict, field: str, default=None):
    """OpenReview v2 wraps fields as `{"value": ...}`; older notes are bare values."""
    v = content.get(field)
    if isinstance(v, dict) and "value" in v:
        return v["value"]
    return v if v is not None else default


def _note_pub(note: dict) -> datetime | None:
    ts = note.get("cdate") or note.get("pdate") or note.get("mdate")
    if ts is None:
        return None
    return datetime.fromtimestamp(ts / 1000, tz=UTC)


def _note_to_paper(note: dict, venue: str, pub: datetime) -> Paper | None:
    content = note.get("content", {}) or {}
    title = (_content_value(content, "title", "") or "").strip()
    if not title:
        return None
    abstract = (_content_value(content, "abstract", "") or "").strip()
    authors = _content_value(content, "authors", []) or []
    if isinstance(authors, str):
        authors = [authors]
    note_id = note.get("id", "")
    if not note_id:
        return None
    venue_short = _venue_label(venue)
    return Paper(
        id=f"or-{note_id}",
        title=title,
        authors=list(authors),
        abstract=abstract,
        url=f"https://openreview.net/forum?id={note_id}",
        pdf_url=f"https://openreview.net/pdf?id={note_id}",
        published_at=pub,
        primary_category=venue_short,
        categories=[venue_short],
        sources=[
            SourceRecord(
                name="openreview",
                fetched_at=datetime.now(UTC),
                extras={"venue": venue},
            )
        ],
    )


if __name__ == "__main__":
    import json
    from pathlib import Path

    import click

    from pipeline._clock import today_utc
    from pipeline.config import load_config

    @click.command()
    @click.option("--backfill-days", default=0, type=int, help="Loop the wrapper N+1 times (default 0 = once). The actual fetch is a single windowed query — see --window-days.")
    @click.option(
        "--window-days",
        default=None,
        type=int,
        help="Single query spanning the last N days (overrides config window_days). 1 API call, output written under today's data/raw/ dir.",
    )
    @click.option("--out-dir", default="data/raw", type=click.Path(path_type=Path))
    def main(backfill_days: int, window_days: int | None, out_dir: Path):
        cfg = load_config()
        sub = cfg.sources.openreview
        if not sub.enabled:
            print("openreview source disabled")
            return
        if not sub.venues:
            print("openreview: no venues configured, skipping")
            return
        effective_window = window_days if window_days is not None else sub.window_days
        src = OpenReviewSource(venues=sub.venues, window_days=effective_window)
        today = today_utc()
        for delta in range(backfill_days + 1):
            target = today - timedelta(days=delta)
            if backfill_days > 0 and (
                Path("digests") / f"{target.strftime('%Y-%m-%d')}.md"
            ).exists():
                print(f"openreview: skip {target.date()} (digest exists)")
                continue
            papers = asyncio.run(src.fetch(target))
            day_dir = out_dir / target.strftime("%Y-%m-%d")
            day_dir.mkdir(parents=True, exist_ok=True)
            (day_dir / "openreview.json").write_text(
                json.dumps([p.model_dump(mode="json") for p in papers], indent=2)
            )
            print(f"openreview: wrote {len(papers)} papers for {target.date()}")

    main()
