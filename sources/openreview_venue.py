"""One-shot fetch of a conference's full accepted-paper set from OpenReview.

Unlike `sources.openreview.OpenReviewSource` (which pulls a rolling window
of recently-created submissions for the daily/weekly incremental pipeline),
this fetches *every* submission for a venue, keeps only the ones whose
decision routed them to the venue itself, and returns them as `Paper`
objects. Meant for one-off conference batch analysis (e.g.
`scripts/venue_report.sh`), not the daily cron.

Decision detection follows the common OpenReview v2 convention: once
decisions are released, an accepted submission's `content.venueid.value` is
rewritten from the `/-/Submission` invitation venue to the venue string
itself (e.g. "MLSys.org/2026/Conference"); rejected/withdrawn submissions
get a `.../Rejected_Submission` or `.../Withdrawn_Submission` suffix
instead. This has not been verified against MLSys 2026 specifically — the
OpenReview API returned intermittent 403s throughout development (see
docs/superpowers/specs/2026-07-08-mlsys-venue-trend-report-design.md).
Run this module's CLI once the API is reachable, inspect a sample of
`content.venueid.value` values in the raw page cache, and adjust
`_is_accepted` if MLSys's actual field differs.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx

from sources._openreview_auth import openreview_auth_headers
from sources.base import Paper
from sources.openreview import _content_value, _note_pub, _note_to_paper

BASE_URL = "https://api2.openreview.net/notes"
PAGE_SIZE = 1000
RETRYABLE_STATUSES = {403, 429}


class VenueFetchIncomplete(RuntimeError):
    """Raised when a page could not be fetched after retries. Callers must
    not treat a partial result as final — see spec Section 3 (Error Handling)."""


def _is_accepted(note: dict, venue: str) -> bool:
    venueid = _content_value(note.get("content", {}) or {}, "venueid", "")
    return venueid == venue


async def _fetch_page(
    client: httpx.AsyncClient,
    invitation: str,
    offset: int,
    max_attempts: int = 6,
) -> list[dict]:
    url = (
        f"{BASE_URL}?invitation={invitation}"
        f"&offset={offset}&limit={PAGE_SIZE}&sort=cdate:desc"
    )
    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            resp = await client.get(url)
            if resp.status_code in RETRYABLE_STATUSES:
                wait = 5 * (2**attempt)
                print(
                    f"openreview_venue: {resp.status_code} at offset={offset}, "
                    f"attempt {attempt + 1}/{max_attempts}, sleep {wait}s"
                )
                await asyncio.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json().get("notes", [])
        except (httpx.TimeoutException, httpx.TransportError) as e:
            last_exc = e
            wait = 5 * (2**attempt)
            print(
                f"openreview_venue: {type(e).__name__} at offset={offset}, sleep {wait}s"
            )
            await asyncio.sleep(wait)
    raise VenueFetchIncomplete(
        f"gave up on offset={offset} after {max_attempts} attempts: {last_exc}"
    )


async def fetch_venue_accepted(
    venue: str,
    max_pages: int = 20,
    page_cache_dir: Path | None = None,
) -> list[Paper]:
    """Fetch every accepted paper for `venue`, e.g. "MLSys.org/2026/Conference".

    Paginates `/-/Submission` notes to completion (no time window), keeps
    only notes whose decision routed them to `venue` itself, and returns
    them as `Paper` objects. Raises `VenueFetchIncomplete` if any page fails
    after retries. If `page_cache_dir` is given, each raw page is persisted
    there and reused on a subsequent call, so a re-run after a mid-run
    failure resumes instead of restarting from offset 0.

    Caution: the cache never expires. It's meant to survive a same-day
    retry after a 403/timeout, not to be reused across runs taken before
    and after decisions are released — a later run pointed at the same
    `page_cache_dir` will serve stale pre-decision pages and miss papers
    whose `venueid` only just flipped to accepted. Delete `page_cache_dir`
    before re-running against a venue whose decisions may have changed.
    """
    invitation = f"{venue}/-/Submission"
    accepted: list[Paper] = []
    offset = 0
    async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
        client.headers.update(await openreview_auth_headers(client))
        for page_num in range(max_pages):
            cache_file = (
                page_cache_dir / f"page-{page_num:04d}.json" if page_cache_dir else None
            )
            if cache_file is not None and cache_file.exists():
                notes = json.loads(cache_file.read_text())
            else:
                notes = await _fetch_page(client, invitation, offset)
                if cache_file is not None:
                    cache_file.parent.mkdir(parents=True, exist_ok=True)
                    cache_file.write_text(json.dumps(notes))
            if not notes:
                break
            for note in notes:
                if not _is_accepted(note, venue):
                    continue
                pub = _note_pub(note)
                if pub is None:
                    continue
                paper = _note_to_paper(note, venue, pub)
                if paper is not None:
                    accepted.append(paper)
            if len(notes) < PAGE_SIZE:
                break
            offset += PAGE_SIZE
            await asyncio.sleep(1.0)
    return accepted


if __name__ == "__main__":
    import click

    MIN_ACCEPTED_PAPERS = 20

    def _slug(venue: str) -> str:
        conf = venue.split(".", 1)[0].split("/", 1)[0].lower()
        year = venue.split("/")[1]
        return f"{conf}-{year}"

    @click.command()
    @click.option("--venue", required=True, help='e.g. "MLSys.org/2026/Conference"')
    @click.option("--out-dir", default="data/raw", type=click.Path(path_type=Path))
    @click.option("--min-papers", default=MIN_ACCEPTED_PAPERS, type=int)
    def main(venue: str, out_dir: Path, min_papers: int):
        venue_dir = out_dir / _slug(venue)
        page_cache_dir = venue_dir / "openreview_pages"
        try:
            papers = asyncio.run(fetch_venue_accepted(venue, page_cache_dir=page_cache_dir))
        except VenueFetchIncomplete as e:
            raise SystemExit(f"venue fetch incomplete, re-run later: {e}")
        if len(papers) < min_papers:
            raise SystemExit(
                f"only {len(papers)} accepted papers found for {venue} "
                f"(expected >= {min_papers}) — treating as incomplete, not writing output"
            )
        venue_dir.mkdir(parents=True, exist_ok=True)
        out_path = venue_dir / "accepted.json"
        out_path.write_text(json.dumps([p.model_dump(mode="json") for p in papers], indent=2))
        print(f"openreview_venue: wrote {len(papers)} accepted papers to {out_path}")

    main()
