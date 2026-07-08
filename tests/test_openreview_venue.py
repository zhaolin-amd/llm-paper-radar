from pathlib import Path

import pytest
import respx
from httpx import Response

from sources.openreview_venue import (
    VenueFetchIncomplete,
    _is_accepted,
    fetch_venue_accepted,
)


def _note(note_id: str, title: str, venueid: str) -> dict:
    return {
        "id": note_id,
        "cdate": 1750000000000,
        "content": {
            "title": {"value": title},
            "abstract": {"value": f"Abstract for {title}."},
            "authors": {"value": ["Alice"]},
            "venueid": {"value": venueid},
        },
    }


def test_is_accepted_matches_exact_venue_string():
    venue = "MLSys.org/2026/Conference"
    assert _is_accepted(_note("1", "X", venue), venue) is True
    assert _is_accepted(_note("1", "X", f"{venue}/Rejected_Submission"), venue) is False
    assert _is_accepted(_note("1", "X", f"{venue}/-/Submission"), venue) is False


@respx.mock
@pytest.mark.asyncio
async def test_fetch_venue_accepted_keeps_only_accepted_notes():
    venue = "TEST.cc/2026/Conference"
    invitation = f"{venue}/-/Submission"
    notes = [
        _note("a1", "Accepted paper", venue),
        _note("r1", "Rejected paper", f"{venue}/Rejected_Submission"),
    ]
    respx.get(
        f"https://api2.openreview.net/notes?invitation={invitation}"
        f"&offset=0&limit=1000&sort=cdate:desc"
    ).mock(return_value=Response(200, json={"notes": notes}))

    papers = await fetch_venue_accepted(venue)
    assert [p.title for p in papers] == ["Accepted paper"]


@respx.mock
@pytest.mark.asyncio
async def test_fetch_venue_accepted_retries_403_then_succeeds():
    venue = "TEST.cc/2026/Conference"
    invitation = f"{venue}/-/Submission"
    route = respx.get(
        f"https://api2.openreview.net/notes?invitation={invitation}"
        f"&offset=0&limit=1000&sort=cdate:desc"
    )
    route.side_effect = [
        Response(403, json={"message": "Challenge verification required"}),
        Response(200, json={"notes": [_note("a1", "Accepted paper", venue)]}),
    ]

    papers = await fetch_venue_accepted(venue)
    assert [p.title for p in papers] == ["Accepted paper"]
    assert route.call_count == 2


@respx.mock
@pytest.mark.asyncio
async def test_fetch_venue_accepted_raises_after_retries_exhausted():
    venue = "TEST.cc/2026/Conference"
    invitation = f"{venue}/-/Submission"
    respx.get(
        f"https://api2.openreview.net/notes?invitation={invitation}"
        f"&offset=0&limit=1000&sort=cdate:desc"
    ).mock(return_value=Response(403, json={"message": "Challenge verification required"}))

    with pytest.raises(VenueFetchIncomplete):
        await fetch_venue_accepted(venue)


@respx.mock
@pytest.mark.asyncio
async def test_fetch_venue_accepted_resumes_from_page_cache(tmp_path: Path):
    """A cached page-0000.json is reused instead of re-fetched; only the
    (not-yet-cached) second page hits the network."""
    venue = "TEST.cc/2026/Conference"
    invitation = f"{venue}/-/Submission"
    cache_dir = tmp_path / "pages"
    cache_dir.mkdir()
    cached_page = [_note(f"cached-{i}", f"Cached {i}", venue) for i in range(1000)]
    (cache_dir / "page-0000.json").write_text(__import__("json").dumps(cached_page))

    route = respx.get(
        f"https://api2.openreview.net/notes?invitation={invitation}"
        f"&offset=1000&limit=1000&sort=cdate:desc"
    ).mock(return_value=Response(200, json={"notes": [_note("a2", "Second page paper", venue)]}))

    papers = await fetch_venue_accepted(venue, page_cache_dir=cache_dir)
    titles = {p.title for p in papers}
    assert "Cached 0" in titles
    assert "Second page paper" in titles
    assert route.call_count == 1
