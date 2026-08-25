"""7-day rollup digest as a single compact table: emits
snapshots/weekly/YYYYMMDD-YYYYMMDD.md and splices the same table into README.md
(between the WEEKLY markers).

The window-loading (`_collect`) and table-rendering (`_render_table`) helpers are
reused by `pipeline.rollup_digest` for the longer monthly / half-year / yearly
cadences, so keep them cadence-agnostic."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from pipeline._clock import today_utc
from pipeline.affiliations import enrich_affiliations
from pipeline.config import AffiliationConfig, load_config
from pipeline.render import _bucket_sort_key, _compact_row, _splice_weekly_into_readme
from sources.base import Paper


def _passed_gate(p: Paper) -> bool:
    if p.relevance_score is None:
        return False
    bd = p.relevance_breakdown or {}
    return not bd.get("hard_gate", False)


def _collect(
    summarized_root: Path,
    start_date: datetime,
    end_date: datetime,
    *,
    require_complete: bool = False,
) -> tuple[list[Paper], dict[str, str]]:
    """Load summarized papers for the inclusive window [start_date, end_date],
    keeping each paper's best-scoring appearance and the digest date it came
    from. Returns (sorted_papers, digest_date_by_id).

    With `require_complete=True`, raises SystemExit listing every day in the
    window that is missing its `summarized/YYYY-MM-DD.json` — used by the
    monthly/half-year/yearly rollups so a pipeline gap never silently produces
    a partial table. Weekly leaves it False (tolerates the occasional gap)."""
    by_id: dict[str, tuple[Paper, str]] = {}
    missing: list[str] = []
    cursor = start_date
    while cursor <= end_date:
        date_str = cursor.strftime("%Y-%m-%d")
        f = summarized_root / f"{date_str}.json"
        if not f.exists():
            missing.append(date_str)
            cursor += timedelta(days=1)
            continue
        for item in json.loads(f.read_text()):
            p = Paper.model_validate(item)
            prev = by_id.get(p.id)
            if not prev or (p.relevance_score or 0) > (prev[0].relevance_score or 0):
                by_id[p.id] = (p, date_str)
        cursor += timedelta(days=1)

    if require_complete and missing:
        raise SystemExit(
            f"rollup aborted: {len(missing)} day(s) in "
            f"{start_date.date()} → {end_date.date()} missing summarized JSON: "
            f"{', '.join(missing)}"
        )

    surviving = [(p, d) for p, d in by_id.values() if _passed_gate(p)]
    digest_date_by_id = {p.id: d for p, d in surviving}
    sorted_papers = sorted((p for p, _ in surviving), key=_bucket_sort_key)
    return sorted_papers, digest_date_by_id


def _render_table(
    sorted_papers: list[Paper],
    digest_date_by_id: dict[str, str],
    start_date: datetime,
    end_date: datetime,
    digest_prefix: str,
    digests_dir: Path,
    label: str = "Weekly",
    note: str | None = None,
) -> str:
    """Build the compact digest table. `digest_prefix` is the path to the
    digests/ dir relative to wherever the table will live (`../digests/` for
    the weekly/ file, `digests/` for the repo-root README, `../../digests/` for
    a rollups/<cadence>/ file). `digests_dir` is the on-disk path used to check
    for an `_en` sibling, so the Why-column can render the symmetric
    `[zh] · [en]` link like the daily and INDEX tables. `label` sets the digest
    heading ("Weekly" / "Monthly" / "Half-Year" / "Yearly"); `note`, if given,
    is emitted as a blockquote under the count (used to flag a truncated
    window)."""
    body: list[str] = []
    body.append(f"# {label} Digest · {start_date.date()} → {end_date.date()}\n")
    body.append(f"> Surfaced: {len(sorted_papers)} papers\n")
    if note:
        body.append(f"> {note}\n")
    body.append("| # | Bucket | Paper | Authors | Date | Why |")
    body.append("|---|--------|-------|---------|------|-----|")
    for i, p in enumerate(sorted_papers, start=1):
        date_str = digest_date_by_id[p.id]
        digest_link = f"{digest_prefix}{date_str}.md"
        en_link = (
            f"{digest_prefix}{date_str}_en.md"
            if (digests_dir / f"{date_str}_en.md").exists()
            else None
        )
        body.append(_compact_row(i, p, digest_link, en_link))
    return "\n".join(body) + "\n"


def render_weekly(
    end_date: datetime,
    summarized_root: Path,
    out_dir: Path,
    readme_path: Path | None = None,
    digests_dir: Path | None = None,
    affiliation_cfg: AffiliationConfig | None = None,
) -> None:
    # digests/ is always at the repo root regardless of out_dir depth.
    if digests_dir is None:
        digests_dir = Path("digests")

    start_date = end_date - timedelta(days=6)
    sorted_papers, digest_date_by_id = _collect(summarized_root, start_date, end_date)
    if affiliation_cfg is not None:
        # Cache-only in practice: each of these papers already went through
        # daily enrichment, so this just reads data/affiliations.json — no new
        # PDF/LLM calls unless a paper predates the feature.
        asyncio.run(enrich_affiliations(sorted_papers, affiliation_cfg))

    fname = f"{start_date.strftime('%Y%m%d')}-{end_date.strftime('%Y%m%d')}.md"

    # Link prefix: depth of out_dir levels back to repo root, then digests/.
    digest_prefix = "../" * len(out_dir.parts) + "digests/"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / fname).write_text(
        _render_table(
            sorted_papers,
            digest_date_by_id,
            start_date,
            end_date,
            digest_prefix,
            digests_dir,
        )
    )

    # README at repo root → digests/<date>.md
    if readme_path is not None:
        _splice_weekly_into_readme(
            readme_path,
            _render_table(
                sorted_papers,
                digest_date_by_id,
                start_date,
                end_date,
                "digests/",
                digests_dir,
            ),
        )


if __name__ == "__main__":
    import click

    @click.command()
    @click.option("--end-date", default=None)
    @click.option("--in-root", default="data/summarized", type=click.Path(path_type=Path))
    @click.option("--out-dir", default="snapshots/weekly", type=click.Path(path_type=Path))
    @click.option("--readme", default="README.md", type=click.Path(path_type=Path))
    def main(end_date, in_root, out_dir, readme):
        cfg = load_config()
        end = (
            datetime.fromisoformat(end_date).replace(tzinfo=UTC)
            if end_date
            else today_utc()
        )
        render_weekly(end, in_root, out_dir, readme_path=readme, affiliation_cfg=cfg.affiliation)
        start = end - timedelta(days=6)
        print(f"weekly: digest written for {start.date()} → {end.date()}")

    main()
