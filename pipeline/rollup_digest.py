"""Periodic rollup digest for an arbitrary inclusive [start, end] window,
rendered purely from cached data/summarized/ — the same clean table as the
weekly digest, for the longer monthly / half-year / yearly cadences. Writes
`snapshots/<cadence>/<start>-<end>.md`.

Differences from `pipeline.weekly`:
  - never splices README (long cadences are archive-only);
  - refuses to render if any day inside the (clamped) window is missing its
    summarized JSON — no silent partial rollup;
  - clamps the window start up to the earliest available summarized day and
    annotates the file header when it does, so a window reaching before the
    project existed still produces a valid, honestly-labelled table.

No fetching or scoring happens here — it is a render-from-cache view, so it is
cheap and safe to re-run. Author-affiliation lookups also read from cache
(data/affiliations.json, populated by daily enrichment); an LLM/PDF call only
happens on a cache miss.

Examples:
  uv run python -m pipeline.rollup_digest \
      --start 2026-05-01 --end 2026-05-31 --label Monthly --out-dir snapshots/monthly
  uv run python -m pipeline.rollup_digest \
      --start 2025-01-01 --end 2025-12-31 --label Yearly --out-dir snapshots/yearly
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

import click

from pipeline.affiliations import enrich_affiliations
from pipeline.config import AffiliationConfig, load_config
from pipeline.weekly import _collect, _render_table


def _digest_prefix(out_dir: Path) -> str:
    """Relative path from a file in `out_dir` back to the repo-root digests/.
    `snapshots/monthly` is two levels deep → `../../digests/`. Assumes
    out_dir is repo-root-relative (the wrapper scripts cd to the project
    root before invoking)."""
    return "../" * len(out_dir.parts) + "digests/"


def _earliest_summarized(summarized_root: Path) -> datetime | None:
    earliest: datetime | None = None
    for f in summarized_root.glob("*.json"):
        try:
            d = datetime.fromisoformat(f.stem).replace(tzinfo=UTC)
        except ValueError:
            continue
        if earliest is None or d < earliest:
            earliest = d
    return earliest


def render_rollup(
    start: datetime,
    end: datetime,
    summarized_root: Path,
    out_dir: Path,
    label: str,
    digests_dir: Path | None = None,
    affiliation_cfg: AffiliationConfig | None = None,
) -> Path:
    # digests/ lives at the repo root regardless of how deep out_dir is.
    if digests_dir is None:
        digests_dir = Path("digests")

    # Clamp the window start up to the earliest day we actually have data for,
    # so a window reaching before the project existed renders the available
    # tail instead of aborting. Loudly annotated in the header.
    eff_start = start
    note: str | None = None
    earliest = _earliest_summarized(summarized_root)
    if earliest is not None and earliest > start:
        eff_start = earliest
        note = (
            f"⚠️ Window truncated to data availability: requested "
            f"{start.date()} → {end.date()}, rendered {eff_start.date()} → {end.date()}."
        )
    if eff_start > end:
        raise SystemExit(
            f"rollup aborted: requested window {start.date()} → {end.date()} ends "
            f"before the earliest available data ({earliest.date() if earliest else 'n/a'})."
        )

    # require_complete: any gap *inside* the clamped window is a real pipeline
    # outage, not a before-the-project artifact, so abort rather than mislead.
    sorted_papers, digest_date_by_id = _collect(
        summarized_root, eff_start, end, require_complete=True
    )
    if affiliation_cfg is not None:
        # Cache-only in practice: these papers already went through daily
        # enrichment, so this just reads data/affiliations.json.
        asyncio.run(enrich_affiliations(sorted_papers, affiliation_cfg))

    # Name by the effective (rendered) window so the filename never overstates
    # coverage. Once history fills in, eff_start == start and it reads as the
    # natural calendar period.
    fname = f"{eff_start.strftime('%Y%m%d')}-{end.strftime('%Y%m%d')}.md"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / fname
    out_path.write_text(
        _render_table(
            sorted_papers,
            digest_date_by_id,
            eff_start,
            end,
            _digest_prefix(out_dir),
            digests_dir,
            label=label,
            note=note,
        )
    )
    return out_path


@click.command()
@click.option("--start", required=True, help="Inclusive window start (YYYY-MM-DD).")
@click.option("--end", required=True, help="Inclusive window end (YYYY-MM-DD).")
@click.option("--label", required=True, help='Digest heading, e.g. "Monthly".')
@click.option("--out-dir", required=True, type=click.Path(path_type=Path))
@click.option("--in-root", default="data/summarized", type=click.Path(path_type=Path))
def main(start: str, end: str, label: str, out_dir: Path, in_root: Path) -> None:
    cfg = load_config()
    s = datetime.fromisoformat(start).replace(tzinfo=UTC)
    e = datetime.fromisoformat(end).replace(tzinfo=UTC)
    if s > e:
        raise SystemExit(f"start {s.date()} is after end {e.date()}")
    out_path = render_rollup(s, e, in_root, out_dir, label, affiliation_cfg=cfg.affiliation)
    print(f"{label}: digest written for {s.date()} → {e.date()} -> {out_path}")


if __name__ == "__main__":
    main()
