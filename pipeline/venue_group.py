"""Pure grouping stage: partitions a venue's scored papers by subfield.
Hard-gated papers are dropped — they never reach the trend-analysis step."""

from __future__ import annotations

import json
from pathlib import Path


def group_by_subfield(scored_path: Path) -> dict[str, list[dict]]:
    papers = json.loads(Path(scored_path).read_text())
    groups: dict[str, list[dict]] = {}
    for p in papers:
        breakdown = p.get("relevance_breakdown") or {}
        if breakdown.get("hard_gate"):
            continue
        subfield = breakdown.get("subfield") or "unknown"
        groups.setdefault(subfield, []).append(p)
    return groups


if __name__ == "__main__":
    import click

    @click.command()
    @click.option("--scored-path", required=True, type=click.Path(path_type=Path))
    @click.option("--out-path", required=True, type=click.Path(path_type=Path))
    def main(scored_path: Path, out_path: Path):
        groups = group_by_subfield(scored_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(groups, indent=2))
        summary = ", ".join(f"{k}={len(v)}" for k, v in sorted(groups.items(), key=lambda kv: -len(kv[1])))
        print(f"venue_group: {summary} -> {out_path}")

    main()
