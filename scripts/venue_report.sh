#!/usr/bin/env bash
# Fetch, score, and group a conference's accepted papers for the venue
# trend report. Does NOT run the trend-analysis Workflow — that step needs
# the Workflow tool and is run separately (see docs/superpowers/plans/
# 2026-07-08-mlsys-venue-trend-report.md, Task 5).
set -euo pipefail

VENUE="${1:?usage: venue_report.sh <venue> e.g. MLSys.org/2026/Conference}"
CONF_SLUG="$(echo "$VENUE" | cut -d'.' -f1 | cut -d'/' -f1 | tr '[:upper:]' '[:lower:]')-$(echo "$VENUE" | cut -d'/' -f2)"

echo "[1/3] fetching accepted papers for $VENUE"
uv run python -m sources.openreview_venue --venue "$VENUE"

echo "[2/3] scoring papers for LLM inference deployment relevance"
uv run python -m pipeline.venue_filter \
  --in-path "data/raw/${CONF_SLUG}/accepted.json" \
  --out-path "data/scored/${CONF_SLUG}.json"

echo "[3/3] grouping by subfield"
uv run python -m pipeline.venue_group \
  --scored-path "data/scored/${CONF_SLUG}.json" \
  --out-path "data/scored/${CONF_SLUG}-grouped.json"

echo "done: data/scored/${CONF_SLUG}-grouped.json"
