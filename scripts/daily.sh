#!/usr/bin/env bash
# Daily pipeline: fetch -> dedupe -> filter -> summarize -> render -> commit + push.
# Designed for cron. Sources ~/.bashrc so the AMD Anthropic proxy env vars are present.
#
# Usage:
#   daily.sh                          # cron default: yesterday-UTC, force re-fetch
#   daily.sh --date 2026-05-20        # fetch arxiv publication date 2026-05-20 (manual rerun)
#   daily.sh --days 60                # backfill last 60 days from today (skip existing digests)
#   daily.sh --days 60 --force        # backfill last 60 days, re-fetch even if digest exists
#   daily.sh --no-fetch               # skip the source fetch step; re-run dedupe→render
#                                       against whatever is already in data/raw/
#   DAYS=14 daily.sh                  # env var also works
#
# Date semantics: the "target date" everywhere (folder names, digest
# filenames, README title, paper Date column) is the **arxiv publication
# date** of the batch being processed. Cron at Beijing 14:00 (= UTC 06:00)
# targets yesterday-UTC via RADAR_DAY_OFFSET=1, by which hour arxiv has
# finished publishing the previous UTC day's batch. Manual --date pins
# RADAR_TARGET_DATE so all components agree on the same publication day.

set -uo pipefail

# Wall-clock start; reported on EXIT so even early `exit N` paths log total.
DAILY_START=$SECONDS
_fmt_duration() {
    local s=$1
    if (( s < 60 )); then
        echo "${s}s"
    else
        printf '%dm%02ds\n' $((s / 60)) $((s % 60))
    fi
}
trap 'echo "[$(date -Is)] daily.sh total: $(_fmt_duration $((SECONDS - DAILY_START)))"' EXIT

DAYS="${DAYS:-1}"
DATE=""

# Shift every "today" the pipeline computes back by one UTC day. The
# cron fires at UTC 06:00 — UTC today is still mostly empty on arxiv,
# while UTC yesterday is the freshly-complete batch. See pipeline/_clock.py.
# Overridden below if --date is passed (RADAR_TARGET_DATE wins).
export RADAR_DAY_OFFSET=1
NO_FETCH=0
FORCE=1   # default on: the daily cron's whole point is to re-render yesterday
DAYS_EXPLICIT=0
FORCE_EXPLICIT=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --date) DATE="$2"; shift 2 ;;
        --date=*) DATE="${1#*=}"; shift ;;
        --days) DAYS="$2"; DAYS_EXPLICIT=1; shift 2 ;;
        --days=*) DAYS="${1#*=}"; DAYS_EXPLICIT=1; shift ;;
        --no-fetch) NO_FETCH=1; shift ;;
        --force) FORCE=1; FORCE_EXPLICIT=1; shift ;;
        --no-force) FORCE=0; FORCE_EXPLICIT=1; shift ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

# --date pins the publication date all components target. Mutually
# exclusive with the offset-based default; clearing OFFSET avoids the
# silent compose where "yesterday" would override the explicit pin.
if [[ -n "$DATE" ]]; then
    if ! [[ "$DATE" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
        echo "--date must be YYYY-MM-DD, got: $DATE" >&2
        exit 1
    fi
    export RADAR_TARGET_DATE="$DATE"
    unset RADAR_DAY_OFFSET
fi

# If the caller passed --days (backfill mode) without an explicit --force/--no-force,
# fall back to the historical behavior of skipping days whose digest already exists.
if [[ "$DAYS_EXPLICIT" -eq 1 && "$FORCE_EXPLICIT" -eq 0 ]]; then
    FORCE=0
fi

if ! [[ "$DAYS" =~ ^[0-9]+$ ]] || [[ "$DAYS" -lt 1 ]]; then
    echo "--days must be a positive integer, got: $DAYS" >&2
    exit 1
fi

BACKFILL=$((DAYS - 1))

PROJECT_ROOT="/proj/xcohdstaff7/zhaolin/code/llm-paper-radar"
LOG_DIR="${PROJECT_ROOT}/scripts/log"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/$(date +%Y-%m-%d).log"

# shellcheck disable=SC1091
source "${PROJECT_ROOT}/scripts/lib/alert.sh"

exec >>"$LOG_FILE" 2>&1
echo "================================================================"
echo "[$(date -Is)] daily.sh start (window=${DAYS}d, backfill=${BACKFILL})"

# Pull in user env (ANTHROPIC_BASE_URL / ANTHROPIC_CUSTOM_HEADERS / PATH / uv).
# Disable -u while sourcing: ~/.bashrc references vars like LS_COLORS that are
# unset in cron's environment, which would otherwise kill the script here.
# shellcheck disable=SC1090
set +u
source "$HOME/.bashrc" || true
set -u

if [[ -z "${ANTHROPIC_CUSTOM_HEADERS:-}" ]]; then
    echo "[$(date -Is)] ERROR: ANTHROPIC_CUSTOM_HEADERS not set after sourcing ~/.bashrc"
    alert_failure "daily.sh" "ANTHROPIC_CUSTOM_HEADERS not set after sourcing ~/.bashrc" "$LOG_FILE"
    exit 2
fi

cd "$PROJECT_ROOT"

# Make git play nice in cron — non-interactive SSH, predictable identity.
export GIT_SSH_COMMAND="ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new"

# Pull latest so we don't push on a stale base.
echo "[$(date -Is)] git pull --rebase"
git pull --rebase --autostash || { echo "git pull failed, aborting"; alert_failure "daily.sh" "git pull --rebase failed" "$LOG_FILE"; exit 3; }

run_step() {
    local name="$1"; shift
    local start=$SECONDS
    echo "[$(date -Is)] step: $name"
    if ! "$@"; then
        local rc=$?
        echo "[$(date -Is)] step '$name' failed (exit $rc) after $((SECONDS - start))s"
        return 1
    fi
    echo "[$(date -Is)] step '$name' done in $((SECONDS - start))s"
}

# Fetch all sources over the requested window. Per-day sources loop via
# --backfill-days; windowed sources (arxiv_authors, openreview) take
# --window-days and fetch once. Source-level failures print and continue.
# --no-fetch skips this section entirely so we can re-run dedupe→render
# against whatever raw data is already on disk (no external API calls).
if [[ "$NO_FETCH" -eq 1 ]]; then
    echo "[$(date -Is)] fetch: skipped (--no-fetch)"
else
    FORCE_FLAG=()
    FORCE_DESC=""
    if [[ "$FORCE" -eq 1 ]]; then
        FORCE_FLAG=(--force)
        FORCE_DESC=" --force"
    fi
    # Inter-source pause: hf_daily/arxiv/arxiv_authors all hit
    # export.arxiv.org from the same IP within seconds, which reliably
    # triggers arxiv's per-IP 429 throttle (observed 2026-05-26: lost
    # ~half the trending IDs + skipped both arxiv day-fetches). 45s gives
    # arxiv's counter time to cool. Cheap compared to a failed daily run.
    SOURCE_PAUSE=45
    fetch_idx=0
    fetch_count=4  # hf_daily, arxiv, arxiv_authors, openreview
    pause_between() {
        fetch_idx=$((fetch_idx + 1))
        if [[ "$fetch_idx" -lt "$fetch_count" ]]; then
            echo "[$(date -Is)] inter-source pause ${SOURCE_PAUSE}s"
            sleep "$SOURCE_PAUSE"
        fi
    }
    for src in hf_daily arxiv; do
        echo "[$(date -Is)] fetch: $src (--backfill-days ${BACKFILL}${FORCE_DESC})"
        uv run python -m "sources.$src" --backfill-days "${BACKFILL}" "${FORCE_FLAG[@]}" \
            || echo "  ($src returned non-zero, continuing)"
        pause_between
    done
    for src in arxiv_authors openreview; do
        echo "[$(date -Is)] fetch: $src (--window-days ${DAYS})"
        uv run python -m "sources.$src" --backfill-days 0 --window-days "${DAYS}" \
            || echo "  ($src returned non-zero, continuing)"
        pause_between
    done
fi

PIPELINE_FORCE_FLAG=()
if [[ "$FORCE" -eq 1 ]]; then PIPELINE_FORCE_FLAG=(--force); fi
run_step "dedupe"    uv run python -m pipeline.dedupe    --backfill-days "${BACKFILL}" "${PIPELINE_FORCE_FLAG[@]}" || { alert_failure "daily.sh" "step 'dedupe' failed" "$LOG_FILE"; exit 4; }
run_step "filter"    uv run python -m pipeline.filter    --backfill-days "${BACKFILL}" "${PIPELINE_FORCE_FLAG[@]}" || { alert_failure "daily.sh" "step 'filter' failed" "$LOG_FILE"; exit 5; }
run_step "summarize" uv run python -m pipeline.summarize --backfill-days "${BACKFILL}" "${PIPELINE_FORCE_FLAG[@]}" || { alert_failure "daily.sh" "step 'summarize' failed" "$LOG_FILE"; exit 6; }

# Auto-generate paper-river .org files for papers surfaced in the
# current rollup window (--window-days matches --days, default 2), so
# the paper-river backlog tracks what's in the README's latest table
# rather than every paper ever surfaced. Delegates to
# scripts/auto_paper_river.py which scans the windowed
# data/summarized/, dedupes against paper-river/, and invokes
# scripts/gen_paper_river.sh per paper (which runs the ljg-paper-river
# Claude Code skill in headless mode). PAPER_RIVER_MAX=N caps further;
# PAPER_RIVER_SKIP=1 disables this step entirely. For historical
# backfills, run auto_paper_river.py --all-history out-of-band.
# Non-fatal: pipeline ships even if generation fails.
if [[ "${PAPER_RIVER_SKIP:-0}" -eq 1 ]]; then
    echo "[$(date -Is)] auto_paper_river: skipped (PAPER_RIVER_SKIP=1)"
else
    run_step "auto_paper_river" uv run python scripts/auto_paper_river.py \
        --window-days "${DAYS}" --no-warn-sleep \
        || echo "  (auto_paper_river failed, continuing)"
fi

# Auto-translate any zh paper-river/*.org that lacks an _en.org sibling
# BEFORE render, so the resulting `_en.org` files get picked up by this
# run's render step. Runs AFTER auto_paper_river so the just-generated
# zh files get translated in the same run. Non-fatal.
run_step "translate_paper_river" uv run python scripts/translate_paper_river.py --all \
    || echo "  (paper-river translation failed, continuing)"

run_step "render"    uv run python -m pipeline.render    --backfill-days "${BACKFILL}" || { alert_failure "daily.sh" "step 'render' failed" "$LOG_FILE"; exit 7; }

# Recover days whose arxiv.json is still empty from an earlier throttle-failed
# fetch. Runs every day so a day zeroed out by a 429 storm gets re-attempted on
# the next (hopefully un-throttled) run instead of staying empty forever. Reuses
# the OAI-PMH fallback baked into sources/arxiv.py. Re-renders any recovered day,
# so its output is folded into this run's commit below. Non-fatal: a hard
# failure is logged with WARN but doesn't sink the daily run. Skip the fetch
# block above (--no-fetch) → skip this too, since we made no external calls.
if [[ "$NO_FETCH" -eq 1 ]]; then
    echo "[$(date -Is)] backfill_empty_arxiv: skipped (--no-fetch)"
elif [[ "${BACKFILL_EMPTY_SKIP:-0}" -eq 1 ]]; then
    echo "[$(date -Is)] backfill_empty_arxiv: skipped (BACKFILL_EMPTY_SKIP=1)"
else
    run_step "backfill_empty_arxiv" ./scripts/backfill_empty_arxiv.sh \
        || { echo "  (backfill_empty_arxiv reported failures, continuing)"; alert_failure "daily.sh" "backfill_empty_arxiv reported failures (non-fatal, run continued)" "$LOG_FILE"; }
fi

# Snapshot the rendered paper list into snapshots/ (single-day: <YYYYMMDD>.md;
# multi-day: <start>-<end>-<N>days.md)
# so each run leaves a browsable record (git history is preserve-only;
# this gives a side-by-side comparison surface without `git show` games).
DAYS="${DAYS}" run_step "snapshot" ./scripts/snapshot.sh || echo "  (snapshot failed, continuing)"

DATE_STR=$(date -u +%Y-%m-%d)

# Commit + push if anything changed. The window may produce multiple digest
# files, so stage the whole digests/ tree rather than just today's file.
if [[ -z "$(git status --porcelain)" ]]; then
    echo "[$(date -Is)] no changes to commit"
else
    git add data/summarized digests/ README.md INDEX.md data/seen.json data/affiliations.json snapshots/ paper-river/ 2>/dev/null || true
    if [[ -z "$(git diff --cached --name-only)" ]]; then
        echo "[$(date -Is)] nothing staged after add"
    else
        git commit -m "📚 Digest ${DATE_STR} (window=${DAYS}d)" || { echo "commit failed"; alert_failure "daily.sh" "git commit failed" "$LOG_FILE"; exit 8; }
        git push || { echo "push failed"; alert_failure "daily.sh" "git push failed" "$LOG_FILE"; exit 9; }
        echo "[$(date -Is)] pushed digest for ${DATE_STR}"
    fi
fi

echo "[$(date -Is)] daily.sh done"
# Trap fires next and prints total wall-clock.
