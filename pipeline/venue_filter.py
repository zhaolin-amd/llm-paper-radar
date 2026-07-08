"""LLM relevance/subfield scoring for a one-shot venue batch (not the daily
incremental pipeline — see pipeline/filter.py for that). No prefilter and no
milestone override: this is a single-pass classification over a venue's full
accepted-paper set, not a recurring cron stage."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from pipeline.llm_client import LLMClient, load_prompt
from sources.base import Paper

_BREAKDOWN_FIELDS = ("hard_gate", "subfield", "reason")


def _hard_gate_result(reason: str) -> dict:
    return {"hard_gate": True, "subfield": "unknown", "reason": reason}


def _composite(hard_gate: bool) -> int:
    return 0 if hard_gate else 1


async def _score_one(paper: Paper, prompt: str, client: LLMClient, sem: asyncio.Semaphore) -> Paper:
    if not paper.title.strip() or not paper.abstract.strip():
        paper.relevance_score = None
        paper.relevance_reason = "skipped: missing title/abstract"
        return paper

    user_msg = f"Title: {paper.title}\n\nAbstract: {paper.abstract}"
    async with sem:
        try:
            result = await client.call_json(prompt, user_msg, max_tokens=512)
        except Exception as e:
            cause = e
            inner = getattr(e, "last_attempt", None)
            if inner is not None and inner.failed:
                try:
                    cause = inner.exception() or e
                except Exception:
                    pass
            print(f"venue_filter: paper {paper.id} failed: {type(cause).__name__}: {cause}")
            result = _hard_gate_result(f"judge unavailable: {type(cause).__name__}: {cause}"[:200])

    hard_gate = bool(result.get("hard_gate"))
    paper.relevance_score = _composite(hard_gate)
    paper.relevance_reason = str(result.get("reason", ""))
    paper.relevance_breakdown = {k: result.get(k) for k in _BREAKDOWN_FIELDS}
    return paper


async def score_papers(
    in_path: Path,
    out_path: Path,
    prompt_path: Path,
    client: LLMClient,
    concurrency: int = 20,
) -> int:
    raw = await asyncio.to_thread(Path(in_path).read_text)
    papers = [Paper.model_validate(p) for p in json.loads(raw)]
    prompt = load_prompt(prompt_path)
    sem = asyncio.Semaphore(concurrency)
    scored = await asyncio.gather(*[_score_one(p, prompt, client, sem) for p in papers])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output = json.dumps([p.model_dump(mode="json") for p in scored], indent=2)
    await asyncio.to_thread(out_path.write_text, output)
    return len(scored)


if __name__ == "__main__":
    import click

    @click.command()
    @click.option("--in-path", required=True, type=click.Path(path_type=Path))
    @click.option("--out-path", required=True, type=click.Path(path_type=Path))
    @click.option("--prompt-path", default="prompts/inference_relevance.md", type=click.Path(path_type=Path))
    @click.option("--model", default="claude-sonnet-4-6")
    @click.option("--concurrency", default=20, type=int)
    def main(in_path: Path, out_path: Path, prompt_path: Path, model: str, concurrency: int):
        client = LLMClient(model=model)
        n = asyncio.run(score_papers(in_path, out_path, prompt_path, client, concurrency))
        print(f"venue_filter: scored {n} papers -> {out_path}")

    main()
