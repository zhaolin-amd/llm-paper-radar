import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from pipeline.venue_filter import score_papers
from sources.base import Paper, SourceRecord


def _mk(id_, title, abstract):
    return Paper(
        id=id_,
        title=title,
        authors=[],
        abstract=abstract,
        url="https://x",
        pdf_url=None,
        published_at=datetime(2026, 5, 10, tzinfo=UTC),
        primary_category="mlsys",
        categories=["mlsys"],
        sources=[SourceRecord(name="openreview", fetched_at=datetime.now(UTC))],
    )


@pytest.mark.asyncio
async def test_score_papers_assigns_subfield_and_reason(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")
    papers = [
        _mk("1", "PagedAttention for LLM Serving", "KV cache paging for vLLM."),
        _mk("2", "LoRA fine-tuning", "Parameter-efficient fine-tuning."),
    ]
    in_path = tmp_path / "in.json"
    in_path.write_text(json.dumps([p.model_dump(mode="json") for p in papers]))
    out_path = tmp_path / "scored.json"
    prompt_path = tmp_path / "p.md"
    prompt_path.write_text("score this")

    fake = AsyncMock()
    fake.call_json.side_effect = [
        {"hard_gate": False, "subfield": "kv_cache", "reason": "KV cache 分页管理"},
        {"hard_gate": True, "subfield": "other", "reason": "训练期微调，无推理角度"},
    ]

    n = await score_papers(in_path, out_path, prompt_path, fake, concurrency=2)
    assert n == 2
    out = {p["id"]: p for p in json.loads(out_path.read_text())}
    assert out["1"]["relevance_breakdown"]["hard_gate"] is False
    assert out["1"]["relevance_breakdown"]["subfield"] == "kv_cache"
    assert out["1"]["relevance_score"] == 1
    assert out["2"]["relevance_breakdown"]["hard_gate"] is True
    assert out["2"]["relevance_score"] == 0


@pytest.mark.asyncio
async def test_score_papers_records_judge_unavailable_on_failure(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")
    papers = [_mk("1", "good", "good")]
    in_path = tmp_path / "in.json"
    in_path.write_text(json.dumps([p.model_dump(mode="json") for p in papers]))
    out_path = tmp_path / "scored.json"
    prompt_path = tmp_path / "p.md"
    prompt_path.write_text("p")

    fake = AsyncMock()
    fake.call_json.side_effect = Exception("boom")

    await score_papers(in_path, out_path, prompt_path, fake, concurrency=1)
    out = json.loads(out_path.read_text())[0]
    assert out["relevance_breakdown"]["hard_gate"] is True
    assert "judge unavailable" in out["relevance_reason"]


@pytest.mark.asyncio
async def test_score_papers_skips_empty_metadata(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")
    stub = _mk("stub", "", "")
    real = _mk("real", "AWQ quantization", "Weight quantization for LLMs.")
    in_path = tmp_path / "in.json"
    in_path.write_text(json.dumps([p.model_dump(mode="json") for p in [stub, real]]))
    out_path = tmp_path / "scored.json"
    prompt_path = tmp_path / "p.md"
    prompt_path.write_text("p")

    fake = AsyncMock()
    fake.call_json.return_value = {"hard_gate": False, "subfield": "quantization", "reason": "ok"}

    await score_papers(in_path, out_path, prompt_path, fake, concurrency=2)
    out = {p["id"]: p for p in json.loads(out_path.read_text())}
    assert out["stub"]["relevance_score"] is None
    assert out["real"]["relevance_score"] == 1
    assert fake.call_json.call_count == 1
