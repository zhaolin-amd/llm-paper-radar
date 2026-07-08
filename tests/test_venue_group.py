import json
from pathlib import Path

from pipeline.venue_group import group_by_subfield


def _scored(id_, subfield, hard_gate, title="T", abstract="A", url="https://x"):
    return {
        "id": id_,
        "title": title,
        "abstract": abstract,
        "url": url,
        "relevance_breakdown": {"hard_gate": hard_gate, "subfield": subfield, "reason": "r"},
    }


def test_group_by_subfield_excludes_hard_gated(tmp_path: Path):
    scored = [
        _scored("1", "kv_cache", False),
        _scored("2", "kv_cache", False),
        _scored("3", "quantization", False),
        _scored("4", "unknown", True),
    ]
    path = tmp_path / "scored.json"
    path.write_text(json.dumps(scored))

    groups = group_by_subfield(path)
    assert set(groups) == {"kv_cache", "quantization"}
    assert len(groups["kv_cache"]) == 2
    assert len(groups["quantization"]) == 1


def test_group_by_subfield_defaults_missing_subfield_to_unknown(tmp_path: Path):
    scored = [{"id": "1", "title": "T", "abstract": "A", "url": "https://x",
               "relevance_breakdown": {"hard_gate": False, "reason": "r"}}]
    path = tmp_path / "scored.json"
    path.write_text(json.dumps(scored))

    groups = group_by_subfield(path)
    assert set(groups) == {"unknown"}
