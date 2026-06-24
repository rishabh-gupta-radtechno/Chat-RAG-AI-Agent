import asyncio
import json
import os

os.environ["DEBUG"] = "false"

from app.ai.pdf_processor import PDFProcessor
from app.ai.table_extractor import TableExtractor


class _FakeVisionLLM:
    """Stand-in for OllamaClient.vision (the real model is validated on the server)."""

    def __init__(self, reply):
        self._reply = reply

    async def vision(self, prompt, image_bytes):
        return self._reply


# A clean, uniform table (rules-based extraction did fine) — must be left alone.
_CLEAN = {
    "title": "Table",
    "header": ["Sl.No.", "Description", "Drg.No.", "Qty"],
    "rows": [["1", "Housing S/A", "2KB758", "1"], ["2*", "Filter", "4A68688", "1"]],
}
# A mangled table: ragged rows + merged/empty cells (TableFormer-style).
_MANGLED = {
    "title": "Table",
    "header": ["Sl.No.", "Description", "Drg.No.", "Qty", "Sl.No.", "Description", "Drg.No.", "Qty"],
    "rows": [
        ["1 Housing S/A", "2KB758", "1", "14", "Spring Housing", "3KB761", "1", ""],
        ["2*", "Filter", "", "", "", "", "", ""],
    ],
}


def test_low_confidence_detects_ragged_and_empty():
    assert PDFProcessor._table_is_low_confidence(_MANGLED) is True
    assert PDFProcessor._table_is_low_confidence(_CLEAN) is False
    assert PDFProcessor._table_is_low_confidence({"header": [], "rows": []}) is False
    # A single-column list is not a misalignment case.
    assert PDFProcessor._table_is_low_confidence(
        {"header": ["Item"], "rows": [["a"], ["b"]]}
    ) is False


def test_analyze_parses_tables_and_handles_failures():
    reply = json.dumps({"tables": [{
        "title": "Part List",
        "header": ["Sl.No.", "Description", "Drg.No.", "Qty"],
        "rows": [["1", "Housing S/A", "2KB758", "1"], ["2*", "Filter", "4A68688", "1"]],
    }]})
    recs = asyncio.run(TableExtractor(_FakeVisionLLM(reply)).analyze(b"PNG"))
    assert len(recs) == 1
    assert recs[0]["rows"][0] == ["1", "Housing S/A", "2KB758", "1"]

    assert asyncio.run(TableExtractor(_FakeVisionLLM('{"tables": []}')).analyze(b"x")) == []

    class _Boom:
        async def vision(self, prompt, image_bytes):
            raise RuntimeError("vision down")

    assert asyncio.run(TableExtractor(_Boom()).analyze(b"x")) == []


def test_merge_replaces_low_conf_and_keeps_clean():
    clean_vision = {
        "title": "Table",
        "header": ["Sl.No.", "Description", "Drg.No.", "Qty", "Sl.No.", "Description", "Drg.No.", "Qty"],
        "rows": [
            ["1", "Housing S/A", "2KB758", "1", "14", "Spring Housing", "3KB761", "1"],
            ["2*", "Filter", "4A68688", "1", "15", "Hex. Hd. Bolt", "IS:1363", "4"],
        ],
    }
    out = PDFProcessor._merge_vision_tables(
        [_CLEAN, _MANGLED], [False, True], [clean_vision]
    )
    assert out[0] is _CLEAN  # high-confidence table untouched
    # Low-confidence table replaced with the matching vision version (columns split).
    assert out[1]["rows"][0][0] == "1"
    assert out[1]["rows"][0][1] == "Housing S/A"


def test_merge_keeps_original_when_no_vision_match():
    unrelated = {"title": "Other", "header": ["X", "Y"], "rows": [["foo", "bar"]]}
    out = PDFProcessor._merge_vision_tables([_MANGLED], [True], [unrelated])
    assert out == [_MANGLED]  # no confident match -> keep rules-based
