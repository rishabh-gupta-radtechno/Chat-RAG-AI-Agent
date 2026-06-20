import asyncio
import json
import os

os.environ["DEBUG"] = "false"

from app.ai.chart_extractor import ChartExtractor
from app.ai.pdf_processor import PDFProcessor
from app.ai.text_processor import TextProcessor


class _FakeVisionLLM:
    """Stand-in for OllamaClient.vision (the real model is validated on the server)."""

    def __init__(self, reply):
        self._reply = reply

    async def vision(self, prompt, image_bytes):
        return self._reply


def test_parse_json_handles_fences_and_prose():
    raw = 'Here is the result:\n```json\n{"is_chart": true, "chart_type": "pie"}\n```'
    parsed = ChartExtractor._parse_json(raw)
    assert parsed and parsed["chart_type"] == "pie"
    assert ChartExtractor._parse_json('blah {"is_chart": false} trailing')["is_chart"] is False
    assert ChartExtractor._parse_json("not json at all") is None


def test_analyze_pie_chart():
    reply = json.dumps({
        "is_chart": True, "chart_type": "Pie", "title": "Reasons for DV Failure",
        "data": [{"label": "DV leakage", "value": "54%"},
                 {"label": "Data Not available", "value": "14%"}],
        "summary": "DV leakage is the dominant failure mode at 54%.",
    })
    rec = asyncio.run(ChartExtractor(_FakeVisionLLM(reply)).analyze(b"PNGBYTES"))
    assert rec["chart_type"] == "pie"  # normalized lower-case
    assert rec["title"] == "Reasons for DV Failure"
    assert len(rec["structured_data"]) == 2
    assert "54%" in rec["summary"]


def test_analyze_non_chart_and_vision_error_return_none():
    assert asyncio.run(ChartExtractor(_FakeVisionLLM('{"is_chart": false}')).analyze(b"x")) is None

    class _Boom:
        async def vision(self, prompt, image_bytes):
            raise RuntimeError("vision model down")

    assert asyncio.run(ChartExtractor(_Boom()).analyze(b"x")) is None


def test_build_pdf_chunks_creates_linked_chart_chunks():
    tp = TextProcessor()
    pages = [{
        "page_number": 1, "native_text": "word " * 100, "ocr_text": "",
        "tables": [], "diagrams": [],
        "charts": [{
            "chart_type": "pie", "title": "Reasons for DV Failure",
            "structured_data": [{"label": "DV leakage", "value": "54%"}],
            "summary": "DV leakage is the dominant failure mode at 54%.",
            "image_url": "/static/uploads/diagrams/x/1/diagram_chart.png",
        }],
    }]
    chunks = tp.build_pdf_chunks(pages, "deck.pdf")
    by_type = {
        c["metadata"]["content_type"]: c
        for c in chunks
        if str(c["metadata"].get("content_type", "")).startswith("chart")
    }
    assert {"chart_data", "chart_summary", "chart_image"} <= set(by_type), list(by_type)

    # All three linked by one related_chart id.
    ids = {by_type[t]["metadata"]["related_chart"] for t in ("chart_data", "chart_summary", "chart_image")}
    assert len(ids) == 1

    # Summary is the embedded human sentence; data chunk carries structured_data.
    assert "54%" in by_type["chart_summary"]["text"]
    assert by_type["chart_data"]["metadata"]["structured_data"][0]["value"] == "54%"
    assert "DV leakage=54%" in by_type["chart_data"]["text"]
    assert by_type["chart_image"]["metadata"]["image_url"].endswith("diagram_chart.png")


def test_build_pdf_chunks_multiple_charts_per_page():
    tp = TextProcessor()
    pages = [{
        "page_number": 2,
        "native_text": "DV Failure Make Wise DV Failure Type Wise " + "x " * 40,
        "ocr_text": "", "tables": [], "diagrams": [],
        "charts": [
            {"chart_type": "pie", "title": "Make Wise",
             "structured_data": [{"label": "ESCORTS", "value": "34%"}],
             "summary": "Escorts has the highest at 34%.", "image_url": "/a.png"},
            {"chart_type": "pie", "title": "Type Wise",
             "structured_data": [{"label": "KEO", "value": "45%"}],
             "summary": "KEO is the highest type at 45%.", "image_url": "/b.png"},
        ],
    }]
    chunks = tp.build_pdf_chunks(pages, "deck.pdf")
    chart_ids = {c["metadata"]["related_chart"] for c in chunks if c["metadata"].get("related_chart")}
    assert len(chart_ids) == 2, f"two charts should yield two related_chart ids: {chart_ids}"
    summaries = [c["text"] for c in chunks if c["metadata"].get("content_type") == "chart_summary"]
    assert any("34%" in s for s in summaries) and any("45%" in s for s in summaries)


def test_chart_region_splitting_two_charts(tmp_path):
    fitz = __import__("fitz")
    pdf = tmp_path / "twocharts.pdf"
    doc = fitz.open()
    page = doc.new_page(width=300, height=400)
    page.draw_rect(fitz.Rect(20, 50, 120, 250), fill=(0.2, 0.4, 0.6))   # left chart
    page.draw_rect(fitz.Rect(180, 50, 280, 250), fill=(0.6, 0.3, 0.3))  # right chart (gap between)
    doc.save(str(pdf))
    doc.close()
    images = PDFProcessor().chart_region_images(str(pdf), 1)
    assert len(images) == 2, f"two separated chart regions expected, got {len(images)}"


def test_chart_region_single_falls_back_to_whole_page(tmp_path):
    fitz = __import__("fitz")
    pdf = tmp_path / "onechart.pdf"
    doc = fitz.open()
    page = doc.new_page(width=300, height=400)
    page.draw_rect(fitz.Rect(40, 50, 260, 350), fill=(0.2, 0.4, 0.6))   # single region
    doc.save(str(pdf))
    doc.close()
    images = PDFProcessor().chart_region_images(str(pdf), 1)
    assert len(images) == 1
