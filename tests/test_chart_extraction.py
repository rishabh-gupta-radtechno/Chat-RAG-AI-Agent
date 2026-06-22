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


def test_analyze_many_splits_two_pies_in_one_image():
    reply = json.dumps({"charts": [
        {"chart_type": "Pie", "title": "DV Failure Make Wise",
         "data": [{"label": "ESCORTS", "value": "34%"}],
         "summary": "Escorts make has the highest failure at 34%."},
        {"chart_type": "pie", "title": "DV Failure Type Wise",
         "data": [{"label": "KEO", "value": "45%"}],
         "summary": "KEO type is highest at 45%."},
    ]})
    recs = asyncio.run(ChartExtractor(_FakeVisionLLM(reply)).analyze_many(b"PNG"))
    assert len(recs) == 2
    assert {r["title"] for r in recs} == {"DV Failure Make Wise", "DV Failure Type Wise"}
    assert recs[0]["chart_type"] == "pie"  # normalized


def test_analyze_many_accepts_single_object_and_rejects_non_chart():
    single = json.dumps({"is_chart": True, "chart_type": "bar", "title": "T",
                         "data": [{"label": "a", "value": 1}], "summary": "s"})
    assert len(asyncio.run(ChartExtractor(_FakeVisionLLM(single)).analyze_many(b"x"))) == 1
    assert asyncio.run(ChartExtractor(_FakeVisionLLM('{"charts": []}')).analyze_many(b"x")) == []
    assert asyncio.run(ChartExtractor(_FakeVisionLLM('{"is_chart": false}')).analyze_many(b"x")) == []

    class _Boom:
        async def vision(self, prompt, image_bytes):
            raise RuntimeError("vision down")

    assert asyncio.run(ChartExtractor(_Boom()).analyze_many(b"x")) == []


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


def test_format_chart_value_flattens_lists_and_dicts():
    f = TextProcessor._format_chart_value
    assert f(["94", "57"]) == "94, 57"
    assert f("54%") == "54%"
    assert f({"Found": 94, "Replaced": 71}) == "Found: 94, Replaced: 71"


def test_build_pdf_chunks_renders_series_list_without_python_repr():
    tp = TextProcessor()
    pages = [{
        "page_number": 3, "native_text": "DV Isolated Vs Replaced " + "x " * 40, "ocr_text": "",
        "tables": [], "diagrams": [],
        "charts": [{
            "chart_type": "bar", "title": "DV Isolated Vs Replaced",
            "structured_data": [{"label": "Found Defective", "value": ["94", "57"]}],
            "summary": "Comparison across zones.", "image_url": "/x.png",
        }],
    }]
    chunks = tp.build_pdf_chunks(pages, "deck.pdf")
    data_chunk = next(c for c in chunks if c["metadata"].get("content_type") == "chart_data")
    assert "Found Defective=94, 57" in data_chunk["text"]
    assert "['94'" not in data_chunk["text"]  # never leak python list repr


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


def test_looks_like_chart_text_separates_charts_from_prose():
    P = PDFProcessor
    pie = ("Reasons for DV Failure Brake not applied 5% DV malfunctioning 8% DV leakage 54% "
           "DV piston rolling 1% DV R charger defective 3% Brake Auto release 1% "
           "Release choke defective 2% DV sensivity issue 2% Other DV defects 8% Data Not available 14%")
    bar = ("DV Isolated Vs Replaced 500 457 450 435 424 404 400 371 373 358 226 197 131 123 "
           "103 94 93 71 57 56 55 49 40 26 20 16 11 SWR ECoR SECR WR NER NCR SR ECR NWR")
    prose = ("Key Observations DV leakage is the most major failure in Zonal Railways. "
             "Escorts make DV has highest failure. KEO type DV is produced by Escorts and KNORR. "
             "NER and SCR have minimal DV failures.")
    assert P._looks_like_chart_text(pie) is True       # the page-1 pie (only 2 drawings) is now caught
    assert P._looks_like_chart_text(bar) is True
    assert P._looks_like_chart_text(prose) is False     # text slide is skipped
    assert P._looks_like_chart_text("") is False


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
