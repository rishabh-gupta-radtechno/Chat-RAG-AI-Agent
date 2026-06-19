import os

os.environ["DEBUG"] = "false"

from app.ai.pdf_processor import PDFProcessor
from app.ai.text_processor import TextProcessor


def test_table_only_page_not_dropped():
    """A page with a table but little prose must not disappear (the page-2 bug)."""
    tp = TextProcessor()
    pages = [
        {"page_number": 1, "native_text": "1.0 INTRO " + "word " * 100, "ocr_text": "",
         "tables": [], "diagrams": []},
        {"page_number": 2, "native_text": "Test procedure intro.", "ocr_text": "",
         "tables": [{"title": "Spec", "header": ["Param", "Value"],
                     "rows": [["Pressure", "5 ksc"], ["Capacity", "3 litres"]]}],
         "diagrams": []},
    ]
    chunks = tp.build_pdf_chunks(pages, "f.pdf")
    page2 = [c for c in chunks if c["metadata"]["page_number"] == 2]
    assert page2, "page 2 (table page) disappeared"
    assert any(c["metadata"]["content_type"] == "table" for c in page2), "page-2 table missing"


def test_every_page_gets_at_least_one_chunk():
    tp = TextProcessor()
    pages = [
        {"page_number": 1, "native_text": "word " * 100, "ocr_text": "", "tables": [], "diagrams": []},
        {"page_number": 2, "native_text": "", "ocr_text": "", "tables": [], "diagrams": []},  # blank
        {"page_number": 3, "native_text": "", "ocr_text": "", "tables": [],
         "diagrams": [{"image_index": 0, "description": "Assembly drawing",
                       "ocr_text": "28 29", "image_url": "/x.png"}]},
    ]
    chunks = tp.build_pdf_chunks(pages, "f.pdf")
    present = {c["metadata"]["page_number"] for c in chunks}
    assert {1, 2, 3} <= present, f"a page disappeared: {present}"


def test_diagram_coverage_gate_keeps_drawing_skips_logo():
    p = PDFProcessor()
    calls = {"ocr": 0}
    p._run_ocr = lambda b: (calls.__setitem__("ocr", calls["ocr"] + 1) or "28 29 DETAIL-D SECTION-GG")
    p._describe_diagram = lambda b, t: f"Diagram. Labels: {t}"
    p._save_diagram_image = lambda **k: "/img/d.png"
    images = [
        {"image_index": 0, "bytes": b"BIGDRAWING", "coverage": 0.8},   # engineering drawing
        {"image_index": 1, "bytes": b"LOGO", "coverage": 0.02},        # repeated logo
    ]
    diagrams = p._extract_diagrams(images, [], file_id="f", page_number=3)
    assert len(diagrams) == 1, "coverage gate failed (logo kept or drawing dropped)"
    assert diagrams[0]["image_index"] == 0
    assert "28 29" in diagrams[0]["ocr_text"], "engineering drawing was not OCR'd"
    assert calls["ocr"] == 1, "the small logo should not have been OCR'd"


def test_image_hash_dedup_across_pages(tmp_path):
    fitz = __import__("fitz")
    from PIL import Image

    logo = tmp_path / "logo.png"
    Image.new("RGB", (80, 80), (10, 20, 30)).save(logo)
    pdf_path = tmp_path / "doc.pdf"
    doc = fitz.open()
    for _ in range(3):  # same logo on 3 pages
        page = doc.new_page(width=300, height=400)
        page.insert_image(fitz.Rect(10, 10, 90, 90), filename=str(logo))
    doc.save(str(pdf_path))
    doc.close()

    images_by_page = PDFProcessor()._extract_page_images(str(pdf_path))
    total = sum(len(v) for v in images_by_page.values())
    assert total == 1, f"identical logo not deduplicated across pages: {total}"
