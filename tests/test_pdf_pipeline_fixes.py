import os

os.environ["DEBUG"] = "false"

from app.ai.pdf_processor import PDFProcessor
from app.ai.text_processor import TextProcessor


# A TableFormer-style mangled dual-column part list (serial merged into description).
_MANGLED_DOCLING = {
    "title": "Table",
    "header": ["Sl. No.", "Description", "Drg.No.", "Qty", "Sl. No.", "Description", "Drg.No.", "Qty"],
    "rows": [
        ["1 Housing S/A", "2KB758", "1", "14", "Spring Housing", "3KB761", "1", ""],
        ["2*", "Filter", "4A68688", "1", "15", "Hex.Hd.Bolt M8x45", "IS:1363", "4"],
    ],
}
# Clean line-based version of the same table (columns correctly split + spaced).
_CLEAN_LINE = {
    "title": "Table",
    "header": ["Sl.No.", "Description", "Drg.No.", "Qty", "Sl.No.", "Description", "Drg.No.", "Qty"],
    "rows": [
        ["1", "Housing S/A", "2KB758", "1", "14", "Spring Housing", "3KB761", "1"],
        ["2*", "Filter", "4A68688", "1", "15", "Hex. Hd. Bolt M8x45", "IS:1363", "4"],
    ],
}
# Repeating header/footer box the line extractor over-detects; must NOT be adopted.
_HEADER_BOX = {
    "title": "Table",
    "header": ["", "OPERATION AND MAINTENANCE MANUAL", ""],
    "rows": [["", "PRESSURE REDUCING VALVE", "Rev.00"],
             ["", "Doc. No. RED-RD-BS-014-OM-03-22", "Date:09/05/2022"]],
}


def test_refine_tables_adopts_line_based_and_skips_header_box(monkeypatch):
    proc = PDFProcessor()
    monkeypatch.setattr(proc, "_extract_tables", lambda fp, pg: [_HEADER_BOX, _CLEAN_LINE])
    out = proc._refine_tables_with_lines("x.pdf", 9, [_MANGLED_DOCLING])
    assert len(out) == 1
    # Serial split out of the description (the bug the user reported).
    assert out[0]["rows"][0][0] == "1"
    assert out[0]["rows"][0][1] == "Housing S/A"
    # The header box was never adopted.
    flat = " ".join(c for r in out[0]["rows"] for c in r)
    assert "MAINTENANCE MANUAL" not in flat


def test_refine_tables_keeps_docling_when_no_line_match(monkeypatch):
    proc = PDFProcessor()
    # Only an unrelated header box available -> no confident match -> keep Docling.
    monkeypatch.setattr(proc, "_extract_tables", lambda fp, pg: [_HEADER_BOX])
    out = proc._refine_tables_with_lines("x.pdf", 9, [_MANGLED_DOCLING])
    assert out == [_MANGLED_DOCLING]


def test_two_column_page_read_in_column_order(tmp_path):
    """A two-column page must be read left-column-then-right, not line-interleaved."""
    fitz = __import__("fitz")
    pdf = tmp_path / "twocol.pdf"
    doc = fitz.open()
    page = doc.new_page(width=600, height=400)
    page.insert_textbox(fitz.Rect(50, 30, 550, 55), "SECTION HEADER ACROSS PAGE")  # full width
    for y, txt in [(80, "Left alpha first"), (140, "Left beta second"), (200, "Left gamma third")]:
        page.insert_textbox(fitz.Rect(50, y, 280, y + 40), txt)        # left column
    for y, txt in [(80, "Right one value"), (140, "Right two value"), (200, "Right three value")]:
        page.insert_textbox(fitz.Rect(320, y, 550, y + 40), txt)       # right column
    doc.save(str(pdf))
    doc.close()

    text = PDFProcessor()._extract_text_columns(str(pdf), 1)
    assert text is not None, "two-column page not detected"
    positions = [text.index(s) for s in
                 ["SECTION HEADER", "Left alpha", "Left gamma", "Right one", "Right three"]]
    # header first, then the whole left column, then the whole right column
    assert positions == sorted(positions), f"wrong reading order: {text!r}"


def test_single_column_page_left_untouched(tmp_path):
    """A single-column page must NOT be treated as multi-column (returns None)."""
    fitz = __import__("fitz")
    pdf = tmp_path / "onecol.pdf"
    doc = fitz.open()
    page = doc.new_page(width=600, height=400)
    for i, y in enumerate([60, 120, 180, 240, 300]):
        page.insert_textbox(fitz.Rect(50, y, 550, y + 40),
                            f"Full width paragraph number {i} spanning the entire page width here.")
    doc.save(str(pdf))
    doc.close()
    assert PDFProcessor()._extract_text_columns(str(pdf), 1) is None


def test_validate_table_keeps_real_digital_table_in_prose():
    """The Faiveley part-list bug: a clean wide table whose cells are also in the
    page text layer must NOT be rejected by the page-similarity rule (Rule 4)."""
    tp = TextProcessor()
    table = {
        "title": "Table",
        "header": ["Sl. No.", "FTIL Part No.", "Qty/Set", "Description"],
        "rows": [
            ["1", "501 0020 00", "1", "C3W2 DV (Al)"],
            ["2", "602 0013 00", "1", "Combined Sandwich Piece Assembly"],
            ["3", "601 0011 00", "1", "Common Pipe Bracket Assembly"],
            ["4", "604 0012 00", "1", "Control Reservoir - 6 ltrs"],
            ["7", "780 2639 00", "1", "N-1 Reducing Valve with 24-A Double Check Valve"],
        ],
    }
    # Page prose CONTAINS the table cells (Docling left the table in the text layer).
    page_text = (
        "8.0 LIST OF MAIN ITEMS ON AIR BRAKE SYSTEM FOR CONCOR WAGONS "
        "Sl. No. FTIL Part No. Qty/Set Description "
        "1 501 0020 00 1 C3W2 DV (Al) 2 602 0013 00 1 Combined Sandwich Piece Assembly "
        "3 601 0011 00 1 Common Pipe Bracket Assembly 4 604 0012 00 1 Control Reservoir 6 ltrs "
        "7 780 2639 00 1 N-1 Reducing Valve with 24-A Double Check Valve"
    )
    assert tp.validate_table(table, page_text) is True


def test_validate_table_keeps_wide_table_with_paragraph_cells():
    """A wide (>=3 col) table legitimately has paragraph cells (component/reference/
    description matrix) and must not be rejected by paragraph-dominance (Rule 3)."""
    tp = TextProcessor()
    table = {
        "header": ["SN", "Air Brake Component", "Reference Document", "Description"],
        "rows": [
            ["a)", "Distributor Valve",
             "OEM Maintenance Manuals Escorts MM-AB/DV-KEO KBIPL GD21266 Greysham "
             "C3W Catalogue C3W2 Catalogue FTRIL SD Technical Stone India",
             "Comprehensive documents on maintenance and overhaul procedure of the "
             "Distributor valve covering tools fixtures lubricants testing troubleshooting "
             "spare parts kits and replacement procedures"],
            ["b)", "Air Brake Hose Coupling",
             "RDSO letter no MW.APB dated 23/25.11.24",
             "Instructions to ensure that the TOP markings on the hose nipple and palm end "
             "coupling head must be properly aligned during assembly and installation to "
             "prevent torsion and uncoupling"],
        ],
    }
    assert tp.validate_table(table, "") is True


def test_validate_table_passes_short_value_two_column_table():
    """A 2-column table of short discrete values (codes/counts) survives even when
    its text is in the page prose."""
    tp = TextProcessor()
    table = {"header": ["Zonal Railway", "Count"],
             "rows": [["SWR", "71"], ["ECoR", "40"], ["SECR", "123"]]}
    page_text = "Zonal Railway Count SWR 71 ECoR 40 SECR 123 total removed for investigation"
    assert tp.validate_table(table, page_text) is True


def test_validate_table_still_rejects_prose_in_two_column_grid():
    """Rule 4 still catches the real case: a paragraph chopped into a 2-column grid."""
    tp = TextProcessor()
    table = {"header": ["A", "B"],
             "rows": [["the changeover valve inside", "the double check valve moves"],
                      ["to apply the brake in", "proportion to the depletion caused"]]}
    page_text = ("the changeover valve inside the double check valve moves to apply the "
                 "brake in proportion to the depletion caused in the brake pipe pressure")
    assert tp.validate_table(table, page_text) is False


def test_table_chunk_carries_structured_json_metadata():
    """Tables persist header/rows as structured JSON metadata (parity with charts)."""
    tp = TextProcessor()
    pages = [{
        "page_number": 1, "native_text": "Specification overview " + "word " * 60, "ocr_text": "",
        "tables": [{"title": "Spec", "header": ["Param", "Value"],
                    "rows": [["Pressure", "5 ksc"], ["Capacity", "3 litres"]]}],
        "diagrams": [],
    }]
    chunks = tp.build_pdf_chunks(pages, "f.pdf")
    table_chunks = [c for c in chunks if c["metadata"].get("content_type") == "table"]
    assert table_chunks, "table chunk missing"
    meta = table_chunks[0]["metadata"]
    assert meta["table_id"] == "table_1_1"
    assert meta["table_header"] == ["Param", "Value"]
    assert ["Pressure", "5 ksc"] in meta["table_rows"]
    assert ["Capacity", "3 litres"] in meta["table_rows"]
    # Embedded text stays markdown + key=value (not JSON).
    assert "Param=Pressure" in table_chunks[0]["text"] or "| Pressure |" in table_chunks[0]["text"]


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
