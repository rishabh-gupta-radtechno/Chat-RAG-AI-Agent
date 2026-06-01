import os

os.environ["DEBUG"] = "false"

from app.ai.pdf_processor import PDFProcessor
from app.ai.text_processor import TextProcessor


def test_clean_pdf_page_text_preserves_document_specific_content():
    text = """
    OPERATION AND MAINTENANCE MANUAL
    PRESSURE REDUCING VALVE Rev.2
    Page 12
    12 / 40
    Set pressure is 5 bar.
    """

    cleaned = TextProcessor.clean_pdf_page_text(text)

    assert "OPERATION AND MAINTENANCE MANUAL" in cleaned
    assert "PRESSURE REDUCING VALVE Rev.2" in cleaned
    assert "Set pressure is 5 bar." in cleaned
    assert "Page 12" not in cleaned
    assert "12 / 40" not in cleaned


def test_collect_paddle_text_supports_nested_ocr_shape():
    processor = PDFProcessor()
    results = [
        [
            [[[0, 0], [10, 0], [10, 10], [0, 10]], ("Accepted line", 0.95)],
            [[[0, 20], [10, 20], [10, 30], [0, 30]], ("Low confidence", 0.10)],
        ]
    ]

    assert processor._collect_paddle_text(results) == ["Accepted line"]


def test_collect_paddle_text_supports_dict_ocr_shape():
    processor = PDFProcessor()
    results = {
        "rec_texts": ["First", "Second"],
        "rec_scores": [0.99, 0.98],
    }

    assert processor._collect_paddle_text(results) == ["First", "Second"]


def test_iter_docling_pages_handles_dict_like_page_map():
    processor = PDFProcessor()

    class FakeDoc:
        def __init__(self):
            self.pages = {1: "first page", 2: "second page"}

    page_items = processor._iter_docling_pages(FakeDoc())

    assert page_items == [(1, "first page"), (2, "second page")]
