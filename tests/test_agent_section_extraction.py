import os

os.environ["DEBUG"] = "false"

from app.ai.agent import ReActAgent


def test_extract_exact_section_for_one_word_manual_heading():
    agent = ReActAgent(rag_pipeline=None)
    documents = [
        {
            "filename": "manual.pdf",
            "page_number": 2,
            "chunk_index": 1,
            "chunk_text": (
                "CONTENTS\n"
                "4.3 LUBRICATION\n"
                "4.4 REASSEMBLING\n"
                "4.5 TESTING ON TEST RACK\n"
            ),
        },
        {
            "filename": "manual.pdf",
            "page_number": 12,
            "chunk_index": 1,
            "chunk_text": (
                "4.3 LUBRICATION\n"
                "After completing cleaning, coat all parts with grease.\n"
                "4.4 REASSEMBLING\n"
                "Fig.18 & 19\n"
                "Clamp traction sleeve 12 lightly in a vice close to the lower end as shown in figure.\n"
                "Place ball bearing 7 on adjuster nut 36 and place adjuster nut in position in traction sleeve 12.\n"
            ),
        },
        {
            "filename": "manual.pdf",
            "page_number": 13,
            "chunk_index": 1,
            "chunk_text": (
                "Screw leader nut unit into place in barrel 22 and fix tab washer 34 and lock screw 33.\n"
                "Fig.10 Screw adjuster ear 28 into threaded end of adjuster tube 41.\n"
                "4.5 TESTING ON TEST RACK\n"
                "Testing starts here."
            ),
        },
    ]

    answer = agent._extract_procedure_answer("REASSEMBLING", documents)

    assert "4.4 REASSEMBLING" in answer
    assert "4.3 LUBRICATION" not in answer
    assert "Clamp traction sleeve 12 lightly" in answer
    assert "Fig.10 Screw adjuster ear 28" in answer
    assert "4.5 TESTING ON TEST RACK" not in answer


def test_build_pdf_chunks_includes_diagram_image_url():
    from app.ai.text_processor import TextProcessor

    processor = TextProcessor()
    page_documents = [
        {
            "page_number": 5,
            "native_text": "",
            "ocr_text": "",
            "tables": [],
            "diagrams": [
                {
                    "image_index": 0,
                    "description": "A test diagram.",
                    "ocr_text": "diagram labels",
                    "image_url": "/static/uploads/diagrams/test-file/5/diagram_0.png",
                }
            ],
        }
    ]

    chunks = processor.build_pdf_chunks(page_documents, "test-file.pdf")
    assert len(chunks) == 1
    assert chunks[0]["metadata"]["content_type"] == "diagram"
    assert chunks[0]["metadata"]["image_url"] == "/static/uploads/diagrams/test-file/5/diagram_0.png"
    assert chunks[0]["metadata"]["diagram_description"] == "A test diagram."


def test_build_pdf_chunks_deduplicates_repeated_page_chunks():
    from app.ai.text_processor import TextProcessor

    processor = TextProcessor()
    page_documents = [
        {
            "page_number": 1,
            "native_text": "Test page content.",
            "ocr_text": "",
            "tables": [],
            "diagrams": [],
        },
        {
            "page_number": 1,
            "native_text": "Test page content.",
            "ocr_text": "",
            "tables": [],
            "diagrams": [],
        },
    ]

    # build_pdf_chunks no longer drops pages and emits one chunk per page segment;
    # identical repeated content is collapsed by the dedup layer (deduplicate_chunks),
    # which is where dedup now lives (it also runs embedding dedup in the RAG pipeline).
    chunks = processor.build_pdf_chunks(page_documents, "test-file.pdf")
    deduped = processor.deduplicate_chunks(chunks)

    assert len(deduped) == 1
    assert deduped[0]["metadata"]["content_type"] == "text"
    assert deduped[0]["text"].strip().endswith("Test page content.")
