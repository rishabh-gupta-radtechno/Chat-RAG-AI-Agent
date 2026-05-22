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

    chunks = processor.build_pdf_chunks(page_documents, "test-file.pdf")

    assert len(chunks) == 1
    assert chunks[0]["metadata"]["chunk_id"].startswith("test-file.pdf|page1|text|")


def test_build_pdf_chunks_embeds_ocr_table_text_with_metadata():
    from app.ai.text_processor import TextProcessor

    processor = TextProcessor()
    page_documents = [
        {
            "page_number": 3,
            "native_text": "",
            "ocr_text": "",
            "tables": [
                {
                    "content_type": "table",
                    "page_number": 3,
                    "header": ["Column1", "Column2", "Column3"],
                    "rows": [["Value1", "Value2", "Value3"]],
                    "table_text": "Column1 | Column2 | Column3\nValue1 | Value2 | Value3",
                    "confidence": 0.91,
                    "source": "ppstructure",
                }
            ],
            "diagrams": [],
        }
    ]

    chunks = processor.build_pdf_chunks(page_documents, "scan.pdf")

    assert len(chunks) == 1
    assert chunks[0]["text"] == "Column1 | Column2 | Column3\nValue1 | Value2 | Value3"
    assert chunks[0]["metadata"]["content_type"] == "table"
    assert chunks[0]["metadata"]["table_text"] == chunks[0]["text"]
    assert chunks[0]["metadata"]["page_number"] == 3
    assert chunks[0]["metadata"]["table_confidence"] == 0.91
    assert chunks[0]["metadata"]["table_source"] == "ppstructure"


def test_build_pdf_chunks_splits_long_ocr_page_by_sections_and_steps():
    from app.ai.text_processor import TextProcessor

    processor = TextProcessor()
    procedure_lines = [
        "ACME HYDRAULIC MANUAL",
        "4.4 REASSEMBLING",
        "1. Clean the traction sleeve and inspect the bore for scoring before installing the bearing.",
        "2. Apply approved grease to the adjuster nut threads and verify free movement through the full range.",
        "3. Install the ball bearing on the adjuster nut and seat the assembly squarely in the sleeve.",
        "4. Tighten the lock screw until the tab washer is secure, then check that no burrs contact the tube.",
        "5. Record the measured end float and compare it with the service limit in the inspection table.",
        "4.5 TESTING ON TEST RACK",
        "1. Mount the unit on the test rack and connect the pressure line to the calibrated supply.",
        "2. Increase pressure in three stages while checking for leakage around the barrel and end cap.",
        "3. Hold final pressure for five minutes and record the observed drop on the inspection sheet.",
    ]
    ocr_text = "\n".join(
        f"{line} Pass {cycle}."
        for cycle in range(1, 9)
        for line in procedure_lines
    )
    page_documents = [
        {
            "page_number": 7,
            "native_text": "",
            "ocr_text": ocr_text,
            "tables": [],
            "diagrams": [],
        }
    ]

    chunks = processor.build_pdf_chunks(page_documents, "manual.pdf")

    assert len(chunks) > 1
    assert all(chunk["metadata"]["content_type"] == "ocr" for chunk in chunks)
    assert all(len(chunk["text"]) <= 1100 for chunk in chunks)
    assert any("4.4 REASSEMBLING" in chunk["text"] for chunk in chunks)
    assert any("4.5 TESTING ON TEST RACK" in chunk["text"] for chunk in chunks)
    assert any("1. Clean the traction sleeve" in chunk["text"] for chunk in chunks)


def test_build_pdf_chunks_removes_repeated_page_headers_from_ocr():
    from app.ai.text_processor import TextProcessor

    processor = TextProcessor()
    page_documents = []
    for page_number in range(1, 4):
        page_documents.append(
            {
                "page_number": page_number,
                "native_text": "",
                "ocr_text": (
                    "ACME HYDRAULIC MANUAL\n"
                    "Service Division\n"
                    f"5.{page_number} INSPECTION\n"
                    "Check the bore diameter and record the result.\n"
                    "Inspect sealing faces for scratches or corrosion.\n"
                    "Replace damaged washers before final assembly.\n"
                    f"Page {page_number}\n"
                ),
                "tables": [],
                "diagrams": [],
            }
        )

    chunks = processor.build_pdf_chunks(page_documents, "manual.pdf")
    combined = "\n".join(chunk["text"] for chunk in chunks)

    assert "ACME HYDRAULIC MANUAL" not in combined
    assert "Service Division" not in combined
    assert "INSPECTION" in combined
