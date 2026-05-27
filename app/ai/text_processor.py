"""
Text processing utilities including chunking and preprocessing.
Supports multilingual content with Devanagari (Hindi/Marathi/Nepali/Sanskrit).
"""

import re
import unicodedata
from typing import Dict, List, Optional

from app.core.config import get_settings

settings = get_settings()


def detect_script(text: str) -> str:
    """Return 'devanagari', 'mixed', or 'latin' based on character distribution."""
    if not text:
        return "latin"
    devanagari = sum(1 for c in text if "ऀ" <= c <= "ॿ")
    total_alpha = sum(1 for c in text if c.isalpha())
    if total_alpha == 0:
        return "latin"
    ratio = devanagari / total_alpha
    if ratio >= 0.6:
        return "devanagari"
    if ratio >= 0.15:
        return "mixed"
    return "latin"


def normalize_unicode(text: str) -> str:
    """
    NFC normalization + zero-width char removal.
    Critical for Devanagari: joins split matras back to base characters
    so embedding models see consistent token forms.
    """
    if not text:
        return text
    text = unicodedata.normalize("NFC", text)
    for zw in ("​", "‌", "‍", "﻿"):
        text = text.replace(zw, "")
    return text


class TextProcessor:
    """Text processing utilities with multilingual support."""

    @staticmethod
    def chunk_text(
        text: str,
        chunk_size: int = settings.chunk_size,
        overlap: int = settings.chunk_overlap,
    ) -> list[str]:
        """Split text into chunks with overlap."""
        chunks = []
        words = text.split()

        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i : i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)

        return chunks

    @staticmethod
    def split_sentences(text: str) -> list[str]:
        """
        Split text into sentences respecting both Latin and Devanagari boundaries.
        Devanagari uses danda (।) and double danda (॥) as sentence terminators.
        """
        # Split on: danda, double danda, and common Latin sentence terminators
        parts = re.split(r"[।॥\n]|(?<=[.!?])\s+", text)
        return [p.strip() for p in parts if p.strip()]

    @staticmethod
    def chunk_by_sentences(
        text: str,
        max_chunk_size: int = settings.chunk_size,
    ) -> list[str]:
        """Split text into chunks by sentences (Latin + Devanagari aware)."""
        sentences = TextProcessor.split_sentences(text)
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if not sentence:
                continue

            if len(current_chunk) + len(sentence) <= max_chunk_size:
                current_chunk += " " + sentence if current_chunk else sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text, preserving Devanagari and Unicode characters."""
        text = normalize_unicode(text)
        # Collapse whitespace but keep Unicode word chars (including Devanagari)
        text = re.sub(r"\s+", " ", text)
        # Remove control characters only — preserve all printable Unicode
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
        return text.strip()

    @staticmethod
    def clean_pdf_page_text(text: str) -> str:
        """Normalize PDF page text while preserving document-specific content."""
        text = normalize_unicode(text)
        cleaned_lines = []
        boilerplate_patterns = [
            r"^Page\s+\d+$",
            r"^\d+\s*/\s*\d+$",
        ]

        for raw_line in text.splitlines():
            line = re.sub(r"\s+", " ", raw_line).strip()
            if not line:
                continue
            if any(re.search(pattern, line, flags=re.IGNORECASE) for pattern in boilerplate_patterns):
                continue
            cleaned_lines.append(line)

        return "\n".join(cleaned_lines).strip()

    @staticmethod
    def chunk_text_by_words(
        text: str,
        chunk_size: int = settings.pdf_chunk_size,
        overlap: int = settings.pdf_chunk_overlap,
    ) -> list[str]:
        """Split text into smaller page-local chunks by words."""
        if not text:
            return []

        words = text.split()
        if len(words) <= chunk_size:
            return [text.strip()]

        chunks: list[str] = []
        step = max(1, chunk_size - overlap)
        for start in range(0, len(words), step):
            chunk = " ".join(words[start : start + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        return chunks

    @staticmethod
    def chunk_text_by_sentences(
        text: str,
        target_words: int = settings.pdf_chunk_size,
        overlap_words: int = settings.pdf_chunk_overlap,
    ) -> list[str]:
        """
        Sentence-boundary-aware chunking.
        Respects both Latin full-stops and Devanagari danda (।).
        Never breaks in the middle of a sentence.
        """
        if not text:
            return []

        sentences = TextProcessor.split_sentences(text)
        if not sentences:
            return TextProcessor.chunk_text_by_words(text, target_words, overlap_words)

        chunks: list[str] = []
        current: list[str] = []
        current_words = 0

        for sentence in sentences:
            sentence_words = len(sentence.split())

            if current_words + sentence_words > target_words and current:
                chunks.append(" ".join(current))
                # Keep last few sentences as overlap
                overlap_buf: list[str] = []
                overlap_count = 0
                for s in reversed(current):
                    w = len(s.split())
                    if overlap_count + w > overlap_words:
                        break
                    overlap_buf.insert(0, s)
                    overlap_count += w
                current = overlap_buf + [sentence]
                current_words = sum(len(s.split()) for s in current)
            else:
                current.append(sentence)
                current_words += sentence_words

        if current:
            chunks.append(" ".join(current))

        return chunks

    @staticmethod
    def extract_text_from_pdf(filepath: str) -> str:
        """Extract text from PDF file."""
        try:
            import pypdf

            text = ""
            with open(filepath, "rb") as f:
                pdf_reader = pypdf.PdfReader(f)
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text += f"\n[Page {page_num + 1}]\n{page_text}"
            return text
        except ImportError:
            raise ImportError("pypdf is required for PDF processing")
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {e}")

    @staticmethod
    def render_table_to_markdown(table: dict) -> str:
        """Convert a structured table to Markdown — preserves structure for embedding."""
        header = table.get("header", [])
        rows = table.get("rows", [])
        title = table.get("title") or "Table"

        lines = [f"**{title}**"]
        if header:
            lines.append("| " + " | ".join(str(h) for h in header) + " |")
            lines.append("|" + "|".join("---" for _ in header) + "|")
        for row in rows:
            lines.append("| " + " | ".join(str(c).strip() for c in row) + " |")
        return "\n".join(lines)

    @staticmethod
    def render_table_to_text(table: dict) -> str:
        """Convert a structured table into a narrative text representation."""
        header = table.get("header", [])
        rows = table.get("rows", [])
        title = table.get("title") or "Table"

        text_lines = [f"Table: {title}"]
        if header:
            text_lines.append(f"Columns: {', '.join(header)}")

        for row_index, row in enumerate(rows, start=1):
            if len(row) == len(header):
                row_dict = {header[i]: str(row[i]).strip() for i in range(len(header))}
                row_text = ", ".join([f"{k}: {v}" for k, v in row_dict.items()])
            else:
                row_text = " | ".join([str(cell).strip() for cell in row])
            text_lines.append(f"Row {row_index}: {row_text}")

        return "\n".join(text_lines).strip()

    def build_pdf_chunks(self, page_documents: list[dict], file_name: str) -> list[dict]:
        """Create metadata-rich chunks for a PDF with page-aware, language-aware sections."""
        chunks: list[dict] = []
        seen_chunk_ids: set[str] = set()

        def _add_chunk(text: str, metadata: dict) -> None:
            chunk_text = normalize_unicode(text.strip())
            if not chunk_text:
                return

            chunk_id = metadata.get("chunk_id")
            if not chunk_id:
                chunk_id = (
                    f"{file_name}|page{metadata.get('page_number', 0)}"
                    f"|{metadata.get('content_type', 'text')}|{len(chunks) + 1:03d}"
                )
                metadata["chunk_id"] = chunk_id

            if chunk_id in seen_chunk_ids:
                return

            seen_chunk_ids.add(chunk_id)
            # Derive script/language from final text if not already set
            if "script" not in metadata:
                metadata["script"] = detect_script(chunk_text)

            chunks.append({"text": chunk_text, "metadata": metadata})

        for page in page_documents:
            page_number = page.get("page_number", 0)
            page_script = page.get("script", "latin")
            page_ocr_confidence = page.get("ocr_confidence", 1.0)
            document_page_number = self.extract_document_page_number(page.get("native_text", ""))
            section_counters = {"text": 0, "ocr": 0, "table": 0, "diagram": 0}

            base_meta = {
                "file_name": file_name,
                "page_number": page_number,
                "document_page_number": document_page_number,
                "ocr_confidence": page_ocr_confidence,
                "script": page_script,
            }

            if page.get("native_text"):
                native_text = self.clean_pdf_page_text(page["native_text"])
                for chunk_text in self.chunk_text_by_sentences(native_text):
                    section_counters["text"] += 1
                    _add_chunk(
                        chunk_text,
                        {
                            **base_meta,
                            "content_type": "text",
                            "ocr_confidence": 1.0,  # Native text has no OCR uncertainty
                            "chunk_id": f"{file_name}|page{page_number}|text|{section_counters['text']:03d}",
                        },
                    )

            # Include OCR text only when no native text was extracted
            include_ocr = page.get("ocr_text") and not page.get("native_text")
            if include_ocr:
                for chunk_text in self.chunk_text_by_sentences(page["ocr_text"]):
                    section_counters["ocr"] += 1
                    _add_chunk(
                        chunk_text,
                        {
                            **base_meta,
                            "content_type": "ocr",
                            "chunk_id": f"{file_name}|page{page_number}|ocr|{section_counters['ocr']:03d}",
                        },
                    )

            for table_index, table in enumerate(page.get("tables", []), start=1):
                # Keep tables as Markdown — better structure for LLMs and embeddings
                table_md = self.render_table_to_markdown(table)
                table_chunks = self.chunk_table_text(table_md, table)
                for part_index, chunk_text in enumerate(table_chunks, start=1):
                    _add_chunk(
                        chunk_text,
                        {
                            **base_meta,
                            "content_type": "table",
                            "has_table": True,
                            "ocr_confidence": page_ocr_confidence,
                            "chunk_id": f"{file_name}|page{page_number}|table|{table_index:03d}.{part_index:02d}",
                        },
                    )

            for diagram in page.get("diagrams", []):
                section_counters["diagram"] += 1
                description = diagram.get("description", "").strip()
                if not description:
                    continue
                _add_chunk(
                    description,
                    {
                        **base_meta,
                        "content_type": "diagram",
                        "has_table": False,
                        "image_index": diagram.get("image_index"),
                        "diagram_description": description,
                        "diagram_ocr_text": diagram.get("ocr_text", "").strip(),
                        "image_url": diagram.get("image_url"),
                        "chunk_id": f"{file_name}|page{page_number}|diagram|{section_counters['diagram']:03d}",
                    },
                )

        return chunks

    @staticmethod
    def extract_document_page_number(text: str) -> Optional[int]:
        matches = re.findall(r"\bPage\s+(\d+)\b", text or "", flags=re.IGNORECASE)
        if not matches:
            return None
        return int(matches[-1])

    def chunk_table_text(self, table_text: str, table: dict) -> list[str]:
        """Create chunks for table text that preserve headers and structure."""
        header = table.get("header", [])
        rows = table.get("rows", [])
        if not rows:
            return self.chunk_text_by_words(table_text)

        # Small tables: keep as single chunk (never split tables if possible)
        if len(rows) <= 5:
            return [table_text]

        # Large tables: chunk by rows, preserving header in each chunk
        chunks: list[str] = []
        header_text = f"**{table.get('title', 'Table')}**\n| " + " | ".join(header) + " |\n|" + "|".join("---" for _ in header) + "|"
        rows_per_chunk = max(3, settings.pdf_chunk_size // 80)
        for start in range(0, len(rows), rows_per_chunk):
            group = rows[start : start + rows_per_chunk]
            row_lines = []
            for row in group:
                row_lines.append("| " + " | ".join(str(c).strip() for c in row) + " |")
            chunks.append(header_text + "\n" + "\n".join(row_lines))
        return chunks
