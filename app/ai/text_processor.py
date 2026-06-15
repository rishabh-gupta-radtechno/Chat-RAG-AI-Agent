"""
Text processing utilities including chunking and preprocessing.
"""

import re
from typing import Dict, List, Optional

from app.core.config import get_settings

settings = get_settings()


class TextProcessor:
    """Text processing utilities."""

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
    def chunk_by_sentences(
        text: str,
        max_chunk_size: int = settings.chunk_size,
    ) -> list[str]:
        """Split text into chunks by sentences."""
        sentences = re.split(r"[.!?]+", text)
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            sentence = sentence.strip()
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
        """Clean and normalize text."""
        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text)
        # Remove special characters (keep alphanumeric, spaces, and basic punctuation)
        text = re.sub(r"[^\w\s.,-]", "", text)
        return text.strip()

    @staticmethod
    def clean_pdf_page_text(text: str) -> str:
        """Normalize PDF page text while preserving document-specific content."""
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

    # Lead characters that appear when UTF-8 text was wrongly decoded as
    # CP1252/Latin-1 (classic "mojibake"). Non-Latin scripts never carry these.
    _MOJIBAKE_MARKERS = ("Ã", "Â", "â")
    # Last-resort targeted fixes (explicit codepoints) for the common sequences.
    _MOJIBAKE_REPLACEMENTS = {
        "Â°": "°",            # Â°  -> °
        "â€œ": "“",      # â€œ -> "
        "â€": "”",      # â€ -> "
        "â€™": "’",      # â€™ -> '
        "â€˜": "‘",      # â€˜ -> '
        "â€“": "–",      # â€“ -> –
        "â€”": "—",      # â€” -> —
        "â€¦": "…",      # â€¦ -> …
    }

    @staticmethod
    def fix_mojibake(text: str) -> str:
        """Repair UTF-8 text that was double-encoded as CP1252/Latin-1.

        e.g. ``180Â°`` -> ``180°``, ``Sheet â€" 8`` -> ``Sheet – 8``. It is a
        no-op for clean text and for non-Latin scripts (Devanagari etc.), which
        never contain the marker characters.
        """
        if not text or not any(m in text for m in TextProcessor._MOJIBAKE_MARKERS):
            return text

        # Best effort: ftfy handles mixed/partial/multi-layer mojibake correctly.
        try:
            import ftfy

            return ftfy.fix_text(text)
        except Exception:
            pass

        def _marker_count(value: str) -> int:
            return sum(value.count(m) for m in TextProcessor._MOJIBAKE_MARKERS)

        # Fallback: reverse a single (CP1252|Latin-1)->UTF-8 layer, kept only when
        # it actually reduces the mojibake (so clean text is never corrupted).
        # Both codecs are tried because the original mis-decode may have used
        # either — Latin-1 also handles the 0x80-0x9F bytes CP1252 leaves undefined.
        for codec in ("cp1252", "latin-1"):
            try:
                repaired = text.encode(codec, errors="strict").decode("utf-8", errors="strict")
            except (UnicodeEncodeError, UnicodeDecodeError):
                continue
            if _marker_count(repaired) < _marker_count(text):
                return repaired

        for bad, good in TextProcessor._MOJIBAKE_REPLACEMENTS.items():
            text = text.replace(bad, good)
        return text

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
    def extract_text_from_pdf(filepath: str) -> list[tuple[int, str]]:
        """Extract text from PDF file, returning a list of (page_number, text) tuples."""
        try:
            import pypdf

            pages = []
            with open(filepath, "rb") as f:
                pdf_reader = pypdf.PdfReader(f)
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        pages.append((page_num + 1, page_text))
            return pages
        except ImportError:
            raise ImportError("pypdf is required for PDF processing")
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {e}")

    @staticmethod
    def render_table_to_text(table: dict) -> str:
        """Convert a structured table into a narrative text representation with better structure."""
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

    # Page types that add no retrieval value — skipped entirely at ingest
    _SKIP_PAGE_TYPES = {"cover", "contact", "revision", "copyright", "catalog"}

    @staticmethod
    def _detect_page_type(text: str, page_number: int) -> str:
        t = (text or "").lower()
        words = len(t.split())

        if page_number <= 2 and words < 80:
            return "cover"
        if re.search(r"\btable\s+of\s+contents?\b|\bcontents\b", t) and words < 600:
            return "toc"
        if re.search(r"\brevision\s+(history|record)\b|\bamendment\s+record\b|\bversion\s+history\b", t):
            return "revision"
        if re.search(r"\bcopyright\b|©|\ball\s+rights?\s+reserved\b", t) and words < 250:
            return "copyright"
        if re.search(r"\bcontact\b.{0,40}\b(us|information|details)\b|\b(phone|tel|fax|email)\b.{0,60}\b(address|office)\b", t) and words < 300:
            return "contact"
        if re.search(r"\bcatalog(?:ue)?\b|\bpart\s+(number|no)\.?\s+list\b", t) and words < 400:
            return "catalog"
        return "content"

    def build_pdf_chunks(self, page_documents: list[dict], file_name: str) -> list[dict]:
        """Create metadata-rich chunks for a PDF with page-aware sections."""
        chunks: list[dict] = []
        seen_chunk_ids: set[str] = set()

        def _add_chunk(text: str, metadata: dict) -> None:
            # Repair encoding mojibake so the same clean text is embedded,
            # keyword-searched, and quoted back in answers.
            chunk_text = self.fix_mojibake(text).strip()
            if not chunk_text:
                return

            chunk_id = metadata.get("chunk_id")
            if not chunk_id:
                chunk_id = f"{file_name}|page{metadata.get('page_number', 0)}|{metadata.get('content_type', 'text')}|{len(chunks) + 1:03d}"
                metadata["chunk_id"] = chunk_id

            if chunk_id in seen_chunk_ids:
                return

            seen_chunk_ids.add(chunk_id)
            chunks.append(
                {
                    "text": chunk_text,
                    "metadata": metadata,
                }
            )

        for page in page_documents:
            page_number = page.get("page_number", 0)
            page_text = page.get("native_text") or page.get("ocr_text") or ""
            page_type = self._detect_page_type(page_text, page_number)

            if page_type in self._SKIP_PAGE_TYPES:
                continue

            document_page_number = self.extract_document_page_number(page.get("native_text", ""))
            section_counters = {
                "text": 0,
                "ocr": 0,
                "table": 0,
                "diagram": 0,
            }

            if page.get("native_text"):
                native_text = self.clean_pdf_page_text(page["native_text"])
                for chunk_text in self.chunk_text_by_words(native_text):
                    section_counters["text"] += 1
                    _add_chunk(
                        chunk_text,
                        {
                            "file_name": file_name,
                            "page_number": page_number,
                            "document_page_number": document_page_number,
                            "content_type": "text",
                            "page_type": page_type,
                            "chunk_id": f"{file_name}|page{page_number}|text|{section_counters['text']:03d}",
                        },
                    )

            include_ocr = page.get("ocr_text") and not page.get("native_text")
            if include_ocr:
                for chunk_text in self.chunk_text_by_words(page["ocr_text"]):
                    section_counters["ocr"] += 1
                    _add_chunk(
                        chunk_text,
                        {
                            "file_name": file_name,
                            "page_number": page_number,
                            "document_page_number": document_page_number,
                            "content_type": "ocr",
                            "page_type": page_type,
                            "chunk_id": f"{file_name}|page{page_number}|ocr|{section_counters['ocr']:03d}",
                        },
                    )

            for table_index, table in enumerate(page.get("tables", []), start=1):
                table_chunks = self.chunk_table_text(self.render_table_to_text(table), table)
                for part_index, chunk_text in enumerate(table_chunks, start=1):
                    _add_chunk(
                        chunk_text,
                        {
                            "file_name": file_name,
                            "page_number": page_number,
                            "document_page_number": document_page_number,
                            "content_type": "table",
                            "page_type": page_type,
                            "chunk_id": f"{file_name}|page{page_number}|table|{table_index:03d}.{part_index:02d}",
                        },
                    )

            for diagram in page.get("diagrams", []):
                section_counters["diagram"] += 1
                description = self.fix_mojibake(diagram.get("description", "")).strip()
                if not description:
                    continue
                _add_chunk(
                    description,
                    {
                        "file_name": file_name,
                        "page_number": page_number,
                        "document_page_number": document_page_number,
                        "content_type": "diagram",
                        "page_type": page_type,
                        "image_index": diagram.get("image_index"),
                        "diagram_description": description,
                        "diagram_ocr_text": self.fix_mojibake(diagram.get("ocr_text", "")).strip(),
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

        # If table is small, keep as one chunk
        if len(rows) <= 3:
            return [table_text]

        # Chunk by rows, keeping header in each
        chunks: list[str] = []
        header_text = f"Table: {table.get('title', 'Table')}\nColumns: {', '.join(header)}"
        rows_per_chunk = max(1, settings.pdf_chunk_size // 100)  # Adjust for table density
        for start in range(0, len(rows), rows_per_chunk):
            group = rows[start : start + rows_per_chunk]
            row_texts = []
            for row_index, row in enumerate(group, start=start + 1):
                if len(row) == len(header):
                    row_dict = {header[i]: str(row[i]).strip() for i in range(len(header))}
                    row_text = ", ".join([f"{k}: {v}" for k, v in row_dict.items()])
                else:
                    row_text = " | ".join([str(cell).strip() for cell in row])
                row_texts.append(f"Row {row_index}: {row_text}")
            chunks.append(f"{header_text}\n" + "\n".join(row_texts))
        return chunks
