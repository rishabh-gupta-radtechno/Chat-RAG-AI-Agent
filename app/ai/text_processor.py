"""
Text processing utilities including chunking and preprocessing.
"""

import re
from difflib import SequenceMatcher
from typing import Dict, List, Optional

from app.core.config import get_settings
from app.core.logging import get_logger

settings = get_settings()
logger = get_logger(__name__)


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
    def clean_pdf_page_text(text: str, repeated_lines: Optional[set[str]] = None) -> str:
        """Normalize PDF page text while preserving document-specific content."""
        cleaned_lines = []
        boilerplate_patterns = [
            r"^Page\s+\d+$",
            r"^\d+\s*/\s*\d+$",
            r"^©\s*\d{4}.*$",
            r"^copyright\b.*$",
        ]

        for raw_line in text.splitlines():
            line = re.sub(r"\s+", " ", raw_line).strip()
            if not line:
                continue
            if repeated_lines and TextProcessor._line_fingerprint(line) in repeated_lines:
                logger.info("DEBUG: Removed repeated PDF boilerplate line: %r", line[:120])
                continue
            if any(re.search(pattern, line, flags=re.IGNORECASE) for pattern in boilerplate_patterns):
                continue
            cleaned_lines.append(line)

        return "\n".join(cleaned_lines).strip()

    @staticmethod
    def _line_fingerprint(line: str) -> str:
        normalized = re.sub(r"\b\d+\b", "#", line.lower())
        normalized = re.sub(r"[^a-z0-9#]+", " ", normalized)
        return re.sub(r"\s+", " ", normalized).strip()

    @staticmethod
    def _normalize_text(text: str) -> str:
        normalized = re.sub(r"[^a-z0-9\s]", " ", text.lower())
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    @staticmethod
    def _token_similarity(text_a: str, text_b: str) -> float:
        tokens_a = set(text_a.split())
        tokens_b = set(text_b.split())
        if not tokens_a or not tokens_b:
            return 0.0
        intersection = tokens_a.intersection(tokens_b)
        return len(intersection) / min(len(tokens_a), len(tokens_b))

    def _is_similar_text(self, text_a: str, text_b: str, threshold: float = 0.9) -> bool:
        normalized_a = self._normalize_text(text_a)
        normalized_b = self._normalize_text(text_b)
        if not normalized_a or not normalized_b:
            return False
        if normalized_a == normalized_b:
            return True
        return self._token_similarity(normalized_a, normalized_b) >= threshold

    @staticmethod
    def _is_duplicate_chunk_text(normalized_a: str, normalized_b: str) -> bool:
        if not normalized_a or not normalized_b:
            return False
        if normalized_a == normalized_b:
            return True
        shorter, longer = sorted([normalized_a, normalized_b], key=len)
        if len(shorter) >= 80 and shorter in longer and len(shorter) / max(len(longer), 1) > 0.92:
            return True
        return SequenceMatcher(None, normalized_a, normalized_b).ratio() >= 0.985

    @staticmethod
    def chunk_text_by_words(
        text: str,
        chunk_size: int = settings.pdf_chunk_size,
        overlap: int = settings.pdf_chunk_overlap,
    ) -> list[str]:
        """Split text into smaller page-local chunks by words.

        Preserve paragraph boundaries when possible and avoid splitting small
        pages into many fragments. If the text fits in one chunk, return it
        as a single chunk.
        """
        if not text:
            return []

        paragraphs = [
            paragraph.strip()
            for paragraph in re.split(r"\n\s*\n+", text)
            if paragraph.strip()
        ]

        if not paragraphs:
            return []

        chunks: list[str] = []
        current_chunk = ""

        for paragraph in paragraphs:
            paragraph_words = paragraph.split()
            if len(paragraph_words) <= chunk_size:
                candidate = f"{current_chunk}\n\n{paragraph}" if current_chunk else paragraph
                if len(candidate.split()) <= chunk_size:
                    current_chunk = candidate
                    continue
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = paragraph
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                step = max(1, chunk_size - overlap)
                for start in range(0, len(paragraph_words), step):
                    chunk = " ".join(paragraph_words[start : start + chunk_size])
                    if chunk.strip():
                        chunks.append(chunk)
        if current_chunk:
            chunks.append(current_chunk.strip())

        if len(chunks) == 0 and text.strip():
            return [text.strip()]

        return chunks

    def chunk_pdf_text_intelligently(
        self,
        text: str,
        page_number: int,
        content_type: str,
        min_chars: int = settings.pdf_chunk_min_chars,
        max_chars: int = settings.pdf_chunk_max_chars,
        overlap_chars: int = settings.pdf_chunk_overlap_chars,
    ) -> list[str]:
        """Split PDF text by engineering-manual structure with bounded character sizes."""
        normalized_text = self._normalize_pdf_text_layout(text)
        if not normalized_text:
            return []

        units = self._pdf_text_units(normalized_text)
        if not units:
            return []

        chunks: list[str] = []
        current = ""
        current_heading = ""

        for unit in units:
            unit_text = unit["text"]
            if not unit_text:
                continue

            if unit.get("kind") == "heading":
                current_heading = unit_text
                if current and len(current) >= min_chars:
                    chunks.append(current.strip())
                    current = ""
                if current and not current.endswith("\n"):
                    current += "\n"
                current += unit_text
                continue

            candidate = self._join_chunk_parts(current, unit_text)
            if len(candidate) <= max_chars:
                current = candidate
                continue

            if current:
                chunks.append(current.strip())

            prefix = current_heading if current_heading and current_heading not in unit_text else ""
            if len(unit_text) > max_chars:
                split_parts = self._split_long_text_unit(unit_text, max_chars=max_chars, overlap_chars=overlap_chars)
                for part in split_parts[:-1]:
                    chunk = self._join_chunk_parts(prefix, part) if prefix else part
                    chunks.append(chunk.strip())
                current = self._join_chunk_parts(prefix, split_parts[-1]) if split_parts else ""
            else:
                current = self._join_chunk_parts(prefix, unit_text) if prefix else unit_text

        if current.strip():
            chunks.append(current.strip())

        merged = self._merge_small_chunks(chunks, max_chars=max_chars)
        with_overlap = self._apply_text_overlap(merged, overlap_chars=overlap_chars, max_chars=max_chars)
        logger.info(
            "DEBUG: Intelligent chunking page=%s type=%s chunks=%s sizes=%s overlap=%s",
            page_number,
            content_type,
            len(with_overlap),
            [len(chunk) for chunk in with_overlap],
            overlap_chars,
        )
        return with_overlap

    @staticmethod
    def _normalize_pdf_text_layout(text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
        return "\n".join(lines).strip()

    def _pdf_text_units(self, text: str) -> list[dict]:
        units: list[dict] = []
        paragraph_lines: list[str] = []

        def flush_paragraph() -> None:
            if paragraph_lines:
                paragraph = " ".join(paragraph_lines).strip()
                if paragraph:
                    units.append({"kind": "paragraph", "text": paragraph})
                paragraph_lines.clear()

        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                flush_paragraph()
                continue

            if self._is_section_heading(stripped):
                flush_paragraph()
                units.append({"kind": "heading", "text": stripped})
                continue

            if self._is_procedure_step(stripped):
                flush_paragraph()
                units.append({"kind": "procedure", "text": stripped})
                continue

            paragraph_lines.append(stripped)

        flush_paragraph()
        return units

    @staticmethod
    def _is_section_heading(line: str) -> bool:
        if len(line) > 120 or len(line) < 3:
            return False
        if re.match(r"^\d+(\.\d+)*\s+[A-Z][A-Z0-9 /,&().:'-]{2,}$", line):
            return True
        if re.match(r"^(chapter|section|part|appendix)\s+\d+[\w .:-]*$", line, flags=re.IGNORECASE):
            return True
        letters = re.sub(r"[^A-Za-z]+", "", line)
        if len(letters) >= 4 and line.upper() == line and not re.search(r"[.!?]$", line):
            return True
        return False

    @staticmethod
    def _is_procedure_step(line: str) -> bool:
        return bool(
            re.match(r"^(\d+[\).]|\([a-zA-Z0-9]+\)|[a-zA-Z][\).]|step\s+\d+[:.)-])\s+", line, flags=re.IGNORECASE)
        )

    @staticmethod
    def _join_chunk_parts(first: str, second: str) -> str:
        if not first:
            return second.strip()
        if not second:
            return first.strip()
        separator = "\n" if TextProcessor._is_section_heading(second) else "\n\n"
        return f"{first.strip()}{separator}{second.strip()}"

    @staticmethod
    def _split_long_text_unit(text: str, max_chars: int, overlap_chars: int) -> list[str]:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        parts: list[str] = []
        current = ""

        for sentence in sentences:
            if not sentence:
                continue
            candidate = f"{current} {sentence}".strip() if current else sentence
            if len(candidate) <= max_chars:
                current = candidate
                continue
            if current:
                parts.append(current.strip())
            if len(sentence) > max_chars:
                start = 0
                step = max(1, max_chars - overlap_chars)
                while start < len(sentence):
                    parts.append(sentence[start : start + max_chars].strip())
                    start += step
                current = ""
            else:
                current = sentence

        if current:
            parts.append(current.strip())
        return [part for part in parts if part]

    @staticmethod
    def _merge_small_chunks(chunks: list[str], max_chars: int) -> list[str]:
        merged: list[str] = []
        current = ""
        for chunk in chunks:
            candidate = TextProcessor._join_chunk_parts(current, chunk) if current else chunk
            if current and len(candidate) > max_chars:
                merged.append(current.strip())
                current = chunk
            else:
                current = candidate
        if current:
            merged.append(current.strip())
        return merged

    @staticmethod
    def _apply_text_overlap(chunks: list[str], overlap_chars: int, max_chars: int) -> list[str]:
        if overlap_chars <= 0 or len(chunks) <= 1:
            return chunks

        overlapped = [chunks[0]]
        for previous, chunk in zip(chunks, chunks[1:]):
            overlap = TextProcessor._last_complete_overlap(previous, overlap_chars)
            candidate = f"{overlap}\n\n{chunk}".strip() if overlap else chunk
            if len(candidate) > max_chars + overlap_chars:
                candidate = candidate[-(max_chars + overlap_chars) :].strip()
            overlapped.append(candidate)
        return overlapped

    @staticmethod
    def _last_complete_overlap(text: str, overlap_chars: int) -> str:
        tail = text[-overlap_chars:].strip()
        if not tail:
            return ""
        match = re.search(r"\s", tail)
        if match:
            tail = tail[match.end() :].strip()
        return tail

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
    def render_table_to_text(table: dict) -> str:
        """Convert a structured table into a narrative text representation with better structure."""
        if table.get("table_text"):
            return str(table["table_text"]).strip()

        header = table.get("header", [])
        rows = table.get("rows", [])

        if header:
            text_lines = [" | ".join(str(cell).strip() for cell in header)]
        else:
            text_lines = []

        for row in rows:
            row_text = " | ".join([str(cell).strip() for cell in row])
            if row_text.strip():
                text_lines.append(row_text)

        return "\n".join(text_lines).strip()

    def _detect_repeated_pdf_boilerplate(self, page_documents: list[dict]) -> set[str]:
        """Find repeated top/bottom page lines that usually represent headers, footers, or logos."""
        occurrences: dict[str, set[int]] = {}
        total_pages = len(page_documents)
        if total_pages < 2:
            return set()

        for page in page_documents:
            page_number = int(page.get("page_number") or 0)
            text = "\n".join([page.get("native_text", ""), page.get("ocr_text", "")])
            lines = [
                re.sub(r"\s+", " ", line).strip()
                for line in text.splitlines()
                if re.sub(r"\s+", " ", line).strip()
            ]
            if len(lines) < 5:
                continue
            edge_lines = lines[:4] + lines[-4:]
            for line in edge_lines:
                if self._is_keepable_section_line(line):
                    continue
                fingerprint = self._line_fingerprint(line)
                if len(fingerprint) < 3:
                    continue
                occurrences.setdefault(fingerprint, set()).add(page_number)

        minimum_pages = max(2, int(total_pages * 0.35))
        repeated = {
            fingerprint
            for fingerprint, pages in occurrences.items()
            if len(pages) >= minimum_pages
        }
        logger.info(
            "DEBUG: Detected %s repeated PDF boilerplate lines across %s pages",
            len(repeated),
            total_pages,
        )
        return repeated

    @staticmethod
    def _is_keepable_section_line(line: str) -> bool:
        return bool(
            re.match(r"^\d+(\.\d+)*\s+[A-Z][A-Z0-9 /,&().:'-]{2,}$", line)
            or re.match(r"^(chapter|section|part|appendix)\s+\d+", line, flags=re.IGNORECASE)
        )

    def build_pdf_chunks(self, page_documents: list[dict], file_name: str) -> list[dict]:
        """Create metadata-rich chunks for a PDF with page-aware sections."""
        chunks: list[dict] = []
        seen_chunk_ids: set[str] = set()
        seen_chunk_texts: list[str] = []
        repeated_lines = self._detect_repeated_pdf_boilerplate(page_documents)

        def _add_chunk(text: str, metadata: dict) -> None:
            chunk_text = text.strip()
            if not chunk_text:
                return

            normalized_text = self._normalize_text(chunk_text)
            for existing_text in seen_chunk_texts:
                if self._is_duplicate_chunk_text(normalized_text, existing_text):
                    logger.info(
                        "DEBUG: Skipped duplicate chunk page=%s type=%s size=%s reason=text_similarity",
                        metadata.get("page_number"),
                        metadata.get("content_type"),
                        len(chunk_text),
                    )
                    return

            chunk_id = metadata.get("chunk_id")
            if not chunk_id:
                chunk_id = f"{file_name}|page{metadata.get('page_number', 0)}|{metadata.get('content_type', 'text')}|{len(chunks) + 1:03d}"
                metadata["chunk_id"] = chunk_id

            if chunk_id in seen_chunk_ids:
                logger.info(f"DEBUG: Skipped duplicate chunk id={chunk_id} reason=chunk_id_collision")
                return

            seen_chunk_ids.add(chunk_id)
            seen_chunk_texts.append(normalized_text)
            chunks.append(
                {
                    "text": chunk_text,
                    "metadata": metadata,
                }
            )
            logger.info(
                f"DEBUG: Created PDF chunk id={chunk_id} page={metadata.get('page_number')} "
                f"type={metadata.get('content_type')} len={len(chunk_text)} "
                f"overlap={settings.pdf_chunk_overlap_chars} "
                f"text={chunk_text[:200]!r}"
            )

        for page in page_documents:
            page_number = page.get("page_number", 0)
            document_page_number = self.extract_document_page_number(page.get("native_text", ""))
            section_counters = {
                "text": 0,
                "ocr": 0,
                "table": 0,
                "diagram": 0,
            }

            if page.get("native_text"):
                native_text = self.clean_pdf_page_text(page["native_text"], repeated_lines=repeated_lines)
                for chunk_text in self.chunk_pdf_text_intelligently(native_text, page_number, "text"):
                    section_counters["text"] += 1
                    _add_chunk(
                        chunk_text,
                        {
                            "file_name": file_name,
                            "page_number": page_number,
                            "document_page_number": document_page_number,
                            "content_type": "text",
                            "chunk_id": f"{file_name}|page{page_number}|text|{section_counters['text']:03d}",
                        },
                    )

            include_ocr = page.get("ocr_text") and not page.get("native_text")
            if include_ocr:
                ocr_text = self.clean_pdf_page_text(page["ocr_text"], repeated_lines=repeated_lines)
                for chunk_text in self.chunk_pdf_text_intelligently(ocr_text, page_number, "ocr"):
                    section_counters["ocr"] += 1
                    _add_chunk(
                        chunk_text,
                        {
                            "file_name": file_name,
                            "page_number": page_number,
                            "document_page_number": document_page_number,
                            "content_type": "ocr",
                            "chunk_id": f"{file_name}|page{page_number}|ocr|{section_counters['ocr']:03d}",
                        },
                    )

            for table_index, table in enumerate(page.get("tables", []), start=1):
                rendered_table_text = self.render_table_to_text(table)
                table_chunks = self.chunk_table_text(rendered_table_text, table)
                for part_index, chunk_text in enumerate(table_chunks, start=1):
                    _add_chunk(
                        chunk_text,
                        {
                            "file_name": file_name,
                            "page_number": page_number,
                            "document_page_number": document_page_number,
                            "content_type": "table",
                            "table_text": rendered_table_text,
                            "table_confidence": table.get("confidence"),
                            "table_source": table.get("source"),
                            "chunk_id": f"{file_name}|page{page_number}|table|{table_index:03d}.{part_index:02d}",
                        },
                    )

            for diagram in page.get("diagrams", []):
                section_counters["diagram"] += 1
                description = diagram.get("description", "").strip()
                if not description:
                    continue

                comparison_text = " ".join(
                    [page.get("native_text", ""), page.get("ocr_text", "")]
                ).strip()
                if comparison_text and self._is_similar_text(description, comparison_text, threshold=0.75):
                    logger.info(
                        f"DEBUG: Skipping diagram chunk on page {page_number} because description is similar to page text"
                    )
                    continue

                _add_chunk(
                    description,
                    {
                        "file_name": file_name,
                        "page_number": page_number,
                        "document_page_number": document_page_number,
                        "content_type": "diagram",
                        "image_index": diagram.get("image_index"),
                        "diagram_description": description,
                        "diagram_ocr_text": diagram.get("ocr_text", "").strip(),
                        "image_url": diagram.get("image_url"),
                        "chunk_id": f"{file_name}|page{page_number}|diagram|{section_counters['diagram']:03d}",
                    },
                )

        logger.info(
            "DEBUG: build_pdf_chunks completed file=%s chunk_count=%s chunk_sizes=%s",
            file_name,
            len(chunks),
            [len(chunk["text"]) for chunk in chunks],
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
            return self.chunk_pdf_text_intelligently(table_text, table.get("page_number", 0), "table")

        # If table is small, keep as one chunk
        if len(table_text) <= settings.pdf_chunk_max_chars:
            return [table_text]

        # Chunk by rows, keeping header in each
        chunks: list[str] = []
        header_text = " | ".join(str(cell).strip() for cell in header if str(cell).strip())
        current_lines = [header_text] if header_text else []

        for row in rows:
            row_text = " | ".join([str(cell).strip() for cell in row])
            if not row_text.strip():
                continue

            candidate_lines = current_lines + [row_text]
            candidate = "\n".join(candidate_lines).strip()
            if len(candidate) > settings.pdf_chunk_max_chars and len(current_lines) > (1 if header_text else 0):
                chunks.append("\n".join(current_lines).strip())
                current_lines = ([header_text] if header_text else []) + [row_text]
            else:
                current_lines = candidate_lines

        if current_lines:
            chunks.append("\n".join(current_lines).strip())

        logger.info(
            "DEBUG: Table chunking page=%s chunks=%s sizes=%s overlap=%s",
            table.get("page_number"),
            len(chunks),
            [len(chunk) for chunk in chunks],
            0,
        )
        return chunks
