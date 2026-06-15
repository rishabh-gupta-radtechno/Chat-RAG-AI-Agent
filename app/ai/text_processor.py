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

    @staticmethod
    def _table_title(table: dict, fallback: str = "Table") -> str:
        """Caption/heading to prepend to every chunk of this table."""
        return str(table.get("title") or table.get("caption") or fallback).strip() or fallback

    @staticmethod
    def table_to_markdown(header: list, rows: list) -> str:
        """Render a table as GitHub-flavored Markdown (header row repeated by caller)."""
        header = [str(h).strip() for h in (header or [])]
        width = len(header) or max((len(r) for r in rows), default=0)
        if not width:
            return ""

        def _fmt(cells: list) -> str:
            cells = [str(c).strip().replace("|", "\\|").replace("\n", " ") for c in cells]
            cells += [""] * (width - len(cells))
            return "| " + " | ".join(cells[:width]) + " |"

        lines = []
        if header:
            lines.append(_fmt(header))
            lines.append("| " + " | ".join(["---"] * width) + " |")
        lines.extend(_fmt(row) for row in rows)
        return "\n".join(lines)

    @staticmethod
    def _row_to_semantic(header: list, row: list) -> str:
        """One row as key=value pairs, e.g. ``Part=BC, Pressure=5 kg/cm²``."""
        cells = [str(c).strip() for c in row]
        header = [str(h).strip() for h in (header or [])]
        if header and len(cells) == len(header):
            return ", ".join(f"{h}={v}" for h, v in zip(header, cells) if v)
        return " | ".join(c for c in cells if c)

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

    # A numbered section heading, e.g. "3.1 Main Valve" or "1.2.0 CONSTRUCTION DETAILS".
    _SECTION_HEADING_RE = re.compile(r"^(\d{1,2}(?:\.\d{1,2}){0,3})\.?\s+([A-Za-z][^\n]{1,69})$")
    _HEADING_CONNECTORS = {
        "of", "and", "to", "the", "for", "in", "on", "or", "a", "an",
        "with", "from", "by", "at", "is", "as", "off",
    }

    @classmethod
    def _match_section_heading(cls, line: str) -> Optional[tuple]:
        """Return (number, title) when a line looks like a numbered section heading.

        Heading titles are Title-Case or ALL CAPS (e.g. "Main Valve", "OVERHAUL"),
        which lets us reject prose and OCR'd table rows like "8. Floor height".
        """
        match = cls._SECTION_HEADING_RE.match(line.strip())
        if not match:
            return None
        number = match.group(1)
        title = re.sub(r"\s*\([^)]*\)\s*$", "", match.group(2)).strip().rstrip(":").strip()
        words = [w for w in title.split() if any(c.isalpha() for c in w)]
        if not (1 <= len(words) <= 9):
            return None
        non_conforming = [
            w for w in words if not (w[:1].isupper() or w.lower() in cls._HEADING_CONNECTORS)
        ]
        if not (title.isupper() or not non_conforming):
            return None
        return number, title

    @staticmethod
    def _boilerplate_key(line: str) -> str:
        """Digit-insensitive key so page-varying running heads ('Sheet - 6/7/8') group."""
        return re.sub(r"\d+", "#", re.sub(r"\s+", " ", line).strip().lower())

    @classmethod
    def _detect_boilerplate_lines(cls, page_texts: list) -> set:
        """Short lines repeating across most pages are running headers/footers."""
        pages = [t for t in page_texts if t]
        n = len(pages)
        if n < 3:
            return set()
        from collections import Counter

        counts: "Counter" = Counter()
        for text in pages:
            seen = set()
            for raw in text.splitlines():
                line = re.sub(r"\s+", " ", raw).strip()
                if not line or len(line.split()) > 10:
                    continue
                key = cls._boilerplate_key(line)
                if key and key not in seen:
                    seen.add(key)
                    counts[key] += 1
        threshold = max(2, (n + 1) // 2)  # present on at least half the pages
        return {key for key, count in counts.items() if count >= threshold}

    @classmethod
    def _strip_boilerplate(cls, text: str, boilerplate_keys: set) -> str:
        """Drop running headers/footers and scanner watermarks before embedding."""
        if not text:
            return text
        kept = []
        for raw in text.splitlines():
            line = re.sub(r"\s+", " ", raw).strip()
            if not line:
                continue
            if re.search(r"scanned by camscanner", line, flags=re.IGNORECASE):
                continue
            if len(line.split()) <= 10 and cls._boilerplate_key(line) in boilerplate_keys:
                continue
            kept.append(line)
        return "\n".join(kept)

    def _split_into_sections(self, page_texts: list) -> list:
        """Split cleaned per-page text into heading-bounded segments, threading the
        current section across page breaks. Each segment keeps its start page."""
        segments: list = []
        current: Optional[dict] = None

        def _flush() -> None:
            if current and current["lines"]:
                current["text"] = "\n".join(current["lines"])
                segments.append(current)

        for page in page_texts:
            for raw in page["text"].splitlines():
                line = raw.strip()
                if not line:
                    continue
                heading = self._match_section_heading(line)
                if heading:
                    _flush()
                    number, title = heading
                    current = {
                        "section_number": number,
                        "section_title": title,
                        "page_number": page["page_number"],
                        "content_type": page["content_type"],
                        "page_type": page["page_type"],
                        "document_page_number": page["document_page_number"],
                        "lines": [line],
                    }
                else:
                    if current is None:
                        current = {
                            "section_number": None,
                            "section_title": None,
                            "page_number": page["page_number"],
                            "content_type": page["content_type"],
                            "page_type": page["page_type"],
                            "document_page_number": page["document_page_number"],
                            "lines": [],
                        }
                    current["lines"].append(line)
        _flush()
        return segments

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

        # Remove repeated running headers/footers across the whole document first.
        raw_page_texts = [
            (page.get("native_text") or page.get("ocr_text") or "") for page in page_documents
        ]
        boilerplate = self._detect_boilerplate_lines(raw_page_texts)

        # Cleaned, content-only per-page text — the source for section chunking.
        page_texts: list = []
        page_meta: dict = {}
        for page in page_documents:
            page_number = page.get("page_number", 0)
            is_native = bool(page.get("native_text"))
            raw = page.get("native_text") or page.get("ocr_text") or ""
            cleaned = self.clean_pdf_page_text(raw) if is_native else raw
            cleaned = self._strip_boilerplate(cleaned, boilerplate)
            page_type = self._detect_page_type(cleaned, page_number)
            document_page_number = self.extract_document_page_number(page.get("native_text", ""))
            page_meta[page_number] = {
                "page_type": page_type,
                "document_page_number": document_page_number,
            }
            if page_type in self._SKIP_PAGE_TYPES or not cleaned.strip():
                continue
            page_texts.append(
                {
                    "page_number": page_number,
                    "text": cleaned,
                    "content_type": "text" if is_native else "ocr",
                    "page_type": page_type,
                    "document_page_number": document_page_number,
                }
            )

        # Chunk text by section (heading-bounded, spanning pages); the section
        # title is stored inside the chunk text and in metadata.
        for seg_index, seg in enumerate(self._split_into_sections(page_texts), start=1):
            number, title = seg["section_number"], seg["section_title"]
            label = f"Section {number} {title}".strip() if (number or title) else ""
            for part_index, chunk_text in enumerate(self.chunk_text_by_words(seg["text"]), start=1):
                body = f"{label}\n{chunk_text}" if label else chunk_text
                _add_chunk(
                    body,
                    {
                        "file_name": file_name,
                        "page_number": seg["page_number"],
                        "document_page_number": seg["document_page_number"],
                        "content_type": seg["content_type"],
                        "page_type": seg["page_type"],
                        "section_number": number,
                        "section_title": title,
                        "chunk_id": f"{file_name}|sec{seg_index:03d}|{seg['content_type']}|{part_index:03d}",
                    },
                )

        # Tables and diagrams stay anchored to their page.
        for page in page_documents:
            page_number = page.get("page_number", 0)
            meta = page_meta.get(page_number, {"page_type": "content", "document_page_number": None})
            page_type = meta["page_type"]
            document_page_number = meta["document_page_number"]
            if page_type in self._SKIP_PAGE_TYPES:
                continue

            for table_index, table in enumerate(page.get("tables", []), start=1):
                table_title = self._table_title(table, fallback=f"Table {table_index}")
                table_markdown = self.table_to_markdown(table.get("header", []), table.get("rows", []))
                for part_index, chunk_text in enumerate(self.chunk_table_text(table), start=1):
                    _add_chunk(
                        chunk_text,
                        {
                            "file_name": file_name,
                            "page_number": page_number,
                            "document_page_number": document_page_number,
                            "content_type": "table",
                            "page_type": page_type,
                            "table_title": table_title,
                            "table_markdown": table_markdown,
                            "chunk_id": f"{file_name}|page{page_number}|table|{table_index:03d}.{part_index:02d}",
                        },
                    )

            diagram_index = 0
            for diagram in page.get("diagrams", []):
                description = self.fix_mojibake(diagram.get("description", "")).strip()
                if not description:
                    continue
                diagram_index += 1
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
                        "chunk_id": f"{file_name}|page{page_number}|diagram|{diagram_index:03d}",
                    },
                )

        return chunks

    @staticmethod
    def extract_document_page_number(text: str) -> Optional[int]:
        matches = re.findall(r"\bPage\s+(\d+)\b", text or "", flags=re.IGNORECASE)
        if not matches:
            return None
        return int(matches[-1])

    def chunk_table_text(self, table: dict) -> list[str]:
        """Chunk a structured table for retrieval.

        Each chunk carries: the table title (caption/heading), a Markdown block
        with the header repeated, and a semantic ``key=value`` line per row so
        BGE-M3 embeds each row as a self-contained fact. Large tables are split
        into row-batches (the header is repeated in every batch), so a long table
        never lands in a single chunk.
        """
        header = [str(h).strip() for h in table.get("header", [])]
        rows = table.get("rows", []) or []
        title = self._table_title(table)

        if not rows:
            md = self.table_to_markdown(header, [])
            return [f"Table: {title}\n\n{md}".strip()] if md else []

        rows_per_chunk = max(1, settings.table_rows_per_chunk)
        chunks: list[str] = []
        for start in range(0, len(rows), rows_per_chunk):
            group = rows[start : start + rows_per_chunk]
            markdown = self.table_to_markdown(header, group)  # header repeated per chunk
            semantic = "\n".join(
                line for line in (self._row_to_semantic(header, row) for row in group) if line
            )
            parts = [f"Table: {title}", markdown]
            if semantic:
                parts.append("Rows:\n" + semantic)
            chunks.append("\n\n".join(p for p in parts if p).strip())
        return chunks
