"""
Text processing utilities including chunking and preprocessing.
"""

import re
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
    def _token_set_ratio(a: str, b: str) -> float:
        """Fuzzy token-set similarity (0-100). Uses RapidFuzz when available."""
        try:
            from rapidfuzz.fuzz import token_set_ratio
            return float(token_set_ratio(a or "", b or ""))
        except Exception:
            ta = set(re.findall(r"[a-z0-9]+", (a or "").lower()))
            tb = set(re.findall(r"[a-z0-9]+", (b or "").lower()))
            if not ta or not tb:
                return 0.0
            # containment of the smaller token set ~ token_set_ratio behaviour
            return 100.0 * len(ta & tb) / min(len(ta), len(tb))

    @staticmethod
    def validate_table(table: dict, page_text: str = "") -> bool:
        """Return True only for a genuine structured table.

        Rejects TableFormer/camelot false positives where ordinary paragraph text
        was forced into a grid (e.g. "T | he changeover Valve ...").
        """
        header = [str(h).strip() for h in (table.get("header") or [])]
        rows = [[str(c).strip() for c in r] for r in (table.get("rows") or [])]

        # Rule 1 — structure: a table needs >= 2 columns and >= 2 rows.
        ncols = max([len(header)] + [len(r) for r in rows] or [0])
        nrows = (1 if any(header) else 0) + len(rows)
        if ncols < 2 or nrows < 2:
            return False

        cells = [c for c in header if c] + [c for row in rows for c in row if c]
        if not cells:
            return False

        # Rule 2 — split-word artifacts: standalone single letters ("T", "he"
        # from a split "The") betray paragraph text chopped into cells.
        single_char = sum(1 for c in cells if len(c) == 1 and c.isalpha())
        if single_char >= 2:
            return False

        # Rule 3 — paragraph dominance: a paragraph forced into a grid shows up as
        # cells that are mostly long sentences. But only a THIN grid (<=2 cols) is
        # plausibly that; a wide (>=3 col) table legitimately has paragraph cells
        # (e.g. a "component | reference | description" matrix), so we must not
        # reject those. Rule 2 still guards the split-word artifact case.
        word_counts = [len(c.split()) for c in cells]
        avg_words = sum(word_counts) / len(word_counts)
        long_cells = sum(1 for w in word_counts if w >= 10)
        if ncols <= 2 and (avg_words > 10 or long_cells > 0.5 * len(cells)):
            return False

        # Rule 4 — page-text similarity, but ONLY for prose-shaped tables.
        # A digital table's cells are ALSO in the page's text layer, so high token
        # overlap with the page is normal and must NOT be read as "reformatted
        # prose" — otherwise genuine tables get rejected whenever Docling didn't
        # carve them out of the prose (the Faiveley part-list bug). Restrict the
        # check to the case it was meant for: a thin (<=2 column), wordy grid that
        # is really a paragraph chopped into cells. Wide tables, and tables of
        # short discrete values, are unambiguously tabular and skip this rule.
        if page_text and ncols <= 2 and avg_words >= 3:
            table_text = " ".join(cells)
            if TextProcessor._token_set_ratio(page_text, table_text) >= settings.table_vs_text_similarity_threshold:
                return False

        return True

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
    # A line that is only a *dotted* section number (OCR wraps the title onto the
    # next line). Requiring a dot avoids mistaking OCR'd integer table-row numbers
    # ("5", "10") followed by an all-caps cell for a heading.
    _NUMBER_ONLY_RE = re.compile(r"^\d{1,2}(?:\.\d{1,2}){1,3}\.?$")
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
        """Split cleaned per-page text into heading-bounded, single-page segments.

        The current section carries across page breaks (so its number/title are
        kept), but a new page always starts a NEW segment tagged with that page,
        so a chunk never mixes content from two pages. The heading text itself is
        not duplicated into the body — build_pdf_chunks prepends the section label.
        """
        segments: list = []
        cur_num: Optional[str] = None
        cur_title: Optional[str] = None
        current: Optional[dict] = None

        def _new_segment(page: dict) -> dict:
            return {
                "section_number": cur_num,
                "section_title": cur_title,
                "page_number": page["page_number"],
                "content_type": page["content_type"],
                "page_type": page["page_type"],
                "document_page_number": page["document_page_number"],
                "lines": [],
            }

        def _flush() -> None:
            nonlocal current
            if current and current["lines"]:
                current["text"] = "\n".join(current["lines"])
                segments.append(current)
            current = None

        for page in page_texts:
            # A section continuing onto a new page becomes a fresh, page-tagged
            # segment — chunks stay within one page for accurate citations.
            _flush()
            current = _new_segment(page)
            lines = [ln.strip() for ln in page["text"].splitlines() if ln.strip()]
            i = 0
            while i < len(lines):
                line = lines[i]
                heading = self._match_section_heading(line)
                # OCR often wraps a heading onto two lines ("1.4" then the title
                # on the next line); join a number-only line with the following
                # line and re-test so those headings are still detected.
                if not heading and i + 1 < len(lines) and self._NUMBER_ONLY_RE.match(line):
                    combined = self._match_section_heading(f"{line.rstrip('.')} {lines[i + 1]}")
                    if combined:
                        heading = combined
                        i += 1  # consume the title line too
                if heading:
                    _flush()
                    cur_num, cur_title = heading
                    current = _new_segment(page)  # heading supplied by the chunk prefix
                else:
                    current["lines"].append(line)
                i += 1
        _flush()
        return segments

    # Higher = preferred to keep when two chunks are duplicates.
    _DEDUP_PRIORITY = {
        "text": 4, "table": 4, "chart_data": 4, "chart_summary": 4,
        "ocr": 3, "diagram": 1, "chart_image": 1,
    }

    @staticmethod
    def _dedup_norm(text: str) -> str:
        return re.sub(r"\s+", " ", (text or "")).strip().lower()

    @staticmethod
    def _dedup_tokens(text: str) -> set:
        """Alphanumeric word tokens (punctuation stripped) for overlap metrics."""
        return set(re.findall(r"[a-z0-9]+", (text or "").lower()))

    @staticmethod
    def _link_image_to_anchor(diagram: dict, anchors: list, anchor_tokens: list, diagram_tokens: set) -> None:
        """Link a duplicate diagram's image to the nearest text/table chunk.

        Picks the anchor whose tokens overlap the diagram most (preferring one on
        the same page) and appends the diagram's image to its ``related_images`` so
        the image is shown when that chunk is retrieved — without storing a second
        searchable vector for the duplicated OCR text.
        """
        if not anchors:
            return
        dmeta = diagram.get("metadata", {})
        if not dmeta.get("image_url"):
            return
        dpage = dmeta.get("page_number")
        best_index, best_score = -1, -1.0
        for index, (anchor, atokens) in enumerate(zip(anchors, anchor_tokens)):
            if not atokens:
                continue
            score = len(atokens & diagram_tokens) / len(atokens)
            if anchor.get("metadata", {}).get("page_number") == dpage:
                score += 1.0  # prefer a chunk on the same page
            if score > best_score:
                best_score, best_index = score, index
        if best_index < 0:
            return
        anchor_meta = anchors[best_index].setdefault("metadata", {})
        anchor_meta.setdefault("related_images", []).append(
            {
                "chunk_id": dmeta.get("chunk_id"),
                "image_url": dmeta.get("image_url"),
                "image_index": dmeta.get("image_index"),
                "page_number": dpage,
            }
        )

    def deduplicate_chunks(self, chunks: list[dict]) -> list[dict]:
        """Remove duplicate chunks before embedding (Stages 1 and 2 of dedup).

        Stage 1 — exact: drop chunks with identical normalized text, keeping the
        highest-priority content_type.
        Stage 2a — diagram suppression: drop a diagram chunk whose OCR text is
        already covered by the combined text/table chunks (containment, because a
        page-image diagram is a superset of the individual text chunks).
        Stage 2b — fuzzy: drop near-duplicate chunks (token Jaccard over the
        threshold), keeping the higher-priority content_type.
        """
        if not settings.dedup_enabled or not chunks:
            return chunks

        priority = self._DEDUP_PRIORITY
        before = len(chunks)

        def ct(chunk: dict) -> str:
            return chunk.get("metadata", {}).get("content_type", "text")

        # Stage 1: exact normalized-text dedup.
        seen: dict = {}
        ordered: list = []
        for chunk in chunks:
            key = self._dedup_norm(chunk.get("text"))
            if not key:
                continue
            current = seen.get(key)
            if current is None:
                seen[key] = chunk
                ordered.append(key)
            elif priority.get(ct(chunk), 0) > priority.get(ct(current), 0):
                seen[key] = chunk
        stage1 = [seen[k] for k in ordered]

        # Stage 2a: a diagram whose OCR text is already covered by text/table
        # chunks is NOT embedded as its own searchable vector. Its image is linked
        # to the nearest text/table chunk via related_images, so retrieval can
        # still show it ("text vector + linked image metadata" pattern).
        anchors = [c for c in stage1 if ct(c) in ("text", "table", "ocr")]
        anchor_tokens = [self._dedup_tokens(c["text"]) for c in anchors]
        text_union: set = set().union(*anchor_tokens) if anchor_tokens else set()
        stage2a: list = []
        for chunk in stage1:
            if ct(chunk) == "diagram" and text_union:
                tokens = self._dedup_tokens(chunk["text"])
                if tokens and len(tokens & text_union) / len(tokens) >= settings.dedup_diagram_suppression_threshold:
                    self._link_image_to_anchor(chunk, anchors, anchor_tokens, tokens)
                    continue  # image kept via the anchor's related_images
            stage2a.append(chunk)

        # Stage 2b: fuzzy near-duplicate (token Jaccard), keep higher priority.
        final: list = []
        final_tokens: list = []
        for chunk in stage2a:
            tokens = self._dedup_tokens(chunk["text"])
            dup_index = -1
            for index, existing in enumerate(final_tokens):
                if not tokens or not existing:
                    continue
                jaccard = len(tokens & existing) / len(tokens | existing)
                if jaccard >= settings.dedup_fuzzy_threshold:
                    dup_index = index
                    break
            if dup_index == -1:
                final.append(chunk)
                final_tokens.append(tokens)
            elif priority.get(ct(chunk), 0) > priority.get(ct(final[dup_index]), 0):
                final[dup_index] = chunk
                final_tokens[dup_index] = tokens

        if len(final) != before:
            logger.info("Deduplicated chunks: %d -> %d (text/diagram dedup)", before, len(final))
        return final

    @staticmethod
    def _format_chart_value(value) -> str:
        """Render a chart data point value as readable text.

        The vision model is asked for a single number/percentage per point, but
        if it returns a list (a whole series) or dict, flatten it gracefully so
        the chunk text never contains Python list repr like ['94', '57'].
        """
        if isinstance(value, (list, tuple)):
            return ", ".join(str(v) for v in value)
        if isinstance(value, dict):
            return ", ".join(f"{k}: {v}" for k, v in value.items())
        return str(value)

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
            # A page carrying a table, diagram, or chart is real content — never
            # let the short-text "cover" heuristic mislabel it (that is what made
            # the table-only page 2 disappear).
            if page.get("tables") or page.get("diagrams") or page.get("charts"):
                page_type = "content"
            document_page_number = self.extract_document_page_number(page.get("native_text", ""))
            page_meta[page_number] = {
                "page_type": page_type,
                "document_page_number": document_page_number,
            }
            # Pages are never dropped by type anymore (a fallback below guarantees
            # every page yields a chunk). Only skip the section loop when there is
            # no text to chunk — the page's tables/diagrams are still handled.
            if not cleaned.strip():
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

            page_prose = page.get("native_text") or page.get("ocr_text") or ""
            for table_index, table in enumerate(page.get("tables", []), start=1):
                # Reject false-positive tables (paragraph text forced into a grid,
                # or a reformatted copy of the page prose) before they pollute RAG.
                if not self.validate_table(table, page_prose):
                    logger.info(
                        "Rejected non-table on page %s of %s (failed validate_table)",
                        page_number,
                        file_name,
                    )
                    continue
                table_id = f"table_{page_number}_{table_index}"
                table_title = self._table_title(table, fallback=f"Table {table_index}")
                # Persist the structured grid (header + rows) alongside the markdown so
                # any source (Docling / pdfplumber / vision) lands as queryable JSON —
                # the table counterpart of charts' structured_data. The embedded TEXT
                # stays markdown + key=value (better for retrieval); JSON is metadata only.
                table_header = [str(h).strip() for h in table.get("header", [])]
                table_rows = [[str(c).strip() for c in row] for row in (table.get("rows") or [])]
                table_markdown = self.table_to_markdown(table_header, table_rows)
                for part_index, chunk_text in enumerate(self.chunk_table_text(table), start=1):
                    _add_chunk(
                        chunk_text,
                        {
                            "file_name": file_name,
                            "page_number": page_number,
                            "document_page_number": document_page_number,
                            "content_type": "table",
                            "page_type": page_type,
                            "table_id": table_id,
                            "table_title": table_title,
                            "table_markdown": table_markdown,
                            "table_header": table_header,
                            "table_rows": table_rows,
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

            # Charts: a vision model produced structured data + a summary. Emit
            # linked chunks — data (with structured_data), summary (embedded for
            # retrieval), and an image pointer — all sharing a related_chart id.
            for chart_index, chart in enumerate(page.get("charts", []), start=1):
                chart_id = f"chart_{page_number}_{chart_index}"
                chart_type = str(chart.get("chart_type") or "chart")
                chart_title = self.fix_mojibake(chart.get("title") or "").strip()
                structured = chart.get("structured_data") or []
                summary = self.fix_mojibake(chart.get("summary") or "").strip()
                image_url = chart.get("image_url")
                base = {
                    "file_name": file_name,
                    "page_number": page_number,
                    "document_page_number": document_page_number,
                    "page_type": page_type,
                    "related_chart": chart_id,
                    "chart_type": chart_type,
                    "chart_title": chart_title,
                    "image_url": image_url,
                }
                points = "; ".join(
                    f"{d.get('label', '')}={self._format_chart_value(d.get('value', ''))}"
                    for d in structured
                    if isinstance(d, dict) and d.get("label")
                )
                _add_chunk(
                    f"{chart_type} chart: {chart_title}\n{points}".strip(),
                    {**base, "content_type": "chart_data", "structured_data": structured,
                     "chunk_id": f"{file_name}|page{page_number}|chartdata|{chart_index:03d}"},
                )
                if summary:
                    _add_chunk(
                        summary,
                        {**base, "content_type": "chart_summary",
                         "chunk_id": f"{file_name}|page{page_number}|chartsummary|{chart_index:03d}"},
                    )
                if image_url:
                    _add_chunk(
                        f"Chart image: {chart_title or chart_type} (page {page_number}).",
                        {**base, "content_type": "chart_image",
                         "chunk_id": f"{file_name}|page{page_number}|chartimage|{chart_index:03d}"},
                    )

        # Guarantee every page is represented: if a page produced no chunk at all
        # (e.g. text was empty and its table was rejected), emit a fallback chunk
        # so no page silently disappears.
        pages_with_chunks = {c["metadata"].get("page_number") for c in chunks}
        for page in page_documents:
            page_number = page.get("page_number", 0)
            if page_number in pages_with_chunks:
                continue
            meta = page_meta.get(page_number, {"page_type": "content", "document_page_number": None})
            raw = (page.get("native_text") or page.get("ocr_text") or "").strip()
            fallback = self._strip_boilerplate(self.clean_pdf_page_text(raw), boilerplate).strip() if raw else ""
            fallback = fallback or f"Page {page_number} (no extractable text content)."
            _add_chunk(
                fallback,
                {
                    "file_name": file_name,
                    "page_number": page_number,
                    "document_page_number": meta["document_page_number"],
                    "content_type": "text",
                    "page_type": meta["page_type"],
                    "section_number": None,
                    "section_title": None,
                    "chunk_id": f"{file_name}|page{page_number}|fallback|001",
                },
            )
            logger.info("page=%s fallback chunk generated (no other content produced)", page_number)

        # Diagnostics: chunks generated per page.
        from collections import Counter
        per_page = Counter(c["metadata"].get("page_number") for c in chunks)
        logger.info(
            "build_pdf_chunks: %s chunks across %s pages | per-page=%s",
            len(chunks),
            len({p.get("page_number") for p in page_documents}),
            dict(sorted(per_page.items())),
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
