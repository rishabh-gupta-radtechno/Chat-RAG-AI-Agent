"""
Text processing utilities including chunking and preprocessing.
"""

import re
import math
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
        """Clean and normalize general text."""
        if not text:
            return ""
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^\w\s.,\-:;()!?/']", " ", text)
        return text.strip()

    @staticmethod
    def clean_ocr_text(text: str) -> str:
        """Fix #1: Clean OCR before embedding."""
        if not text:
            return ""

        text = re.sub(r"[\|_~^\\<>\[\]{}]", " ", text) # Remove OCR symbol artifacts
        text = re.sub(r"\s+", " ", text)
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

    @staticmethod
    def split_into_sentences(text: str) -> list[str]:
        """Splits text into individual sentences while preserving punctuation."""
        if not text:
            return []
        return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

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
    @staticmethod
    def _is_low_quality(text: str) -> bool:
        """Filter out chunks that are mostly OCR noise or gibberish."""
        if not text or len(text) < 20:
            return True
        alpha_chars = sum(c.isalpha() for c in text)
        return (alpha_chars / len(text)) < 0.4

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
        """Create metadata-rich chunks for a PDF with page-aware sections and title boosting."""
        chunks: list[dict] = []
        seen_chunk_ids: set[str] = set()

        def _add_chunk(text: str, metadata: dict) -> None:
            chunk_text = text.strip()
            if not chunk_text:
                return

            chunk_id = metadata.get("chunk_id")
            if not chunk_id:
                chunk_id = f"{file_name}|page{metadata.get('page_number', 0)}|{metadata.get('content_type', 'text')}|{len(chunks) + 1:03d}"
                metadata["chunk_id"] = chunk_id

            if chunk_id in seen_chunk_ids:
                return

            # Filter Low-Quality OCR Chunks (already implemented in _is_low_quality)
            # This ensures chunks with mostly noise or gibberish are skipped.
            # Filter out noise and low-alpha OCR fragments
            if self._is_low_quality(chunk_text):
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
            
            # Fix: Merge OCR and Native (Problem #1 from review)
            native = self.clean_pdf_page_text(page.get("native_text") or "")
            ocr = self.clean_ocr_text(page.get("ocr_text") or "")
            
            combined_text = f"{native}\n{ocr}".strip()
            page_type = self._detect_page_type(combined_text, page_number)
            
            document_page_number = self.extract_document_page_number(combined_text)

            if page_type in self._SKIP_PAGE_TYPES:
                continue

            document_page_number = self.extract_document_page_number(combined_text)
            section_counters = {
                "content": 0,
                "table": 0,
                "diagram": 0,
            }

            # Refactor: Chunk by headings first
            section_chunks = self._split_page_into_sections(combined_text, file_name, page_number, document_page_number, page_type)
            for chunk_data in section_chunks:
                _add_chunk(chunk_data["text"], chunk_data["metadata"])

            # Process tables (keep existing logic)
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

            # Process diagrams (keep existing logic)
            for diagram in page.get("diagrams", []):
                section_counters["diagram"] += 1
                description = diagram.get("description", "").strip()
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
                        "diagram_ocr_text": diagram.get("ocr_text", "").strip(),
                        "image_url": diagram.get("image_url"),
                        "chunk_id": f"{file_name}|page{page_number}|diagram|{section_counters['diagram']:03d}",
                    },
                )

        return chunks

    def _split_page_into_sections(self, page_text: str, file_name: str, page_number: int, document_page_number: Optional[int], page_type: str) -> list[dict]:
        """Splits page text into sections based on headings and then chunks them."""
        sections_data = []
        current_section_lines = []
        current_section_title = None
        current_section_number = None
        
        lines = page_text.splitlines()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            section_number, section_title = self._extract_section_title_and_number(line)
            
            if section_title: # Found a new heading
                # If we have accumulated text for a previous section, process it
                if current_section_lines:
                    sections_data.extend(self._chunk_section_text(
                        "\n".join(current_section_lines),
                        file_name,
                        page_number,
                        document_page_number,
                        page_type,
                        current_section_title,
                        current_section_number
                    ))
                
                # Start new section
                current_section_title = section_title
                current_section_number = section_number
                current_section_lines = [line] # Include the heading in its own section
            else:
                current_section_lines.append(line)
                
        # Add the last accumulated section
        if current_section_lines:
            sections_data.extend(self._chunk_section_text(
                "\n".join(current_section_lines),
                file_name,
                page_number,
                document_page_number,
                page_type,
                current_section_title,
                current_section_number
            ))
            
        return sections_data

    def _chunk_section_text(self, text: str, file_name: str, page_number: int, document_page_number: Optional[int], page_type: str, section_title: Optional[str], section_number: Optional[str]) -> list[dict]:
        """Subdivides a section's text into smaller chunks with overlap."""
        chunks = []
        # Use settings.chunk_size (800) and settings.chunk_overlap (150) for subdivision
        # These are character-based as per the user's suggestion.
        
        # Simple character-based splitter
        for i in range(0, len(text), settings.chunk_size - settings.chunk_overlap):
            chunk_text = text[i : i + settings.chunk_size]
            if chunk_text.strip():
                chunks.append({
                    "text": chunk_text,
                    "metadata": {
                        "file_name": file_name,
                        "page_number": page_number,
                        "document_page_number": document_page_number,
                        "content_type": "text_section", # Differentiate from generic 'text'
                        "page_type": page_type,
                        "section_title": section_title,
                        "section_number": section_number,
                    }
                })
        return chunks

    @staticmethod
    def _extract_section_title(text: str) -> Optional[str]:
        """Extract the first heading from text for metadata boosting.
        
        Looks for:
        - Numbered section headings (1.2.3 TITLE)
        - ALL CAPS headings
        - Lines starting with patterns like "CHAPTER", "SECTION"
        """
        if not text:
            return None
        
        lines = text.strip().split('\n')
        for line in lines[:5]:  # Check first 5 lines
            line = line.strip()
            if not line:
                continue
            
            # Numbered heading (e.g., "2.3 REASSEMBLY OF VTA VALVE")
            numbered_match = re.match(r'^(\d+(?:\.\d+)*)\s+(.+)$', line)
            if numbered_match:
                return numbered_match.group(2).strip()
            
            # ALL CAPS heading
            if line.isupper() and len(line.split()) >= 2:
                return line
            
            # Section keywords
            if re.match(r'^(CHAPTER|SECTION|PROCEDURE|STEP|ASSEMBLY|DISASSEMBLY|REASSEMBLY|OVERHAUL)\b', line, re.IGNORECASE):
                return line
        
        return None

    @staticmethod
    def _extract_section_title_and_number(line: str) -> tuple[Optional[str], Optional[str]]:
        """Extracts section number and title from a line if it's a heading."""
        line = line.strip()
        if not line:
            return None, None

        # Pattern for numbered headings (e.g., "1.0 INTRODUCTION", "3.2.1 SUB-SECTION TITLE")
        numbered_match = re.match(r'^((\d+(\.\d+)*)\s+)?([A-Z0-9\s\-\/&().,]{5,})$', line)
        if numbered_match:
            section_number = numbered_match.group(2)
            section_title = numbered_match.group(4).strip()
            # Heuristic: if it's all caps or starts with a number and is relatively short, it's likely a heading
            if line.isupper() or (section_number and len(line.split()) < 10):
                return section_number, section_title
        
        # Pattern for unnumbered ALL CAPS headings (e.g., "INTRODUCTION", "GENERAL INFORMATION")
        if line.isupper() and len(line.split()) >= 2 and len(line) > 5: # At least 2 words and reasonable length
            return None, line

        return None, None

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

    @staticmethod
    def _important_terms(text: str) -> list[str]:
        stop_words = {
            "a", "an", "and", "are", "can", "could", "for", "how", "i",
            "in", "is", "of", "on", "or", "please", "should", "the",
            "to", "what", "when", "where", "which", "with", "you",
            "about", "describe", "explain", "give", "tell", "why",
        }
        return [
            term
            for term in re.findall(r"[a-z0-9]+", text.lower())
            if len(term) > 2 and term not in stop_words
        ]
