"""
Text processing utilities including chunking and preprocessing.
"""

import re
from typing import Optional

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
