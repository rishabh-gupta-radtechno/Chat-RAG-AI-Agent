"""
PDF extraction pipeline for native text, images, OCR, tables, and diagram descriptions.
"""
import io
import re
import threading
from html.parser import HTMLParser
from numbers import Real
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()
_PADDLE_OCR_INSTANCE = None
_PADDLE_OCR_LOCK = threading.Lock()
_PADDLE_OCR_UNAVAILABLE = False
_PPSTRUCTURE_INSTANCE = None
_PPSTRUCTURE_LOCK = threading.Lock()
_PPSTRUCTURE_UNAVAILABLE = False


class _TableHTMLParser(HTMLParser):
    """Small stdlib parser for PPStructure table HTML."""

    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self._current_row: list[str] = []
        self._current_cell: list[str] = []
        self._in_cell = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        if tag == "tr":
            self._current_row = []
        elif tag in {"td", "th"}:
            self._current_cell = []
            self._in_cell = True

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            self._current_cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self._in_cell:
            cell = re.sub(r"\s+", " ", " ".join(self._current_cell)).strip()
            self._current_row.append(cell)
            self._in_cell = False
        elif tag == "tr" and self._current_row:
            self.rows.append(self._current_row)


class PDFProcessor:
    """Extract page-level multimodal data from PDF documents."""

    def __init__(self) -> None:
        self._ocr_engine = settings.ocr_engine or "paddleocr"
        self._diagram_captioning_enabled = settings.enable_diagram_captioning
        self._caption_model = None
        self._caption_processor = None
        self._paddle_ocr = None  # Singleton instance to prevent PDX reinitialization

    def extract_page_documents(self, filepath: str, file_id: Optional[str] = None) -> list[Dict[str, Any]]:
        """Extract native text, OCR text, tables, and diagrams for each PDF page."""
        logger.info(f"DEBUG: Starting PDF extraction for file={filepath} file_id={file_id}")
        logger.info(f"DEBUG: use_docling={settings.use_docling} ocr_full_page={settings.ocr_full_page}")
        # Try Docling first for better structured extraction if enabled
        if settings.use_docling:
            try:
                pages = self._extract_with_docling(filepath, file_id=file_id)
                logger.info(f"DEBUG: Docling extracted {len(pages)} pages")
                return pages
            except Exception as exc:
                logger.warning("Docling extraction failed: %s. Falling back to traditional extraction.", exc)
                logger.info(f"DEBUG: Docling failure details: {exc}")
        
        # Fallback to traditional extraction
        try:
            import pypdf
        except ImportError as exc:  # pragma: no cover
            raise ImportError("pypdf is required for PDF processing") from exc

        native_pages = self._extract_native_text_pages(filepath)

        images_by_page = self._extract_page_images(filepath)
        rendered_pages = self._render_pages_for_ocr(filepath, native_pages)

        pages: list[Dict[str, Any]] = []
        for page_number, native_text in enumerate(native_pages, start=1):
            page_images = images_by_page.get(page_number, [])
            ocr_inputs = []
            rendered_page = rendered_pages.get(page_number)
            if rendered_page:
                ocr_inputs.extend(rendered_page)
            elif not native_text.strip():
                ocr_inputs.extend(page_images)

            ocr_text = self._extract_ocr_text(ocr_inputs)
            tables = self._extract_tables(filepath, page_number)
            diagrams = self._extract_diagrams(
                filepath=filepath,
                file_id=file_id,
                page_number=page_number,
                native_text=native_text,
                ocr_text=ocr_text,
            )

            pages.append(
                {
                    "page_number": page_number,
                    "native_text": native_text.strip(),
                    "ocr_text": ocr_text.strip(),
                    "tables": tables,
                    "diagrams": diagrams,
                }
            )

        logger.info(f"DEBUG: extract_page_documents built {len(pages)} pages")
        for page in pages:
            logger.info(
                f"DEBUG: page={page['page_number']} native_text_len={len(page.get('native_text',''))} "
                f"ocr_text_len={len(page.get('ocr_text',''))} tables={len(page.get('tables',[]))} "
                f"diagrams={len(page.get('diagrams',[]))}"
            )
        return pages

    def _extract_with_docling(self, filepath: str, file_id: Optional[str] = None) -> list[Dict[str, Any]]:
        """Extract document content using Docling for better structured parsing.
        
        Properly handles:
        - Text blocks (paragraphs, headings)
        - Tables with structure preservation
        - Figures/images with captions
        - Per-page error handling with fallback
        """
        try:
            from docling.document_converter import DocumentConverter
            from docling.document_converter import PdfFormatOption
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
        except ImportError as exc:
            raise ImportError("docling is required for advanced PDF processing") from exc

        logger.info(f"DEBUG: Loading PDF with Docling: {filepath}")

        # Configure Docling pipeline options
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False  # We handle OCR separately
        pipeline_options.do_table_structure = True  # Extract table structure

        doc_converter = DocumentConverter(
            allowed_formats=[InputFormat.PDF],
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                    backend=PyPdfiumDocumentBackend,
                )
            },
        )

        try:
            result = doc_converter.convert(filepath)
            doc = result.document
            logger.info(f"DEBUG: Successfully loaded PDF with Docling. Total pages: {len(doc.pages)}")
        except Exception as exc:
            logger.error(f"Docling document conversion failed: {exc}")
            raise

        images_by_page = self._extract_page_images(filepath)
        pages: list[Dict[str, Any]] = []

        # Process each page
        for page_idx, page_obj in enumerate(doc.pages):
            page_number = page_idx + 1
            logger.info(f"DEBUG: Processing page {page_number} with Docling")

            try:
                native_text = ""
                tables: list[Dict[str, Any]] = []
                headings: list[str] = []
                paragraphs: list[str] = []

                # Extract blocks from the page
                if not hasattr(page_obj, "blocks"):
                    logger.warning(f"DEBUG: Page {page_number} has no blocks attribute. Skipping block extraction.")
                else:
                    total_blocks = len(page_obj.blocks)
                    logger.info(f"DEBUG: Page {page_number} has {total_blocks} blocks")

                    for block_idx, block in enumerate(page_obj.blocks):
                        block_type = type(block).__name__
                        logger.info(f"DEBUG: Page {page_number} block {block_idx + 1}/{total_blocks}: type={block_type}")

                        # Extract text blocks (paragraphs, headings, etc.)
                        if hasattr(block, "text"):
                            block_text = block.text.strip() if isinstance(block.text, str) else ""
                            if block_text:
                                # Try to detect if this is a heading based on block type name
                                if "Heading" in block_type or (hasattr(block, "level") and block.level and block.level < 3):
                                    headings.append(block_text)
                                    logger.info(
                                        f"DEBUG: Page {page_number} heading block {block_idx + 1}: {block_text[:100]!r}"
                                    )
                                else:
                                    paragraphs.append(block_text)
                                    logger.info(
                                        f"DEBUG: Page {page_number} paragraph block {block_idx + 1}: {block_text[:100]!r}"
                                    )

                        # Extract tables
                        if "Table" in block_type:
                            try:
                                table_data = self._extract_docling_table(block)
                                if table_data:
                                    tables.append(table_data)
                                    logger.info(
                                        f"DEBUG: Page {page_number} extracted table block {block_idx + 1}: "
                                        f"header={len(table_data.get('header', []))} cols, "
                                        f"rows={len(table_data.get('rows', []))}"
                                    )
                            except Exception as exc:
                                logger.warning(
                                    f"DEBUG: Failed to extract table from block {block_idx + 1} on page {page_number}: {exc}"
                                )

                        # Extract figures/images
                        if "Figure" in block_type or "Image" in block_type:
                            try:
                                figure_data = self._extract_docling_figure(block)
                                if figure_data:
                                    logger.info(
                                        f"DEBUG: Page {page_number} extracted figure block {block_idx + 1}: {figure_data.get('description', '')[:100]!r}"
                                    )
                            except Exception as exc:
                                logger.warning(
                                    f"DEBUG: Failed to extract figure from block {block_idx + 1} on page {page_number}: {exc}"
                                )

                # Combine text from headings and paragraphs
                native_text = "\n\n".join(headings + paragraphs).strip()
                logger.info(
                    f"DEBUG: Page {page_number} extraction complete: "
                    f"native_text_len={len(native_text)} headings={len(headings)} "
                    f"paragraphs={len(paragraphs)} tables={len(tables)}"
                )

                # Extract page images for diagram processing
                diagrams = self._extract_diagrams(
                    filepath=filepath,
                    file_id=file_id,
                    page_number=page_number,
                    native_text=native_text,
                    ocr_text="",
                )

                pages.append(
                    {
                        "page_number": page_number,
                        "native_text": native_text,
                        "ocr_text": "",
                        "tables": tables,
                        "diagrams": diagrams,
                    }
                )

            except Exception as exc:
                logger.warning(f"Docling extraction failed for page {page_number}: {exc}. Using fallback extraction.")
                pages.append(
                    {
                        "page_number": page_number,
                        "native_text": "",
                        "ocr_text": "",
                        "tables": [],
                        "diagrams": [],
                    }
                )

        # OCR pass for low-text pages
        native_pages = [page.get("native_text", "") for page in pages]
        rendered_pages = self._render_pages_for_ocr(filepath, native_pages)
        for page in pages:
            page_number = page["page_number"]
            page_images = images_by_page.get(page_number, [])
            ocr_inputs = []
            rendered_page = rendered_pages.get(page_number)
            if rendered_page:
                ocr_inputs.extend(rendered_page)
            elif not page["native_text"].strip():
                ocr_inputs.extend(page_images)
            page["ocr_text"] = self._extract_ocr_text(ocr_inputs).strip()

        logger.info(f"DEBUG: Docling extraction complete. Total pages: {len(pages)}")
        return pages

    def _extract_docling_table(self, table_block: Any) -> Optional[Dict[str, Any]]:
        """Extract table data from Docling TableBlock.
        
        Args:
            table_block: A Docling TableBlock object
            
        Returns:
            Dictionary with 'header', 'rows', and 'title' keys, or None if extraction fails
        """
        try:
            data = None
            if hasattr(table_block, "data"):
                data = table_block.data
            elif hasattr(table_block, "table") and hasattr(table_block.table, "data"):
                data = table_block.table.data
            
            if not data or len(data) == 0:
                logger.debug("Table block has no extractable data")
                return None
            
            # First row is header, rest are data rows
            header = [str(cell).strip() for cell in data[0]] if len(data) > 0 else []
            rows = [[str(cell).strip() for cell in row] for row in data[1:]] if len(data) > 1 else []
            
            if not header and not rows:
                return None
            
            return {
                "title": "Extracted Table",
                "header": header,
                "rows": rows,
            }
        except Exception as exc:
            logger.warning(f"Failed to extract Docling table data: {exc}")
            return None

    def _extract_docling_figure(self, figure_block: Any) -> Optional[Dict[str, Any]]:
        """Extract figure/image data from Docling FigureBlock.
        
        Args:
            figure_block: A Docling FigureBlock object
            
        Returns:
            Dictionary with 'description' and 'ocr_text' keys, or None if extraction fails
        """
        try:
            caption = ""
            if hasattr(figure_block, "caption"):
                caption_obj = figure_block.caption
                if hasattr(caption_obj, "text"):
                    caption = caption_obj.text.strip() if isinstance(caption_obj.text, str) else ""
            
            if not caption:
                caption = "A figure or diagram"
            
            return {
                "description": caption,
                "ocr_text": "",
            }
        except Exception as exc:
            logger.warning(f"Failed to extract Docling figure data: {exc}")
            return None

    def _extract_native_text_pages(self, filepath: str) -> list[str]:
        """Extract native text with pypdf, then fill weak pages with pdfplumber text."""
        import pypdf

        native_pages: list[str] = []
        with open(filepath, "rb") as f:
            reader = pypdf.PdfReader(f)
            for page in reader.pages:
                native_pages.append(page.extract_text() or "")

        try:
            import pdfplumber
        except ImportError:
            return native_pages

        try:
            with pdfplumber.open(filepath) as pdf:
                for page_index, page in enumerate(pdf.pages):
                    plumber_text = page.extract_text(x_tolerance=1, y_tolerance=3) or ""
                    if page_index >= len(native_pages):
                        native_pages.append(plumber_text)
                        continue
                    current = native_pages[page_index] or ""
                    if len(plumber_text.strip()) > len(current.strip()) * 1.1:
                        native_pages[page_index] = plumber_text
        except Exception as exc:
            logger.warning("pdfplumber text extraction failed: %s", exc)

        logger.info(f"DEBUG: _extract_native_text_pages returned {len(native_pages)} pages")
        for index, text in enumerate(native_pages, start=1):
            logger.info(f"DEBUG: native page {index} len={len(text)}")
        return native_pages

    def _render_pages_for_ocr(self, filepath: str, native_pages: list[str]) -> Dict[int, List[Dict[str, Any]]]:
        """Render low-text pages so scanned PDFs can be OCR'd."""
        if not settings.ocr_full_page:
            return {}

        try:
            import fitz
        except ImportError:
            logger.warning("PyMuPDF is not installed: full-page OCR will be disabled.")
            return {}

        rendered_pages: Dict[int, List[Dict[str, Any]]] = {}
        min_text_chars = settings.ocr_full_page_min_text_chars
        zoom = settings.ocr_full_page_dpi / 72
        matrix = fitz.Matrix(zoom, zoom)

        try:
            with fitz.open(filepath) as document:
                for page_index, page in enumerate(document):
                    native_text = native_pages[page_index] if page_index < len(native_pages) else ""
                    if len(native_text.strip()) >= min_text_chars:
                        continue

                    pix = page.get_pixmap(matrix=matrix, alpha=False)
                    rendered_pages[page_index + 1] = [
                        {
                            "image_index": "page",
                            "bytes": pix.tobytes("png"),
                            "source": "full_page",
                        }
                    ]
        except Exception as exc:
            logger.warning("Unable to render pages for OCR: %s", exc)

        return rendered_pages

    def _extract_page_images(self, filepath: str) -> Dict[int, List[Dict[str, Any]]]:
        try:
            import fitz
        except ImportError:
            logger.warning("PyMuPDF is not installed: image extraction will be disabled.")
            return {}

        images_by_page: Dict[int, List[Dict[str, Any]]] = {}
        try:
            with fitz.open(filepath) as document:
                for page_index in range(len(document)):
                    page = document[page_index]
                    page_images: List[Dict[str, Any]] = []
                    for image_index, image_info in enumerate(page.get_images(full=True)):
                        xref = image_info[0]
                        try:
                            pix = fitz.Pixmap(document, xref)
                            if pix.n > 4:
                                pix = fitz.Pixmap(fitz.csRGB, pix)
                            image_bytes = pix.tobytes("png")
                            page_images.append({
                                "image_index": image_index,
                                "bytes": image_bytes,
                                "width": pix.width,
                                "height": pix.height,
                                "page_width": page.rect.width,
                                "page_height": page.rect.height,
                            })
                        except Exception as exc:
                            logger.warning(
                                "Failed to extract image page=%s image=%s: %s",
                                page_index + 1,
                                image_index,
                                exc,
                            )
                    if page_images:
                        images_by_page[page_index + 1] = page_images
        except Exception as exc:
            logger.warning("Unable to extract images from PDF: %s", exc)

        page_counts = {page: len(images) for page, images in images_by_page.items()}
        logger.info(f"DEBUG: _extract_page_images extracted images_by_page={page_counts}")
        return images_by_page

    def _extract_ocr_text(self, images: List[Dict[str, Any]]) -> str:
        if not images:
            return ""

        ocr_texts: list[str] = []
        for image in images:
            image_bytes = image.get("bytes")
            if not image_bytes:
                continue
            text = self._run_ocr(image_bytes)
            if text:
                ocr_texts.append(text)

        result = "\n".join(ocr_texts)
        logger.info(f"DEBUG: _extract_ocr_text returned len={len(result)} text={result[:250]!r}")
        return result

    def _run_ocr(self, image_bytes: bytes) -> str:
        text = ""

        if self._ocr_engine == "paddleocr":
            text = self._run_paddleocr(image_bytes)
            if text:
                return text

        text = self._run_tesseract(image_bytes)
        if not text:
            logger.debug("No OCR engines available or OCR processing returned no text. Proceeding without OCR.")
        return text

    def _run_paddleocr(self, image_bytes: bytes) -> str:
        global _PADDLE_OCR_INSTANCE, _PADDLE_OCR_UNAVAILABLE

        if _PADDLE_OCR_UNAVAILABLE:
            return ""

        try:
            from paddleocr import PaddleOCR
        except ImportError:
            logger.warning("PaddleOCR not installed, skipping PaddleOCR path.")
            _PADDLE_OCR_UNAVAILABLE = True
            return ""
        except Exception as exc:
            logger.warning("PaddleOCR import failed: %s. Falling back to Tesseract.", exc)
            _PADDLE_OCR_UNAVAILABLE = True
            self._ocr_engine = "tesseract"
            return ""

        if self._paddle_ocr is None:
            with _PADDLE_OCR_LOCK:
                if _PADDLE_OCR_INSTANCE is not None:
                    self._paddle_ocr = _PADDLE_OCR_INSTANCE
                elif not _PADDLE_OCR_UNAVAILABLE:
                    try:
                        _PADDLE_OCR_INSTANCE = PaddleOCR(
                            use_angle_cls=True,
                            lang="en",
                            use_gpu=False,
                            use_space_char=True,
                        )
                        self._paddle_ocr = _PADDLE_OCR_INSTANCE
                    except Exception as exc:
                        logger.warning("PaddleOCR initialization failed: %s. Falling back to Tesseract.", exc)
                        _PADDLE_OCR_UNAVAILABLE = True
                        self._ocr_engine = "tesseract"
                        return ""

        if self._paddle_ocr is None:
            return ""

        image = self._prepare_image_for_ocr(image_bytes)
        try:
            results = self._paddle_ocr.ocr(image, cls=True)
        except Exception as exc:
            logger.warning("PaddleOCR inference failed: %s. Falling back to Tesseract.", exc)
            if "PDX has already been initialized" in str(exc):
                _PADDLE_OCR_UNAVAILABLE = True
            self._ocr_engine = "tesseract"
            return ""

        return "\n".join(self._collect_paddle_text(results))

    def _run_paddleocr_with_density(self, image_bytes: bytes) -> tuple[str, float]:
        """Run PaddleOCR and estimate how much of the crop is covered by OCR text boxes."""
        global _PADDLE_OCR_INSTANCE, _PADDLE_OCR_UNAVAILABLE

        if _PADDLE_OCR_UNAVAILABLE:
            return self._run_ocr(image_bytes), 0.0

        try:
            from paddleocr import PaddleOCR
        except ImportError:
            logger.warning("PaddleOCR not installed, skipping OCR density calculation.")
            _PADDLE_OCR_UNAVAILABLE = True
            return self._run_ocr(image_bytes), 0.0
        except Exception as exc:
            logger.warning("PaddleOCR import failed during OCR density calculation: %s", exc)
            _PADDLE_OCR_UNAVAILABLE = True
            return self._run_ocr(image_bytes), 0.0

        if self._paddle_ocr is None:
            with _PADDLE_OCR_LOCK:
                if _PADDLE_OCR_INSTANCE is not None:
                    self._paddle_ocr = _PADDLE_OCR_INSTANCE
                elif not _PADDLE_OCR_UNAVAILABLE:
                    try:
                        _PADDLE_OCR_INSTANCE = PaddleOCR(
                            use_angle_cls=True,
                            lang="en",
                            use_gpu=False,
                            use_space_char=True,
                        )
                        self._paddle_ocr = _PADDLE_OCR_INSTANCE
                    except Exception as exc:
                        logger.warning("PaddleOCR initialization failed during OCR density calculation: %s", exc)
                        _PADDLE_OCR_UNAVAILABLE = True
                        return self._run_ocr(image_bytes), 0.0

        if self._paddle_ocr is None:
            return self._run_ocr(image_bytes), 0.0

        image = self._prepare_image_for_ocr(image_bytes)
        image_width = getattr(image, "width", 0) or 0
        image_height = getattr(image, "height", 0) or 0
        try:
            results = self._paddle_ocr.ocr(image, cls=True)
        except Exception as exc:
            logger.warning("PaddleOCR density inference failed: %s", exc)
            if "PDX has already been initialized" in str(exc):
                _PADDLE_OCR_UNAVAILABLE = True
            return self._run_ocr(image_bytes), 0.0

        text = "\n".join(self._collect_paddle_text(results))
        text_area = sum(self._ocr_box_area(box) for box in self._collect_paddle_boxes(results))
        image_area = float(image_width * image_height)
        density = min(text_area / image_area, 1.0) if image_area else 0.0
        return text, density

    def _collect_paddle_text(self, results: Any) -> list[str]:
        """Normalize PaddleOCR result shapes across v2/v3 releases."""
        lines: list[str] = []

        def walk(node: Any) -> None:
            if not node:
                return

            if isinstance(node, dict):
                for key in ("rec_texts", "texts"):
                    texts = node.get(key)
                    scores = node.get("rec_scores") or node.get("scores") or []
                    if isinstance(texts, list):
                        for index, text in enumerate(texts):
                            confidence = scores[index] if index < len(scores) else 1.0
                            if text and confidence >= settings.ocr_confidence_threshold:
                                lines.append(str(text))
                        return
                for value in node.values():
                    walk(value)
                return

            if isinstance(node, (list, tuple)):
                if (
                    len(node) >= 2
                    and isinstance(node[1], (list, tuple))
                    and len(node[1]) >= 2
                    and isinstance(node[1][0], str)
                ):
                    confidence = float(node[1][1] or 0.0)
                    if confidence >= settings.ocr_confidence_threshold:
                        lines.append(node[1][0])
                    return

                for item in node:
                    walk(item)

        walk(results)
        return lines

    def _collect_paddle_boxes(self, results: Any) -> list[Any]:
        """Normalize PaddleOCR text box shapes across v2/v3 releases."""
        boxes: list[Any] = []

        def normalize(value: Any) -> Any:
            if hasattr(value, "tolist"):
                return value.tolist()
            return value

        def add_many(values: Any) -> None:
            values = normalize(values)
            if isinstance(values, list):
                for value in values:
                    boxes.append(normalize(value))

        def walk(node: Any) -> None:
            node = normalize(node)
            if not node:
                return

            if isinstance(node, dict):
                for key in ("dt_polys", "rec_polys", "boxes", "rec_boxes"):
                    if key in node:
                        add_many(node.get(key))
                for key, value in node.items():
                    if key in {"dt_polys", "rec_polys", "boxes", "rec_boxes"}:
                        continue
                    walk(value)
                return

            if isinstance(node, (list, tuple)):
                if self._looks_like_ocr_box(node):
                    boxes.append(node)
                    return
                for item in node:
                    walk(item)

        walk(results)
        return boxes

    @staticmethod
    def _looks_like_ocr_box(value: Any) -> bool:
        if hasattr(value, "tolist"):
            value = value.tolist()
        if not isinstance(value, (list, tuple)):
            return False
        if len(value) == 4 and all(isinstance(point, (list, tuple)) and len(point) >= 2 for point in value):
            return True
        return len(value) == 4 and all(isinstance(number, Real) for number in value)

    @staticmethod
    def _ocr_box_area(box: Any) -> float:
        if hasattr(box, "tolist"):
            box = box.tolist()
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            return 0.0

        if all(isinstance(point, (list, tuple)) and len(point) >= 2 for point in box):
            points = [(float(point[0]), float(point[1])) for point in box]
            area = 0.0
            for index, point in enumerate(points):
                next_point = points[(index + 1) % len(points)]
                area += point[0] * next_point[1] - next_point[0] * point[1]
            return abs(area) / 2.0

        if all(isinstance(number, Real) for number in box):
            x1, y1, x2, y2 = [float(number) for number in box]
            return max(0.0, x2 - x1) * max(0.0, y2 - y1)

        return 0.0

    def _run_tesseract(self, image_bytes: bytes) -> str:
        try:
            import pytesseract
            from PIL import Image
        except ImportError:
            logger.warning("Tesseract or Pillow not installed, skipping fallback OCR.")
            return ""

        try:
            image = self._prepare_image_for_ocr(image_bytes)
            if isinstance(image, bytes):
                image = Image.open(io.BytesIO(image))
            return pytesseract.image_to_string(image, lang="eng")
        except Exception as exc:
            logger.warning("Tesseract OCR failed: %s. Skipping OCR for this image.", exc)
            return ""

    def _prepare_image_for_ocr(self, image_bytes: bytes) -> Any:
        try:
            import cv2
            import numpy as np
            from PIL import Image

            array = np.frombuffer(image_bytes, dtype=np.uint8)
            image = cv2.imdecode(array, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Unable to decode image bytes")

            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Denoise
            denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)

            # Adaptive thresholding for better binarization
            thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

            # Upscale for better OCR
            upscale = cv2.resize(thresh, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

            return Image.fromarray(upscale)
        except Exception:
            try:
                from PIL import Image, ImageOps, ImageFilter

                image = Image.open(io.BytesIO(image_bytes)).convert("L")
                # Apply filters
                image = image.filter(ImageFilter.MedianFilter(size=3))
                image = ImageOps.autocontrast(image)
                image = image.resize((image.width * 2, image.height * 2), Image.LANCZOS)
                return image
            except Exception as exc:  # pragma: no cover
                logger.warning("Fallback image preprocessing failed: %s", exc)
                return image_bytes

    def _extract_tables(self, filepath: str, page_number: int) -> list[Dict[str, Any]]:
        """Extract native PDF tables first, then OCR tables from scanned/image pages."""
        try:
            native_tables = self._extract_tables_with_pdfplumber(filepath, page_number)
            if native_tables:
                return native_tables
        except Exception as exc:
            logger.warning("Native table extraction failed for page=%s: %s", page_number, exc)

        return self._extract_tables_with_ppstructure(filepath, page_number)

    def _extract_tables_with_pdfplumber(self, filepath: str, page_number: int) -> list[Dict[str, Any]]:
        try:
            import pdfplumber
        except ImportError:
            raise

        tables: list[Dict[str, Any]] = []
        with pdfplumber.open(filepath) as pdf:
            if page_number - 1 >= len(pdf.pages):
                return tables
            page = pdf.pages[page_number - 1]
            extracted = page.extract_tables()
            for table_index, raw_table in enumerate(extracted):
                if not raw_table:
                    continue
                header = ["" if cell is None else str(cell).strip() for cell in raw_table[0]]
                rows = [
                    ["" if cell is None else str(cell).strip() for cell in row]
                    for row in raw_table[1:]
                    if row and any(cell is not None for cell in row)
                ]
                table = self._build_table_result(
                    page_number=page_number,
                    table_index=table_index + 1,
                    header=header,
                    rows=rows,
                    confidence=1.0,
                    source="pdfplumber",
                )
                if table:
                    tables.append(table)
                    logger.info(
                        "DEBUG: Detected table page=%s source=pdfplumber table=%s rows=%s columns=%s confidence=%.3f",
                        page_number,
                        table_index + 1,
                        len(table["rows"]),
                        len(table["header"]),
                        table["confidence"],
                    )
        return tables

    def _extract_tables_with_ppstructure(self, filepath: str, page_number: int) -> list[Dict[str, Any]]:
        """Extract tables from scanned pages using PaddleOCR PPStructure."""
        global _PPSTRUCTURE_INSTANCE, _PPSTRUCTURE_UNAVAILABLE

        if _PPSTRUCTURE_UNAVAILABLE:
            return []

        try:
            import cv2
            import numpy as np
            from paddleocr import PPStructure
        except ImportError as exc:
            logger.warning("PaddleOCR PPStructure is unavailable for OCR table extraction: %s", exc)
            _PPSTRUCTURE_UNAVAILABLE = True
            return []
        except Exception as exc:
            logger.warning("PaddleOCR PPStructure import failed: %s", exc)
            _PPSTRUCTURE_UNAVAILABLE = True
            return []

        if _PPSTRUCTURE_INSTANCE is None:
            with _PPSTRUCTURE_LOCK:
                if _PPSTRUCTURE_INSTANCE is None and not _PPSTRUCTURE_UNAVAILABLE:
                    try:
                        _PPSTRUCTURE_INSTANCE = self._create_ppstructure_engine(PPStructure)
                    except Exception as exc:
                        logger.warning("PPStructure initialization failed: %s", exc)
                        _PPSTRUCTURE_UNAVAILABLE = True
                        return []

        image_bytes = self._render_page_png(filepath, page_number, dpi=220)
        if not image_bytes:
            return []

        page_array = np.frombuffer(image_bytes, dtype=np.uint8)
        page_image = cv2.imdecode(page_array, cv2.IMREAD_COLOR)
        if page_image is None:
            logger.warning("Unable to decode rendered page %s for OCR table extraction.", page_number)
            return []

        try:
            structure_results = _PPSTRUCTURE_INSTANCE(page_image)
        except Exception as exc:
            logger.warning("PPStructure table extraction failed for page=%s: %s", page_number, exc)
            return []

        tables: list[Dict[str, Any]] = []
        for result in structure_results or []:
            if (result.get("type") or "").lower() != "table":
                continue

            html = self._extract_ppstructure_html(result)
            matrix = self._parse_table_html(html)
            if not matrix:
                logger.info("DEBUG: Skipping table page=%s source=ppstructure reason=no_html_rows", page_number)
                continue

            header, rows = self._split_table_matrix(matrix)
            confidence = self._ppstructure_confidence(result, matrix)
            table = self._build_table_result(
                page_number=page_number,
                table_index=len(tables) + 1,
                header=header,
                rows=rows,
                confidence=confidence,
                source="ppstructure",
            )
            if not table:
                logger.info(
                    "DEBUG: Skipping table page=%s source=ppstructure reason=false_positive rows=%s confidence=%.3f",
                    page_number,
                    len(rows),
                    confidence,
                )
                continue

            tables.append(table)
            logger.info(
                "DEBUG: Detected table page=%s source=ppstructure table=%s rows=%s columns=%s confidence=%.3f",
                page_number,
                len(tables),
                len(table["rows"]),
                len(table["header"]),
                table["confidence"],
            )

        return tables

    @staticmethod
    def _create_ppstructure_engine(ppstructure_cls: Any) -> Any:
        option_sets = [
            {
                "show_log": False,
                "lang": "en",
                "table": True,
                "ocr": True,
                "layout": True,
                "recovery": False,
                "use_gpu": False,
            },
            {
                "lang": "en",
                "table": True,
                "ocr": True,
                "layout": True,
                "recovery": False,
            },
            {
                "lang": "en",
                "table": True,
                "ocr": True,
            },
        ]
        last_error: Optional[Exception] = None
        for options in option_sets:
            try:
                return ppstructure_cls(**options)
            except TypeError as exc:
                last_error = exc
                continue
        if last_error:
            raise last_error
        return ppstructure_cls()

    def _render_page_png(self, filepath: str, page_number: int, dpi: int = 220) -> Optional[bytes]:
        try:
            import fitz
        except ImportError:
            logger.warning("PyMuPDF is not installed: unable to render page %s.", page_number)
            return None

        try:
            with fitz.open(filepath) as document:
                if page_number < 1 or page_number > len(document):
                    return None
                zoom = dpi / 72
                pix = document[page_number - 1].get_pixmap(
                    matrix=fitz.Matrix(zoom, zoom),
                    alpha=False,
                )
                return pix.tobytes("png")
        except Exception as exc:
            logger.warning("Unable to render page %s: %s", page_number, exc)
            return None

    @staticmethod
    def _extract_ppstructure_html(result: Dict[str, Any]) -> str:
        res = result.get("res") or {}
        if isinstance(res, dict):
            html = res.get("html") or res.get("table_html") or ""
            if isinstance(html, str):
                return html
        if isinstance(res, str):
            return res
        html = result.get("html") or ""
        return html if isinstance(html, str) else ""

    @staticmethod
    def _parse_table_html(html: str) -> list[list[str]]:
        if not html:
            return []
        parser = _TableHTMLParser()
        try:
            parser.feed(html)
        except Exception as exc:
            logger.warning("Failed to parse OCR table HTML: %s", exc)
            return []
        return [
            [cell.strip() for cell in row]
            for row in parser.rows
            if any(cell.strip() for cell in row)
        ]

    @staticmethod
    def _split_table_matrix(matrix: list[list[str]]) -> tuple[list[str], list[list[str]]]:
        if not matrix:
            return [], []
        max_columns = max(len(row) for row in matrix)
        normalized = [
            row + [""] * (max_columns - len(row))
            for row in matrix
        ]
        header = normalized[0]
        rows = normalized[1:]
        return header, rows

    def _build_table_result(
        self,
        page_number: int,
        table_index: int,
        header: list[str],
        rows: list[list[str]],
        confidence: float,
        source: str,
    ) -> Optional[Dict[str, Any]]:
        header = [self._clean_table_cell(cell) for cell in header]
        rows = [
            [self._clean_table_cell(cell) for cell in row]
            for row in rows
        ]
        rows = [row for row in rows if any(cell for cell in row)]
        column_count = max([len(header), *(len(row) for row in rows)] or [0])
        if column_count:
            header = header + [""] * (column_count - len(header))
            rows = [row + [""] * (column_count - len(row)) for row in rows]

        if self._is_false_positive_table(header, rows, confidence):
            return None

        table_text = self._table_to_pipe_text(header, rows)
        return {
            "title": f"Page {page_number} table {table_index}",
            "header": header,
            "rows": rows,
            "table_text": table_text,
            "page_number": page_number,
            "content_type": "table",
            "confidence": confidence,
            "source": source,
        }

    @staticmethod
    def _clean_table_cell(value: Any) -> str:
        if value is None:
            return ""
        return re.sub(r"\s+", " ", str(value)).strip()

    @staticmethod
    def _table_to_pipe_text(header: list[str], rows: list[list[str]]) -> str:
        lines = []
        if any(header):
            lines.append(" | ".join(header))
        for row in rows:
            if any(row):
                lines.append(" | ".join(row))
        return "\n".join(lines).strip()

    def _is_false_positive_table(self, header: list[str], rows: list[list[str]], confidence: float) -> bool:
        if confidence < 0.35:
            return True
        if len(rows) < 1:
            return True
        column_count = max([len(header), *(len(row) for row in rows)] or [0])
        if column_count < 2:
            return True

        cells = header + [cell for row in rows for cell in row]
        non_empty_cells = [cell for cell in cells if cell]
        if len(non_empty_cells) < 4:
            return True

        total_cells = max(len(cells), 1)
        fill_ratio = len(non_empty_cells) / total_cells
        if fill_ratio < 0.25:
            return True

        long_text_cells = [cell for cell in non_empty_cells if len(cell.split()) > 18]
        if len(long_text_cells) / max(len(non_empty_cells), 1) > 0.50:
            return True

        return False

    @staticmethod
    def _ppstructure_confidence(result: Dict[str, Any], matrix: list[list[str]]) -> float:
        candidates: list[float] = []

        def collect(node: Any) -> None:
            if isinstance(node, dict):
                for key in ("confidence", "score", "cell_confidence", "rec_scores"):
                    value = node.get(key)
                    if isinstance(value, Real):
                        candidates.append(float(value))
                    elif isinstance(value, list):
                        candidates.extend(float(item) for item in value if isinstance(item, Real))
                for value in node.values():
                    collect(value)
            elif isinstance(node, (list, tuple)):
                for item in node:
                    collect(item)

        collect(result)
        if candidates:
            return max(0.0, min(sum(candidates) / len(candidates), 1.0))

        cells = [cell for row in matrix for cell in row]
        non_empty = [cell for cell in cells if cell.strip()]
        return len(non_empty) / max(len(cells), 1)

    def _extract_diagrams(
        self,
        filepath: str,
        file_id: Optional[str],
        page_number: int,
        native_text: str,
        ocr_text: str,
    ) -> list[Dict[str, Any]]:
        """Detect and save cropped engineering diagram regions from a rendered PDF page."""
        diagrams: list[Dict[str, Any]] = []
        candidates = self._detect_diagram_candidates(filepath, page_number)
        if not candidates:
            logger.info("DEBUG: _extract_diagrams found no diagram candidates for page=%s", page_number)
            return diagrams

        for candidate in candidates:
            image_index = candidate["image_index"]
            image_bytes = candidate["bytes"]
            extracted_text, ocr_density = self._run_paddleocr_with_density(image_bytes)
            logger.info(
                "DEBUG: OCR density page=%s candidate=%s density=%.3f text_len=%s",
                page_number,
                image_index,
                ocr_density,
                len(extracted_text),
            )

            if ocr_density >= 0.60:
                logger.info(
                    "DEBUG: Skipping image page=%s candidate=%s because OCR text density %.3f >= 0.600.",
                    page_number,
                    image_index,
                    ocr_density,
                )
                continue

            if native_text.strip() and extracted_text.strip():
                if self._is_similar_text(extracted_text, native_text, threshold=0.78):
                    logger.info(
                        "DEBUG: Skipping image page=%s candidate=%s because crop OCR matches native page text.",
                        page_number,
                        image_index,
                    )
                    continue
            if ocr_text.strip() and extracted_text.strip():
                if self._is_similar_text(extracted_text, ocr_text, threshold=0.78):
                    logger.info(
                        "DEBUG: Skipping image page=%s candidate=%s because crop OCR matches page OCR text.",
                        page_number,
                        image_index,
                    )
                    continue

            description = self._describe_diagram(image_bytes, extracted_text)
            if not description.strip():
                logger.info(
                    "DEBUG: Skipping image page=%s candidate=%s because no diagram description was produced.",
                    page_number,
                    image_index,
                )
                continue

            image_url = self._save_diagram_image(
                file_id=file_id,
                page_number=page_number,
                image_index=image_index,
                image_bytes=image_bytes,
            )
            diagrams.append(
                {
                    "image_index": image_index,
                    "description": description.strip(),
                    "ocr_text": extracted_text.strip(),
                    "image_url": image_url,
                }
            )
            logger.info(
                "DEBUG: Detected diagram page=%s candidate=%s image_url=%s "
                "bbox=%s area_ratio=%.3f edge_density=%.3f ocr_density=%.3f",
                page_number,
                image_index,
                image_url,
                candidate["bbox"],
                candidate["area_ratio"],
                candidate["edge_density"],
                ocr_density,
            )

        logger.info(f"DEBUG: _extract_diagrams returned {len(diagrams)} diagrams for page={page_number}")
        return diagrams

    def _detect_diagram_candidates(self, filepath: str, page_number: int) -> list[Dict[str, Any]]:
        """Use OpenCV contours on a rendered page to find large, structured diagram crops."""
        try:
            import cv2
            import fitz
            import numpy as np
        except ImportError as exc:
            logger.warning("Diagram detection requires PyMuPDF, OpenCV, and NumPy: %s", exc)
            return []

        try:
            with fitz.open(filepath) as document:
                if page_number < 1 or page_number > len(document):
                    return []
                page = document[page_number - 1]
                zoom = 200 / 72
                pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
                page_bytes = pix.tobytes("png")
        except Exception as exc:
            logger.warning("Unable to render page %s for diagram detection: %s", page_number, exc)
            return []

        page_array = np.frombuffer(page_bytes, dtype=np.uint8)
        page_image = cv2.imdecode(page_array, cv2.IMREAD_COLOR)
        if page_image is None:
            logger.warning("Unable to decode rendered page %s for diagram detection.", page_number)
            return []

        page_height, page_width = page_image.shape[:2]
        page_area = float(page_width * page_height)
        gray = cv2.cvtColor(page_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(blurred, 50, 150)

        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (19, 19))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, close_kernel, iterations=2)
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        grouped = cv2.dilate(closed, dilate_kernel, iterations=1)
        contours, _ = cv2.findContours(grouped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates: list[Dict[str, Any]] = []
        for contour_index, contour in enumerate(contours):
            contour_area = float(cv2.contourArea(contour))
            x, y, width, height = cv2.boundingRect(contour)
            bbox_area = float(width * height)
            area_ratio = bbox_area / page_area if page_area else 0.0
            aspect_ratio = width / max(height, 1)
            edge_density = self._edge_density(edges, x, y, width, height)
            logger.info(
                "DEBUG: Contour size page=%s contour=%s bbox=(%s,%s,%s,%s) "
                "contour_area=%.1f bbox_area=%.1f area_ratio=%.3f aspect_ratio=%.3f edge_density=%.3f",
                page_number,
                contour_index,
                x,
                y,
                width,
                height,
                contour_area,
                bbox_area,
                area_ratio,
                aspect_ratio,
                edge_density,
            )

            skip_reason = self._diagram_candidate_skip_reason(
                page_width=page_width,
                page_height=page_height,
                x=x,
                y=y,
                width=width,
                height=height,
                contour_area=contour_area,
                area_ratio=area_ratio,
                aspect_ratio=aspect_ratio,
                edge_density=edge_density,
            )
            if skip_reason:
                logger.info(
                    "DEBUG: Skipping image page=%s contour=%s reason=%s bbox=(%s,%s,%s,%s)",
                    page_number,
                    contour_index,
                    skip_reason,
                    x,
                    y,
                    width,
                    height,
                )
                continue

            crop_x, crop_y, crop_w, crop_h = self._expand_bbox(
                x,
                y,
                width,
                height,
                page_width,
                page_height,
                padding=12,
            )
            crop = page_image[crop_y : crop_y + crop_h, crop_x : crop_x + crop_w]
            ok, encoded = cv2.imencode(".png", crop)
            if not ok:
                logger.info(
                    "DEBUG: Skipping image page=%s contour=%s reason=unable_to_encode_crop",
                    page_number,
                    contour_index,
                )
                continue

            candidates.append(
                {
                    "image_index": f"diagram_{len(candidates) + 1}",
                    "bytes": encoded.tobytes(),
                    "bbox": (crop_x, crop_y, crop_w, crop_h),
                    "area_ratio": (crop_w * crop_h) / page_area if page_area else 0.0,
                    "edge_density": edge_density,
                    "contour_area": contour_area,
                }
            )

        return self._dedupe_diagram_candidates(candidates)

    def _diagram_candidate_skip_reason(
        self,
        page_width: int,
        page_height: int,
        x: int,
        y: int,
        width: int,
        height: int,
        contour_area: float,
        area_ratio: float,
        aspect_ratio: float,
        edge_density: float,
    ) -> str:
        if width <= 300:
            return "width_below_300px"
        if area_ratio <= 0.15:
            return "area_below_15_percent_of_page"
        if area_ratio >= 0.92:
            return "full_page_screenshot_candidate"
        if contour_area <= 2_000:
            return "contour_area_too_small"
        if aspect_ratio < 0.25 or aspect_ratio > 6.0:
            return "aspect_ratio_out_of_range"
        if edge_density < 0.003:
            return "edge_density_too_low"
        if edge_density > 0.35:
            return "edge_density_too_high_for_diagram"
        if self._is_header_or_footer_region(y, height, page_height):
            return "header_or_footer_region"
        if self._touches_page_boundary(x, y, width, height, page_width, page_height):
            return "near_full_page_boundary"
        return ""

    @staticmethod
    def _edge_density(edges: Any, x: int, y: int, width: int, height: int) -> float:
        region = edges[y : y + height, x : x + width]
        if region.size == 0:
            return 0.0
        return float((region > 0).sum()) / float(region.size)

    @staticmethod
    def _expand_bbox(
        x: int,
        y: int,
        width: int,
        height: int,
        page_width: int,
        page_height: int,
        padding: int,
    ) -> tuple[int, int, int, int]:
        crop_x = max(0, x - padding)
        crop_y = max(0, y - padding)
        crop_right = min(page_width, x + width + padding)
        crop_bottom = min(page_height, y + height + padding)
        return crop_x, crop_y, crop_right - crop_x, crop_bottom - crop_y

    @staticmethod
    def _is_header_or_footer_region(y: int, height: int, page_height: int) -> bool:
        center_y = y + (height / 2)
        return center_y < page_height * 0.10 or center_y > page_height * 0.90

    @staticmethod
    def _touches_page_boundary(
        x: int,
        y: int,
        width: int,
        height: int,
        page_width: int,
        page_height: int,
    ) -> bool:
        margin_x = page_width * 0.015
        margin_y = page_height * 0.015
        covers_most_width = width >= page_width * 0.94
        covers_most_height = height >= page_height * 0.94
        touches_horizontal = x <= margin_x and x + width >= page_width - margin_x
        touches_vertical = y <= margin_y and y + height >= page_height - margin_y
        return (covers_most_width and touches_horizontal) or (covers_most_height and touches_vertical)

    def _dedupe_diagram_candidates(self, candidates: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
        deduped: list[Dict[str, Any]] = []
        for candidate in sorted(candidates, key=lambda item: item["area_ratio"], reverse=True):
            if any(self._bbox_overlap_ratio(candidate["bbox"], kept["bbox"]) > 0.80 for kept in deduped):
                logger.info(
                    "DEBUG: Skipping image candidate=%s reason=overlaps_larger_candidate bbox=%s",
                    candidate["image_index"],
                    candidate["bbox"],
                )
                continue
            candidate["image_index"] = f"diagram_{len(deduped) + 1}"
            deduped.append(candidate)
        return deduped

    @staticmethod
    def _bbox_overlap_ratio(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        left = max(ax, bx)
        top = max(ay, by)
        right = min(ax + aw, bx + bw)
        bottom = min(ay + ah, by + bh)
        if right <= left or bottom <= top:
            return 0.0
        intersection = float((right - left) * (bottom - top))
        smaller_area = float(min(aw * ah, bw * bh))
        return intersection / smaller_area if smaller_area else 0.0

    def _normalize_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip().lower()

    def _is_similar_text(self, a: str, b: str, threshold: float = 0.8) -> bool:
        a_norm = self._normalize_text(a)
        b_norm = self._normalize_text(b)
        if not a_norm or not b_norm:
            return False

        if a_norm in b_norm or b_norm in a_norm:
            return True

        words_a = a_norm.split()
        words_b = b_norm.split()
        overlap = len(set(words_a) & set(words_b))
        match_ratio = overlap / max(len(set(words_a)), len(set(words_b)), 1)
        return match_ratio >= threshold

    def _describe_diagram(self, image_bytes: bytes, extracted_text: str) -> str:
        caption = self._caption_image(image_bytes)
        if extracted_text:
            caption = f"{caption} Diagram labels: {extracted_text}" if caption else f"Diagram labels: {extracted_text}"
        if not caption:
            caption = "A diagram or engineering drawing with visual structure and text labels."
        return caption

    def _save_diagram_image(
        self,
        file_id: Optional[str],
        page_number: int,
        image_index: Optional[Any],
        image_bytes: bytes,
    ) -> Optional[str]:
        if not file_id or image_index is None:
            return None

        try:
            diagram_root = Path(settings.upload_dir) / "diagrams" / str(file_id) / str(page_number)
            diagram_root.mkdir(parents=True, exist_ok=True)
            safe_index = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(image_index)).strip("_")
            diagram_name = f"{safe_index or 'diagram'}.png"
            diagram_path = diagram_root / diagram_name
            diagram_path.write_bytes(image_bytes)
            return f"/static/uploads/diagrams/{file_id}/{page_number}/{diagram_name}"
        except Exception as exc:
            logger.warning("Failed to save diagram image: %s", exc)
            return None

    def _caption_image(self, image_bytes: bytes) -> str:
        if not self._diagram_captioning_enabled:
            return "A diagram or figure image with structural layout."

        try:
            from PIL import Image
            from transformers import BlipProcessor, BlipForConditionalGeneration
            import torch
        except ImportError:
            logger.warning(
                "Transformer captioning stack is not installed. Falling back to a generic diagram description."
            )
            return "A diagram or figure image with structural layout."

        if self._caption_model is None or self._caption_processor is None:
            self._caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
            self._caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            # Generate multiple captions with different prompts
            prompts = ["", "This is a technical diagram showing ", "An engineering drawing of "]
            captions = []
            for prompt in prompts:
                pixel_values = self._caption_processor(images=image, text=prompt, return_tensors="pt").pixel_values
                with torch.no_grad():
                    generated_ids = self._caption_model.generate(pixel_values, max_new_tokens=50)
                caption = self._caption_processor.decode(generated_ids[0], skip_special_tokens=True)
                captions.append(caption.strip())
            # Combine captions
            return " | ".join(set(captions))  # Remove duplicates
        except Exception as exc:
            logger.warning("Image captioning failed: %s", exc)
            return "A diagram or figure image with structural layout."
