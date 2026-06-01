"""
PDF extraction pipeline for native text, images, OCR, tables, and diagram descriptions.
"""
import io
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()
_PADDLE_OCR_INSTANCE = None
_PADDLE_OCR_LOCK = threading.Lock()
_PADDLE_OCR_UNAVAILABLE = False


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
        # Try Docling first for better structured extraction if enabled
        if settings.use_docling:
            try:
                return self._extract_with_docling(filepath, file_id=file_id)
            except Exception as exc:
                logger.warning("Docling extraction failed: %s. Falling back to traditional extraction.", exc)
        
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
            ocr_inputs.extend(page_images)
            ocr_text = self._extract_ocr_text(ocr_inputs)
            tables = self._extract_tables(filepath, page_number)
            diagrams = self._extract_diagrams(page_images, tables, file_id=file_id, page_number=page_number)

            pages.append(
                {
                    "page_number": page_number,
                    "native_text": native_text.strip(),
                    "ocr_text": ocr_text.strip(),
                    "tables": tables,
                    "diagrams": diagrams,
                }
            )

        return pages

    def _extract_with_docling(self, filepath: str, file_id: Optional[str] = None) -> list[Dict[str, Any]]:
        """Extract document content using Docling for better structured parsing."""
        try:
            from docling.document_converter import DocumentConverter
            from docling.document_converter import PdfFormatOption
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
        except ImportError as exc:
            raise ImportError("docling is required for advanced PDF processing") from exc

        # Configure Docling pipeline options
        pipeline_options = PdfPipelineOptions()
        # Keep Docling focused on text/table structure. OCR is handled below so
        # Paddle/PaddleX initialization errors cannot abort document embedding.
        pipeline_options.do_ocr = False
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

        # Convert document
        result = doc_converter.convert(filepath)
        doc = result.document

        images_by_page = self._extract_page_images(filepath)
        native_pages_by_number: dict[int, str] = {}
        pages: list[Dict[str, Any]] = []

        page_items = self._iter_docling_pages(doc)
        for page_number, page_obj in page_items:
            try:
                if not hasattr(page_obj, "blocks"):
                    raise AttributeError("Docling page object has no attribute 'blocks'")

                # Extract text and tables from blocks
                text_items = []
                tables = []

                for block_idx, block in enumerate(page_obj.blocks):
                    block_type = type(block).__name__

                    # Extract text from text blocks
                    if block_type == "TextBlock":
                        for line in block.text.split('\n'):
                            if line.strip():
                                text_items.append(line.strip())

                    # Extract table structure
                    elif block_type == "TableBlock":
                        table_data = self._extract_docling_table(block)
                        if table_data:
                            tables.append(table_data)

                native_text = " ".join(text_items)
                native_pages_by_number[page_number] = native_text

                # Extract images for OCR and diagram processing
                page_images = images_by_page.get(page_number, [])
                diagrams = self._extract_diagrams(page_images, tables, file_id=file_id, page_number=page_number)

                pages.append({
                    "page_number": page_number,
                    "native_text": native_text.strip(),
                    "ocr_text": "",
                    "tables": tables,
                    "diagrams": diagrams,
                })
                logger.debug(f"Page {page_number}: extracted {len(text_items)} text items, {len(tables)} tables")
            except Exception as e:
                logger.warning(f"Error processing page {page_number} with Docling: {e}")
                # Append empty page structure for consistency
                native_pages_by_number[page_number] = ""
                pages.append({
                    "page_number": page_number,
                    "native_text": "",
                    "ocr_text": "",
                    "tables": [],
                    "diagrams": [],
                })

        native_pages = [native_pages_by_number.get(page_number, "") for page_number, _ in page_items]
        rendered_pages = self._render_pages_for_ocr(filepath, native_pages)
        for page in pages:
            page_number = page["page_number"]
            page_images = images_by_page.get(page_number, [])
            ocr_inputs = []
            rendered_page = rendered_pages.get(page_number)
            if rendered_page:
                ocr_inputs.extend(rendered_page)
            ocr_inputs.extend(page_images)
            page["ocr_text"] = self._extract_ocr_text(ocr_inputs).strip()
        
        logger.info("Successfully extracted %d pages using Docling", len(pages))
        return pages

    def _iter_docling_pages(self, doc: Any) -> list[tuple[int, Any]]:
        """Normalize Docling page containers into an ordered list of (page_number, page_object)."""
        pages = getattr(doc, "pages", [])
        if pages is None:
            return []

        if isinstance(pages, dict):
            try:
                return sorted(pages.items(), key=lambda item: int(item[0]))
            except Exception:
                return list(pages.items())

        if hasattr(pages, "items") and callable(getattr(pages, "items")):
            try:
                return sorted(pages.items(), key=lambda item: int(item[0]))
            except Exception:
                return list(pages.items())

        return list(enumerate(pages, start=1))

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

        return "\n".join(ocr_texts)

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
        tables: list[Dict[str, Any]] = []

        table_extractors = [self._extract_tables_with_camelot, self._extract_tables_with_pdfplumber]
        for extractor in table_extractors:
            try:
                extracted = extractor(filepath, page_number)
                if extracted:
                    tables.extend(extracted)
                    break
            except Exception as exc:
                logger.warning("Table extraction with %s failed: %s", extractor.__name__, exc)

        return tables

    def _extract_tables_with_camelot(self, filepath: str, page_number: int) -> list[Dict[str, Any]]:
        try:
            import camelot
        except ImportError:
            raise

        page = str(page_number)
        tables: list[Dict[str, Any]] = []

        for flavor in ["lattice", "stream"]:
            try:
                camelot_tables = camelot.read_pdf(filepath, pages=page, flavor=flavor)
                for table_index, table in enumerate(camelot_tables):
                    data = table.data if hasattr(table, "data") else []
                    if not data:
                        continue
                    header = [str(cell).strip() for cell in data[0]]
                    rows = [[str(cell).strip() for cell in row] for row in data[1:]]
                    tables.append(
                        {
                            "title": f"Page {page_number} table {table_index + 1}",
                            "header": header,
                            "rows": rows,
                        }
                    )
                if tables:
                    return tables
            except Exception:
                continue

        return tables

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
                header = [str(cell).strip() for cell in raw_table[0]]
                rows = [
                    [str(cell).strip() for cell in row]
                    for row in raw_table[1:]
                    if row and any(cell is not None for cell in row)
                ]
                if header and rows:
                    tables.append(
                        {
                            "title": f"Page {page_number} table {table_index + 1}",
                            "header": header,
                            "rows": rows,
                        }
                    )
        return tables

    def _extract_docling_table(self, table_block: Any) -> Optional[Dict[str, Any]]:
        """Extract table structure from Docling TableBlock."""
        try:
            # Try to access table data
            table_data = None
            if hasattr(table_block, 'data'):
                table_data = table_block.data
            elif hasattr(table_block, 'table') and hasattr(table_block.table, 'data'):
                table_data = table_block.table.data
            
            if not table_data:
                return None
            
            # Convert to list of lists if needed
            if not isinstance(table_data, list):
                return None
            
            if len(table_data) == 0:
                return None
            
            # First row becomes header, rest becomes rows
            header = [str(cell).strip() for cell in table_data[0] if cell]
            rows = [
                [str(cell).strip() for cell in row if cell]
                for row in table_data[1:]
            ]
            
            if not header:
                return None
            
            return {
                "title": "Extracted Table",
                "header": header,
                "rows": rows,
            }
        except Exception as e:
            logger.debug(f"Error extracting Docling table: {e}")
            return None

    def _extract_diagrams(
        self,
        images: List[Dict[str, Any]],
        tables: list[Dict[str, Any]],
        file_id: Optional[str],
        page_number: int,
    ) -> list[Dict[str, Any]]:
        diagrams: list[Dict[str, Any]] = []
        if not images:
            return diagrams

        for image in images:
            image_bytes = image.get("bytes")
            if not image_bytes:
                continue

            extracted_text = self._run_ocr(image_bytes)
            description = self._describe_diagram(image_bytes, extracted_text)
            if description:
                image_url = self._save_diagram_image(
                    file_id=file_id,
                    page_number=page_number,
                    image_index=image.get("image_index"),
                    image_bytes=image_bytes,
                )
                diagrams.append(
                    {
                        "image_index": image.get("image_index"),
                        "description": description.strip(),
                        "ocr_text": extracted_text.strip(),
                        "image_url": image_url,
                    }
                )

        return diagrams

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
        image_index: Optional[int],
        image_bytes: bytes,
    ) -> Optional[str]:
        if not file_id or image_index is None:
            return None

        try:
            diagram_root = Path(settings.upload_dir) / "diagrams" / str(file_id) / str(page_number)
            diagram_root.mkdir(parents=True, exist_ok=True)
            diagram_name = f"diagram_{image_index}.png"
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
