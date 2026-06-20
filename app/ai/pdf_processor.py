"""
PDF extraction pipeline for native text, images, OCR, tables, and diagram descriptions.
"""
import hashlib
import io
import re
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
_DOCLING_UNAVAILABLE = False

# OCR images are resized into this band: PaddleOCR's detector struggles with
# tiny crops, while very large renders are slow and memory-hungry (peak RAM
# scales with pixel count, and a too-large page can OOM a small host). 2000px
# on the long side keeps dense full-page tables readable while staying lean.
_MAX_OCR_SIDE = 2000
_MIN_OCR_SIDE = 720
# Embedded images smaller than this (icons, rules, bullets) carry no text.
_MIN_EMBEDDED_IMAGE_SIDE = 50

_TESSERACT_LANG_MAP = {"en": "eng", "hi": "hin", "ch": "chi_sim"}


class PDFProcessor:
    """Extract page-level multimodal data from PDF documents."""

    def __init__(self) -> None:
        self._ocr_engine = settings.ocr_engine or "paddleocr"
        self._diagram_captioning_enabled = settings.enable_diagram_captioning
        self._caption_model = None
        self._caption_processor = None
        self._paddle_ocr = None  # Singleton instance to prevent PDX reinitialization
        self._ocr_cache: Dict[str, str] = {}  # bytes-hash -> text, so each image is OCR'd once
        self._camelot_unavailable = False

    def extract_page_documents(self, filepath: str, file_id: Optional[str] = None) -> list[Dict[str, Any]]:
        """Extract native text, OCR text, tables, and diagrams for each PDF page."""
        global _DOCLING_UNAVAILABLE

        self._ocr_cache.clear()

        # Try Docling first for better structured extraction if enabled
        if settings.use_docling and not _DOCLING_UNAVAILABLE:
            try:
                return self._extract_with_docling(filepath, file_id=file_id)
            except ImportError as exc:
                _DOCLING_UNAVAILABLE = True
                logger.warning("Docling is not installed: %s. Using traditional extraction from now on.", exc)
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
        for page_number, raw_native_text in enumerate(native_pages, start=1):
            page_images = images_by_page.get(page_number, [])
            native_text, ocr_text = self._resolve_page_text(
                raw_native_text, rendered_pages.get(page_number), page_images
            )
            # Surface silent content loss: an image-based page that yields no text
            # usually means OCR failed (often out-of-memory on a large render),
            # not that the page is blank — flag it instead of dropping it quietly.
            if not native_text and not ocr_text and (rendered_pages.get(page_number) or page_images):
                logger.warning(
                    "Page %s produced no text from OCR; it will have no chunks "
                    "(OCR may have failed, e.g. low memory).",
                    page_number,
                )
            # Tables come from a real text/vector layer (a sparse data table still
            # counts). A table-extraction failure must never lose the page.
            tables = []
            if self._has_extractable_text_layer(raw_native_text):
                try:
                    tables = self._extract_tables(filepath, page_number)
                except Exception as exc:
                    logger.warning("Table extraction failed on page %s: %s", page_number, exc)
            diagrams = self._extract_diagrams(
                page_images, tables, file_id=file_id, page_number=page_number
            )
            logger.info(
                "page=%s mode=traditional tables=%s images=%s diagrams=%s "
                "text=%s ocr=%s",
                page_number, len(tables), len(page_images), len(diagrams),
                bool(native_text), bool(ocr_text),
            )

            pages.append(
                {
                    "page_number": page_number,
                    "native_text": native_text,
                    "ocr_text": ocr_text,
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
        # By default keep OCR with our Paddle pipeline (below) so Docling stays
        # light and Paddle errors can't abort embedding. Enable docling_do_ocr to
        # let Docling OCR pages itself — required to recover tables from SCANNED
        # PDFs via TableFormer, at the cost of extra memory/models.
        pipeline_options.do_ocr = settings.docling_do_ocr
        pipeline_options.do_table_structure = True  # TableFormer cell structure
        if settings.docling_do_ocr:
            # A fully-scanned page is one big image; without full-page OCR Docling
            # only reads detected sub-regions and TableFormer gets no cell text,
            # so it finds no table. Force OCR over the whole page.
            try:
                pipeline_options.ocr_options.force_full_page_ocr = True
            except Exception as exc:
                logger.debug("Could not set force_full_page_ocr: %s", exc)
        # Cross-link table cells with the page text so TableFormer reconstructs values.
        try:
            pipeline_options.table_structure_options.do_cell_matching = True
        except Exception:
            pass
        # Render pages at higher resolution so TableFormer sees clean cell geometry
        # (reduces row/cell off-by-one misalignment on scanned tables).
        try:
            pipeline_options.images_scale = settings.docling_images_scale
        except Exception as exc:
            logger.debug("Could not set docling images_scale: %s", exc)
        # Prefer the accurate (heavier) TableFormer model for better cell matching.
        if settings.docling_table_accurate:
            try:
                from docling.datamodel.pipeline_options import TableFormerMode

                pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
            except Exception as exc:
                logger.debug("Could not set TableFormer ACCURATE mode: %s", exc)

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

        # doc.pages is Dict[PageNo, PageItem] in Docling v2 — must use .items()
        page_count = len(doc.pages) if doc.pages else 0
        if page_count == 0:
            raise ValueError("Docling returned 0 pages")

        # Collect per-page text via provenance (iterate_items is the stable Docling v2 API)
        pages_text: dict[int, list[str]] = {i: [] for i in range(1, page_count + 1)}
        try:
            for item, _ in doc.iterate_items():
                # Tables and pictures are captured via their own paths. Keep their
                # text OUT of the flowing page text so a table's content lives only
                # in its dedicated table chunk, never duplicated into a text chunk.
                if type(item).__name__ in ("TableItem", "PictureItem"):
                    continue
                prov_list = getattr(item, "prov", None) or []
                page_no = prov_list[0].page_no if prov_list else None
                if page_no is None or page_no not in pages_text:
                    continue
                item_text = getattr(item, "text", None)
                if item_text and str(item_text).strip():
                    pages_text[page_no].append(str(item_text).strip())
        except Exception as exc:
            logger.warning("Failed to iterate Docling items: %s", exc)

        docling_tables = self._extract_docling_tables(doc)
        images_by_page = self._extract_page_images(filepath)
        # Join items with newlines (not spaces) so each heading/paragraph stays on
        # its own line — section detection in build_pdf_chunks splits on newlines.
        native_pages = ["\n".join(pages_text.get(i, [])) for i in range(1, page_count + 1)]
        rendered_pages = self._render_pages_for_ocr(filepath, native_pages)

        pages: list[Dict[str, Any]] = []
        for page_number in range(1, page_count + 1):
            raw_native_text = "\n".join(pages_text.get(page_number, []))
            page_images = images_by_page.get(page_number, [])
            native_text, ocr_text = self._resolve_page_text(
                raw_native_text, rendered_pages.get(page_number), page_images
            )
            # Prefer Docling's structured tables (with captions + cell grid). Fall
            # back to camelot/pdfplumber only for text-layer pages Docling missed.
            # A table failure must never lose the page.
            tables = docling_tables.get(page_number, [])
            if not tables and self._has_extractable_text_layer(raw_native_text):
                try:
                    tables = self._extract_tables(filepath, page_number)
                except Exception as exc:
                    logger.warning("Table extraction failed on page %s: %s", page_number, exc)
            diagrams = self._extract_diagrams(
                page_images, tables, file_id=file_id, page_number=page_number
            )
            logger.info(
                "page=%s mode=docling tables=%s images=%s diagrams=%s text=%s ocr=%s",
                page_number, len(tables), len(page_images), len(diagrams),
                bool(native_text), bool(ocr_text),
            )

            pages.append({
                "page_number": page_number,
                "native_text": native_text,
                "ocr_text": ocr_text,
                "tables": tables,
                "diagrams": diagrams,
            })

        logger.info("Successfully extracted %d pages using Docling", len(pages))
        return pages

    def _extract_docling_tables(self, doc: Any) -> Dict[int, List[Dict[str, Any]]]:
        """Collect Docling's structured tables, keyed by page number.

        Each table is normalized to ``{title, header, rows, markdown}`` so the
        downstream chunker can store Markdown + semantic rows under a caption.
        Defensive across Docling versions whose table API differs.

        KNOWN LIMITATION (accepted): on low-quality scans whose column labels are
        vertically offset (e.g. the BLC wagon dimensions table, where 'A'Car and
        'B'Car sit at different heights), TableFormer maps a header label's box
        into the first data row and shifts every row's values by one. Raising
        docling_images_scale and enabling docling_table_accurate were verified NOT
        to fix this (it's a box-to-cell geometry issue, not resolution/model). The
        values are still present, only mis-rowed. Reliable fix would be vision-LLM
        extraction on the table crop — deferred for now.
        """
        tables_by_page: Dict[int, List[Dict[str, Any]]] = {}
        for table in getattr(doc, "tables", None) or []:
            try:
                prov = getattr(table, "prov", None) or []
                page_no = prov[0].page_no if prov else None
            except Exception:
                page_no = None
            if not page_no:
                continue

            title = ""
            try:
                title = (table.caption_text(doc) or "").strip()
            except Exception:
                pass

            header: list = []
            rows: list = []
            df = None
            for export in (lambda: table.export_to_dataframe(doc), lambda: table.export_to_dataframe()):
                try:
                    df = export()
                    break
                except TypeError:
                    continue
                except Exception as exc:
                    logger.debug("Docling table dataframe export failed: %s", exc)
                    break
            if df is not None:
                header = [str(c).strip() for c in df.columns.tolist()]
                rows = [[str(c).strip() for c in row] for row in df.values.tolist()]

            markdown = ""
            for call in (lambda: table.export_to_markdown(doc), lambda: table.export_to_markdown()):
                try:
                    markdown = call() or ""
                    break
                except Exception:
                    continue

            if not (rows or markdown):
                continue

            tables_by_page.setdefault(page_no, []).append(
                {"title": title or "Table", "header": header, "rows": rows, "markdown": markdown}
            )

        if tables_by_page:
            logger.info("Docling extracted tables on %d page(s)", len(tables_by_page))
        return tables_by_page

    def _extract_native_text_pages(self, filepath: str) -> list[str]:
        """Extract native text with pypdf, then fill weak pages with pdfplumber text."""
        import pypdf

        native_pages: list[str] = []
        with open(filepath, "rb") as f:
            reader = pypdf.PdfReader(f)
            if reader.is_encrypted:
                try:
                    reader.decrypt("")
                except Exception as exc:
                    logger.warning("PDF is password-protected and could not be opened: %s", exc)
                    return []
            for page in reader.pages:
                try:
                    native_pages.append(page.extract_text() or "")
                except Exception as exc:
                    logger.warning("pypdf failed to extract text from page %s: %s", len(native_pages) + 1, exc)
                    native_pages.append("")

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
        zoom = settings.ocr_full_page_dpi / 72
        matrix = fitz.Matrix(zoom, zoom)

        try:
            with fitz.open(filepath) as document:
                for page_index, page in enumerate(document):
                    native_text = native_pages[page_index] if page_index < len(native_pages) else ""
                    if self._is_meaningful_text(native_text):
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

    def render_page_png(self, filepath: str, page_number: int, scale: float = 2.0) -> Optional[bytes]:
        """Render a single page to PNG bytes (for vision/chart analysis)."""
        try:
            import fitz
        except ImportError:
            return None
        try:
            with fitz.open(filepath) as document:
                if not (1 <= page_number <= len(document)):
                    return None
                pix = document[page_number - 1].get_pixmap(
                    matrix=fitz.Matrix(scale, scale), alpha=False
                )
                return pix.tobytes("png")
        except Exception as exc:
            logger.warning("Failed to render page %s: %s", page_number, exc)
            return None

    def chart_region_images(self, filepath: str, page_number: int, scale: float = 2.0) -> List[bytes]:
        """Split a page into chart regions and render each to PNG.

        Vector charts are clusters of drawing ops; pages often hold several
        side-by-side (e.g. two pie charts). We cluster the drawings into separated
        regions and crop each, so every chart is analysed on its own. Falls back
        to the whole page when there is only one (or no) clear region.
        """
        try:
            import fitz
        except ImportError:
            return []
        try:
            with fitz.open(filepath) as document:
                if not (1 <= page_number <= len(document)):
                    return []
                page = document[page_number - 1]
                matrix = fitz.Matrix(scale, scale)
                regions = self._cluster_drawing_regions(page)
                if len(regions) < 2:
                    return [page.get_pixmap(matrix=matrix, alpha=False).tobytes("png")]
                images: List[bytes] = []
                for rect in regions:
                    pix = page.get_pixmap(matrix=matrix, clip=rect, alpha=False)
                    images.append(pix.tobytes("png"))
                return images
        except Exception as exc:
            logger.warning("chart_region_images failed page %s: %s", page_number, exc)
            return []

    @staticmethod
    def _cluster_drawing_regions(page: Any) -> List[Any]:
        """Cluster a page's vector-drawing rects into separated regions (charts)."""
        import fitz

        page_rect = page.rect
        page_area = abs(page_rect.width * page_rect.height) or 1.0
        rects = []
        try:
            drawings = page.get_drawings()
        except Exception:
            return []
        for drawing in drawings:
            rect = drawing.get("rect")
            if rect is None or abs(rect.width * rect.height) < page_area * 0.0005:
                continue  # skip hairline strokes / dots
            rects.append(fitz.Rect(rect))
        if not rects:
            return []

        # Merge rects that touch (within a small gap) into connected clusters.
        gap = min(page_rect.width, page_rect.height) * 0.04
        clusters: List[Any] = []
        for rect in rects:
            grown = fitz.Rect(rect.x0 - gap, rect.y0 - gap, rect.x1 + gap, rect.y1 + gap)
            target = next((c for c in clusters if c.intersects(grown)), None)
            if target is None:
                clusters.append(fitz.Rect(rect))
            else:
                target.include_rect(rect)

        # Iterate until clusters stop merging (handles transitive overlaps).
        changed = True
        while changed and len(clusters) > 1:
            changed = False
            merged: List[Any] = []
            for cluster in clusters:
                hit = next((m for m in merged if m.intersects(cluster)), None)
                if hit is None:
                    merged.append(fitz.Rect(cluster))
                else:
                    hit.include_rect(cluster)
                    changed = True
            clusters = merged

        # Keep only regions large enough to be a real chart; biggest first.
        big = [c for c in clusters if abs(c.width * c.height) >= page_area * 0.05]
        big.sort(key=lambda c: abs(c.width * c.height), reverse=True)
        return big[:4]

    def chart_candidate_pages(self, filepath: str) -> set:
        """Pages likely to contain a chart, by any of three signals.

        A single signal is unreliable (a PowerPoint pie chart can be only ~2
        vector drawings, while a bar chart is dozens), so a page is a candidate
        if it has enough vector drawings, OR a large raster image, OR chart-like
        text (many %/number labels with few sentences). Vision then confirms or
        rejects each candidate, so over-including is cheap-ish; pure prose pages
        are still skipped to avoid wasting vision calls on manuals.
        """
        candidates: set = set()
        try:
            import fitz
        except ImportError:
            return candidates
        try:
            with fitz.open(filepath) as document:
                page_total = len(document)
                for index in range(page_total):
                    reason = self._chart_candidate_reason(document[index])
                    if reason:
                        candidates.add(index + 1)
                        logger.info("page=%s chart-candidate (%s)", index + 1, reason)
        except Exception as exc:
            logger.warning("chart_candidate_pages failed: %s", exc)
            return candidates
        logger.info("chart candidates: %s of %s pages -> %s",
                    len(candidates), page_total, sorted(candidates))
        return candidates

    @classmethod
    def _chart_candidate_reason(cls, page: Any) -> Optional[str]:
        """Return why a page looks like a chart, or None."""
        try:
            drawings = len(page.get_drawings())
        except Exception:
            drawings = 0
        if drawings >= settings.chart_candidate_min_drawings:
            return f"drawings={drawings}"

        page_area = abs(page.rect.width * page.rect.height) or 1.0
        try:
            for image_info in page.get_images(full=True):
                rects = page.get_image_rects(image_info[0])
                if rects and max(abs(r.width * r.height) for r in rects) >= page_area * settings.diagram_min_coverage:
                    return "large_raster"
        except Exception:
            pass

        if cls._looks_like_chart_text(page.get_text() or ""):
            return "chart_like_text"
        return None

    @staticmethod
    def _looks_like_chart_text(text: str) -> bool:
        """Heuristic: chart slides are dominated by short %/number labels, not prose."""
        if not text.strip():
            return False
        numbers = len(re.findall(r"\d+\s*%|\b\d{1,4}\b", text))
        sentences = len(re.findall(r"[.!?](?:\s|$)", text))
        words = max(len(text.split()), 1)
        return numbers >= 8 and sentences <= 6 and (numbers / words) >= 0.12

    @staticmethod
    def _is_meaningful_text(text: str) -> bool:
        """Whether a page's native text layer is real content, not CID/encoding junk.

        Scanned PDFs sometimes carry a broken text layer (e.g. "(cid:12)" runs or
        replacement characters) that is long enough to pass a plain length check;
        such pages still need full-page OCR.
        """
        stripped = (text or "").strip()
        if len(stripped) < settings.ocr_full_page_min_text_chars:
            return False

        # Unmapped-glyph runs: pypdf emits "(cid:NN)" when a font carries no
        # usable encoding. That layer is unreadable however long it is.
        if stripped.count("(cid:") >= 3:
            return False

        # Latin or Devanagari words of 3+ characters are a good proxy for prose.
        words = re.findall(r"[A-Za-zऀ-ॿ]{3,}", stripped)
        if len(words) < 10:
            return False

        # A layer drowning in U+FFFD replacement characters is corrupt.
        if stripped.count("�") > len(words):
            return False

        return True

    def _resolve_page_text(
        self,
        native_text: str,
        rendered_page: Optional[List[Dict[str, Any]]],
        page_images: List[Dict[str, Any]],
    ) -> tuple[str, str]:
        """Pick a page's text source: native layer first, OCR only as fallback.

        Returns ``(native_text, ocr_text)`` with exactly one side populated. A
        meaningful native layer is used as-is and the page is never OCR'd (fast
        and exact for digital PDFs); a missing or junk layer is discarded so its
        corrupted characters cannot reach retrieval, and the page is OCR'd.
        """
        native_text = (native_text or "").strip()
        if self._is_meaningful_text(native_text):
            return native_text, ""
        ocr_inputs = rendered_page or page_images
        ocr_text = self._extract_ocr_text(ocr_inputs).strip()
        return "", ocr_text

    @staticmethod
    def _has_extractable_text_layer(text: str) -> bool:
        """Whether a page carries a real text layer camelot/pdfplumber can mine
        for tables.

        Deliberately looser than _is_meaningful_text: a digital data table can be
        almost all numbers with very few prose words, yet still be perfectly
        text-based. We only reject pages with no usable layer at all — empty
        (image-only scans) or dominated by CID/replacement-character junk.
        """
        stripped = (text or "").strip()
        if len(stripped) < 20:
            return False
        # Unmapped-glyph runs => the layer is unreadable, not table-extractable.
        if stripped.count("(cid:") >= 3:
            return False
        # More than ~5% replacement characters => corrupt layer, not usable.
        if stripped.count("�") > len(stripped) // 20:
            return False
        return True

    def _extract_page_images(self, filepath: str) -> Dict[int, List[Dict[str, Any]]]:
        try:
            import fitz
        except ImportError:
            logger.warning("PyMuPDF is not installed: image extraction will be disabled.")
            return {}

        images_by_page: Dict[int, List[Dict[str, Any]]] = {}
        seen_hashes: set = set()  # dedup identical images (e.g. a logo on every page)
        try:
            with fitz.open(filepath) as document:
                for page_index in range(len(document)):
                    page = document[page_index]
                    page_area = abs(page.rect.width * page.rect.height) or 1.0
                    page_images: List[Dict[str, Any]] = []
                    for image_index, image_info in enumerate(page.get_images(full=True)):
                        xref = image_info[0]
                        try:
                            pix = fitz.Pixmap(document, xref)
                            # Icons, rules, and bullets carry no text or diagram content.
                            if min(pix.width, pix.height) < _MIN_EMBEDDED_IMAGE_SIDE:
                                continue
                            if pix.n > 4:
                                pix = fitz.Pixmap(fitz.csRGB, pix)
                            image_bytes = pix.tobytes("png")
                            # Deduplicate repeated images (logos/headers) by content hash.
                            image_hash = hashlib.md5(image_bytes).hexdigest()
                            if image_hash in seen_hashes:
                                continue
                            seen_hashes.add(image_hash)
                            # Fraction of the page this image covers — distinguishes a
                            # full-page diagram/drawing from a small logo/icon.
                            coverage = 0.0
                            try:
                                rects = page.get_image_rects(xref)
                                if rects:
                                    coverage = max(abs(r.width * r.height) for r in rects) / page_area
                            except Exception:
                                coverage = (pix.width * pix.height) / page_area
                            page_images.append({
                                "image_index": image_index,
                                "bytes": image_bytes,
                                "hash": image_hash,
                                "coverage": coverage,
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
        cache_key = hashlib.sha1(image_bytes).hexdigest()
        cached = self._ocr_cache.get(cache_key)
        if cached is not None:
            return cached

        text = ""

        if self._ocr_engine == "paddleocr":
            text = self._run_paddleocr(image_bytes)

        if not text:
            text = self._run_tesseract(image_bytes)
        if not text:
            logger.debug("No OCR engines available or OCR processing returned no text. Proceeding without OCR.")

        self._ocr_cache[cache_key] = text
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
                        _PADDLE_OCR_INSTANCE = self._create_paddle_ocr(PaddleOCR)
                        self._paddle_ocr = _PADDLE_OCR_INSTANCE
                    except Exception as exc:
                        logger.warning("PaddleOCR initialization failed: %s. Falling back to Tesseract.", exc)
                        _PADDLE_OCR_UNAVAILABLE = True
                        self._ocr_engine = "tesseract"
                        return ""

        if self._paddle_ocr is None:
            return ""

        # PaddleOCR accepts numpy arrays, paths, and raw bytes — never PIL images.
        image = self._prepare_image_for_paddle(image_bytes)
        try:
            if hasattr(self._paddle_ocr, "predict"):
                results = self._paddle_ocr.predict(image)
            else:
                results = self._paddle_ocr.ocr(image, cls=True)
        except Exception as exc:
            # A single bad image must not disable PaddleOCR for the whole document.
            logger.warning("PaddleOCR inference failed for one image: %s. Trying Tesseract for it.", exc)
            if "PDX has already been initialized" in str(exc):
                _PADDLE_OCR_UNAVAILABLE = True
                self._ocr_engine = "tesseract"
            return ""

        return "\n".join(self._collect_paddle_text(results))

    @staticmethod
    def _create_paddle_ocr(paddle_ocr_cls: Any) -> Any:
        import paddleocr as paddleocr_module

        lang = settings.ocr_lang or "en"
        version = getattr(paddleocr_module, "__version__", "3")

        if version.startswith("2."):
            # PaddleOCR 2.x API. det_limit_side_len matters: the default (960)
            # shrinks full-page renders so far that body text becomes unreadable.
            return paddle_ocr_cls(
                use_angle_cls=True,
                lang=lang,
                use_gpu=False,
                use_space_char=True,
                show_log=False,
                det_limit_side_len=_MAX_OCR_SIDE,
                det_limit_type="max",
            )

        try:
            # enable_mkldnn=False works around a paddlepaddle 3.x oneDNN/PIR bug
            # on Windows CPU ("ConvertPirAttribute2RuntimeAttribute not support")
            # that otherwise fails every inference.
            #
            # The document-orientation and unwarping models are skipped: we deskew
            # in _prepare_image_for_paddle and these scans are flat, so those two
            # extra networks only add per-image latency (and a first-run download)
            # without improving accuracy.
            return paddle_ocr_cls(
                lang=lang,
                use_textline_orientation=True,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                enable_mkldnn=False,
                text_det_limit_side_len=_MAX_OCR_SIDE,
                text_det_limit_type="max",
            )
        except (TypeError, ValueError):
            return paddle_ocr_cls(lang=lang, use_textline_orientation=True)

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

        lang = _TESSERACT_LANG_MAP.get(settings.ocr_lang or "en", settings.ocr_lang or "eng")
        try:
            image = self._prepare_image_for_tesseract(image_bytes)
            if isinstance(image, bytes):
                image = Image.open(io.BytesIO(image))
            return pytesseract.image_to_string(image, lang=lang)
        except Exception as exc:
            logger.warning("Tesseract OCR failed: %s. Skipping OCR for this image.", exc)
            return ""

    def _prepare_image_for_paddle(self, image_bytes: bytes) -> Any:
        """Deskew and resize for PaddleOCR, returning a numpy array.

        PP-OCR models are trained on natural images, so the page is left in
        color/grayscale: hard binarization (helpful for Tesseract) degrades
        Paddle's text detector on photographed scans with uneven lighting.
        """
        try:
            import cv2
            import numpy as np

            array = np.frombuffer(image_bytes, dtype=np.uint8)
            image = cv2.imdecode(array, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Unable to decode image bytes")

            image = self._deskew(image, cv2, np)
            return self._resize_for_ocr(image, cv2)
        except Exception as exc:
            logger.debug("Paddle image preprocessing failed (%s); passing raw bytes.", exc)
            return image_bytes

    def _prepare_image_for_tesseract(self, image_bytes: bytes) -> Any:
        try:
            import cv2
            import numpy as np
            from PIL import Image

            array = np.frombuffer(image_bytes, dtype=np.uint8)
            image = cv2.imdecode(array, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Unable to decode image bytes")

            image = self._deskew(image, cv2, np)
            image = self._resize_for_ocr(image, cv2)

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            denoised = cv2.medianBlur(gray, 3)
            # Adaptive threshold copes with the lighting gradients of photographed
            # pages; the block size must comfortably exceed the stroke width.
            thresh = cv2.adaptiveThreshold(
                denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15
            )
            return Image.fromarray(thresh)
        except Exception:
            try:
                from PIL import Image, ImageOps, ImageFilter

                image = Image.open(io.BytesIO(image_bytes)).convert("L")
                image = image.filter(ImageFilter.MedianFilter(size=3))
                image = ImageOps.autocontrast(image)
                return image
            except Exception as exc:  # pragma: no cover
                logger.warning("Fallback image preprocessing failed: %s", exc)
                return image_bytes

    @staticmethod
    def _resize_for_ocr(image: Any, cv2: Any) -> Any:
        """Keep the long side within [_MIN_OCR_SIDE, _MAX_OCR_SIDE]."""
        height, width = image.shape[:2]
        long_side = max(height, width)
        if long_side > _MAX_OCR_SIDE:
            scale = _MAX_OCR_SIDE / long_side
            return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if long_side < _MIN_OCR_SIDE:
            scale = min(2.0, _MIN_OCR_SIDE / long_side)
            return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        return image

    @staticmethod
    def _deskew(image: Any, cv2: Any, np: Any) -> Any:
        """Straighten slightly rotated scans (phone/CamScanner pages).

        Angle classification in the OCR engines only fixes 90/180° flips, not
        the few-degree tilt that breaks line detection on skewed scans.
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape
            long_side = max(height, width)
            if long_side > 1200:
                scale = 1200 / long_side
                small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            else:
                small = gray

            binary = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            coordinates = cv2.findNonZero(binary)
            if coordinates is None:
                return image

            angle = cv2.minAreaRect(coordinates)[-1]
            if angle > 45:
                angle -= 90
            # Tiny angles are not worth a resample; big ones are usually a
            # mis-estimate (e.g. landscape tables), so leave both alone.
            if not 0.3 < abs(angle) <= 10:
                return image

            center = (width / 2, height / 2)
            rotation = cv2.getRotationMatrix2D(center, angle, 1.0)
            return cv2.warpAffine(
                image,
                rotation,
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(255, 255, 255),
            )
        except Exception:
            return image

    def _extract_tables(self, filepath: str, page_number: int) -> list[Dict[str, Any]]:
        tables: list[Dict[str, Any]] = []

        table_extractors = [self._extract_tables_with_camelot, self._extract_tables_with_pdfplumber]
        for extractor in table_extractors:
            if extractor is self._extract_tables_with_camelot and self._camelot_unavailable:
                continue
            try:
                extracted = extractor(filepath, page_number)
                if extracted:
                    tables.extend(extracted)
                    break
            except ImportError:
                if extractor is self._extract_tables_with_camelot and not self._camelot_unavailable:
                    self._camelot_unavailable = True
                    logger.warning("camelot is not installed: using pdfplumber for table extraction.")
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

    def _extract_diagrams(
        self,
        images: List[Dict[str, Any]],
        tables: list[Dict[str, Any]],
        file_id: Optional[str],
        page_number: int,
    ) -> list[Dict[str, Any]]:
        """Build diagram records for a page's significant embedded images.

        An image is treated as a diagram (engineering drawing, figure, or a
        full-page scan) only when it covers >= ``diagram_min_coverage`` of the
        page — this skips small logos/icons. Each kept image is OCR'd for its
        labels and described, and the image is saved for the viewer. Redundant
        full-page scans whose OCR merely repeats the page text are collapsed
        later by chunk dedup (which links the image to the text chunk).
        """
        diagrams: list[Dict[str, Any]] = []
        if not images:
            return diagrams

        min_coverage = settings.diagram_min_coverage
        for image in images:
            image_bytes = image.get("bytes")
            if not image_bytes:
                continue
            # Small images (logos/icons) are not diagrams. Coverage defaults high
            # so an image whose placement is unknown is kept rather than lost.
            if image.get("coverage", 1.0) < min_coverage:
                continue

            extracted_text = self._run_ocr(image_bytes)
            description = self._describe_diagram(image_bytes, extracted_text)
            if not description:
                continue

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
