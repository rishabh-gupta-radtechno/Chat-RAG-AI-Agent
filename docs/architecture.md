# Architecture — Chat RAG AI Agent

How a PDF becomes answerable, end to end. This reflects what the code actually
does (not an idealized design). Everything runs **on-prem** — the source manuals
(RDSO / Escorts) are confidential, so no page content ever leaves the host.

## Pipeline

```
PDF
 ↓
process_document
 ↓
PER-PAGE ROUTING  (Docling is NOT the single front door)
  • Native text (pypdf / pdfplumber)  → used when meaningful
  • PaddleOCR (PP-OCRv6)              → when native text is missing/junk (scanned)
  • Docling                          → digital text layout + TableFormer table structure
 ↓
PER-PAGE EXTRACTION  (parallel — one page can emit several types at once)
  • Text     → section-aware chunks (heading-bounded, boilerplate-stripped, mojibake-fixed)
  • Tables   → Docling TableFormer ⊕ pdfplumber/camelot line-refine (digital)
               → validate_table → markdown + semantic-row chunks
  • Charts   → candidate detect (vector-drawings / large raster / chart-like OCR text)
               → region-split → Qwen2.5-VL (Ollama vision)
               → 3 linked chunks: chart_data + chart_summary + chart_image
  • Diagrams → PyMuPDF extract + coverage gate → PaddleOCR (+ optional BLIP caption, OFF)
               → diagram chunk OR linked to a text chunk (related_images)
 ↓
Deduplication  (exact/fuzzy → diagram-containment → embedding-cosine)
 ↓
Embedding — BGE-M3 (Ollama, 1024-d)
 ↓
Qdrant
 ↓
Retriever  (section filter + neighbor pages + content-type bonus + image re-attachment)
 ↓
Answer LLM — local Ollama qwen3:8b   (on-prem only; NOT GPT/cloud)
```

## Stages in detail

### 1. Ingest routing (per page)
There is **no single "PDF → Docling" front door** and **no single layout-classifier
node**. Each page is handled by the cheapest route that yields good text, and the
different content types are detected by different tools in parallel:

- **Native text first** — pypdf/pdfplumber. `_is_meaningful_text` / `_resolve_page_text`
  decide whether the native text layer is real; only then is it used.
- **PaddleOCR (PP-OCRv6)** — fallback for scanned/flattened pages whose native text is
  missing or junk. (`enable_mkldnn=False` is required on this CPU host.)
- **Docling** — used (when `use_docling=true`) for digital text layout and, via
  **TableFormer**, for table structure. A non-Docling "traditional" path also exists.

### 2. Text → chunks
Section-aware chunking, not naive fixed windows: heading-bounded and page-bounded
segments, running header/footer removal, and mojibake repair. Each chunk carries
`section_number` / `section_title` for section-scoped retrieval and citation.

**Two-column reading order:** a multi-column digital page can otherwise be read as one
full-width flow (columns interleaved line by line). `_extract_text_columns` detects a
clear vertical gutter from the text-block bounding boxes and re-reads the page column by
column (band-aware, so full-width headings stay in place). It activates **only** for
pages with a real text layer that are clearly two-column — single-column and scanned
pages fall through to the normal Docling/OCR text unchanged.

### 3. Tables → structured chunks
Docling/TableFormer **detects** table regions (and correctly ignores header/footer
boxes). On **digital** pages with ruled cells, TableFormer's ML cell-mapping can merge
and shift columns, so `_refine_tables_with_lines` swaps each detected table's *content*
for a **line-based extraction** (pdfplumber/camelot read the actual grid lines), but only
when the two match by token overlap (Jaccard ≥ 0.5) — so over-detected header boxes are
never adopted. `validate_table` then rejects paragraph-as-table false positives. Output:
Markdown + semantic key=value rows, large tables split with repeated headers.

**Vision fallback (on-prem, gated):** only the tables rules-based extraction still gets
wrong — ragged/merged digital tables (`_table_is_low_confidence`) or tables on scanned
(OCR-sourced) pages — are re-read by the local vision model (`TableExtractor` →
Qwen2.5-VL), and the result is merged back only when it matches the detected table by
token overlap. Verifiable rules-based tables are kept untouched (no hallucination risk);
controlled by `enable_vision_table_fallback`.

Every table chunk also persists the structured grid as JSON metadata — `table_id`,
`table_header`, `table_rows` (plus `table_title`, `table_markdown`) — the table
counterpart of charts' `structured_data`. The embedded *text* stays Markdown + key=value
(better for retrieval); the JSON is metadata for exact lookups and UI rendering.

### 4. Charts → vision extraction (NOT OCR)
Charts are **not** OCR'd. Candidate pages are detected by any of: enough vector drawing
ops, a large raster image, or chart-like OCR text (many %/number labels). Each candidate
is region-split (a page may hold several charts side by side) and sent to a **local
Qwen2.5-VL** model via Ollama, which returns structured JSON + a human summary. Each chart
produces three linked chunks sharing a `related_chart` id: `chart_data` (structured),
`chart_summary` (prose, embedded for retrieval), and `chart_image` (rendered PNG pointer).
`analyze_many` also splits multiple charts out of a single flattened image.

### 5. Diagrams → OCR (+ optional caption)
Embedded images above a coverage threshold are treated as real diagrams: extracted with
PyMuPDF, de-duplicated by hash, OCR'd with PaddleOCR, and optionally captioned with **BLIP**
(`enable_diagram_captioning`, default **off**). Note: Qwen2.5-VL is used for **charts**, not
diagrams. Diagram images may be stored as their own chunk or linked to the nearest text
chunk via `related_images`.

### 6. Deduplication
Three stages: exact/fuzzy text (normalized hash + token-Jaccard), diagram-containment
suppression (drop a diagram whose text is covered by a text/table chunk), and
embedding-cosine near-duplicate removal — with a content-type priority so the richest
representation survives.

### 7. Embedding → vector store
BGE-M3 via Ollama (1024-dim). Stored in **Qdrant**; the collection auto-recreates on a
vector-dimension mismatch.

### 8. Retrieval → answer
The retriever adds section filtering, neighbor-page expansion, content-type scoring
bonuses, diagram exclusion from text context, and re-attaches chart/diagram images to the
sources (`get_diagrams_for_sources`). The answer is generated by a **local Ollama model
(`qwen3:8b`)** — there is **no GPT/cloud** step, by design.

## Key constraints & design decisions

- **On-prem only.** Confidential RDSO/Escorts documents must never be sent to an external
  API; chart vision (Qwen2.5-VL) and the answer LLM both run on local Ollama.
- **Native-text-first.** OCR is the fallback, not the default — it's slower and lossier.
- **Distributed detection.** No monolithic layout classifier; tables, charts, and diagrams
  are each detected by the tool best suited to them, and a page can emit several types.
- **Digital vs scanned tables.** Line-based extraction fixes digital ruled tables; the
  ragged/merged and scanned cases fall back to the local vision model (gated). Verifiable
  rules-based output is never overwritten by an LLM, so exact engineering values
  (drawing numbers, quantities) aren't put at hallucination risk.

## Relevant config knobs (`app/core/config.py`)

| Setting | Purpose |
|---|---|
| `use_docling`, `docling_do_ocr`, `docling_table_accurate`, `docling_images_scale` | Docling text/table behavior |
| `ocr_engine`, `ocr_lang`, `ocr_full_page_dpi` | PaddleOCR fallback |
| `enable_chart_extraction`, `chart_vision_model`, `chart_candidate_min_drawings`, `chart_render_scale` | Chart vision pipeline (`chart_vision_model` also serves the table fallback) |
| `enable_vision_table_fallback`, `table_low_conf_empty_frac`, `table_low_conf_ragged_frac` | On-prem vision fallback for low-confidence tables |
| `enable_diagram_captioning`, `diagram_min_coverage` | Diagram handling |
| `pdf_chunk_size`, `pdf_chunk_overlap`, `table_rows_per_chunk` | Chunking |
| `dedup_*` | Deduplication thresholds |
| `embedding_model`, `embedding_dimension` | BGE-M3 / Qdrant vector size |
| `ollama_chat_model` | Answer LLM (local) |
```
