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
Embedding — BGE-M3 (Ollama, 1024-d)  (oversized chunks split-and-retry; never dropped)
 ↓
Qdrant  (written in page-windows — incremental, bounded memory; upserts sub-batched + retried)
 ↓
Retriever — HYBRID
  • Dense (semantic) ⊕ BM25 (exact-token) ⊕ keyword ⊕ section  → merged/deduped
  • Cross-encoder reranker reorders the merged pool by true query relevance
  • + section filter + neighbor pages + content-type bonus + image re-attachment
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

**Formula enrichment:** Docling's CodeFormula model converts equations to **LaTeX**
(`docling_formula_enrichment`), so math reaches retrieval as structured text instead of
being flattened or garbled. Adds a model download + ingest time; disable on RAM-bound
hosts.

**Multi-column reading order (N columns):** a multi-column digital page can otherwise be
read as one full-width flow (columns interleaved line by line). `_extract_text_columns`
projects the text-block bounding boxes onto the x-axis, finds the vertical whitespace
gutters (`_detect_column_bands` — works for 2, 3, 4+ columns), and re-reads the page
column by column (`_order_columns`, band-aware so full-width headings stay in place). It
activates **only** for pages with a real text layer split into clear columns —
single-column and scanned pages fall through to the normal Docling/OCR text unchanged.

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

### 4. Figures → one local vision call (charts AND diagrams)
Candidate pages are detected by any of: enough vector drawing ops, a large raster image,
or chart-like OCR text (many %/number labels). Each candidate is region-split (a page may
hold several figures side by side) and each region goes through **one** local Qwen2.5-VL
call (`ChartExtractor.analyze_region`) that **classifies and reads it** — a region is a
chart **XOR** a diagram:

- **Chart** (pie/bar/line) → structured JSON + a human summary. Produces three linked
  chunks sharing a `related_chart` id: `chart_data` (structured), `chart_summary` (prose,
  embedded for retrieval), and `chart_image` (rendered PNG pointer). Handles multiple
  charts in one flattened image.
- **Diagram / schematic** (a labelled drawing/cutaway/flow diagram — often a **vector**
  figure that `get_images()` can't see) → the rendered region is saved and the vision
  model writes a **description** of what it shows; emitted as a `diagram` chunk (the
  description is what makes the figure retrievable). This is why vector engineering
  diagrams are captured at all — `get_images` only sees raster.
- **Table / plain text** → neither (the table/text pipelines own it).

### 5. Raster diagrams → OCR (+ optional caption)
Separately, embedded **raster** images above a coverage threshold are extracted with
PyMuPDF, de-duplicated by hash, OCR'd with PaddleOCR, and optionally captioned with **BLIP**
(`enable_diagram_captioning`, default **off**). The vision figure pass (stage 4) skips a
page that already has a raster diagram, so the same figure isn't captured twice. Diagram
images may be their own chunk or linked to the nearest text chunk via `related_images`.

### 6. Deduplication
Three stages: exact/fuzzy text (normalized hash + token-Jaccard), diagram-containment
suppression (drop a diagram whose text is covered by a text/table chunk), and
embedding-cosine near-duplicate removal — with a content-type priority so the richest
representation survives.

### 7. Embedding → vector store
BGE-M3 via Ollama (1024-dim). Stored in **Qdrant**; the collection auto-recreates on a
vector-dimension mismatch.

**No chunk is silently dropped.** bge-m3's window is 8192 tokens, but earlier the embed
request set no `num_ctx`, so Ollama applied its 2048 default and 500'd on longer chunks
(then skipped them). Now the embed pass declares the real window + `truncate=true`,
estimates tokens (exact HF tokenizer when available, char/word heuristic otherwise), and
**recursively splits oversized chunks and retries** before embedding.

**Incremental, bounded-memory writes.** After all chunks are built and dedup'd
document-wide (so chunk relationships are already resolved), embeddings are computed and
upserted in **page-windows** (`embed_page_batch_size`), flushing after each window — only
one window's embeddings are held at once, and a crash mid-document keeps the windows
already saved. Each upsert is further **sub-batched** (`qdrant_upsert_batch_size`) and
**retried** so a large payload can't drop the Qdrant connection.

### 8. Retrieval → answer (hybrid search + reranking)
Retrieval is **hybrid**, not dense-only: dense-only search misses exact-token lookups (a
specific part/wagon/code that is one row inside a big table, whose averaged embedding ranks
low). For each query the retriever runs **dense (semantic)**, **BM25 (exact-token)**,
**keyword**, and **section** searches, then merges and de-duplicates the pools (diagram and
table-of-contents/index chunks are dropped from text context).

- **BM25** (`rank_bm25`, `enable_bm25_search`) recalls exact-token matches. Its index is
  built once over the whole collection and **cached** (`bm25_cache_ttl_seconds`), and
  **invalidated the moment a document is embedded** (`invalidate_bm25_cache`) so freshly
  synced content is searchable immediately.
- **Cross-encoder reranker** (`enable_reranking`, `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`)
  reorders the merged candidate pool by true query relevance and keeps the top
  `rerank_top_k`. The model loads **once** as a shared singleton (not per query). If the
  reranker is off/unavailable, candidates fall back to the combined lexical+semantic score.
  This is why `table_rows_per_chunk` is kept small — each table chunk stays within the
  reranker's ~512-token window, so a single-row query can actually surface it.

The retriever then adds section filtering, neighbor-page expansion, content-type scoring
bonuses, diagram exclusion from text context, and re-attaches chart/diagram images to the
sources (`get_diagrams_for_sources`). The answer is generated by a **local Ollama model
(`qwen3:8b`)** — there is **no GPT/cloud** step, by design.

## Chat orchestration (query side)
Before retrieval runs, the chat service decides **how much of the conversation to fold into
the query** — history is a help for follow-ups but a liability for fresh questions.

- **Follow-up vs self-contained** (`ChatService._is_followup`): a message is treated as a
  follow-up only when it carries little standalone meaning — almost no content terms, a
  continuation opener ("and…", "what about…", "then…"), or an anaphor ("it/that/this") with
  little else. A self-contained question ("what is the principle of operation?") is **not**
  a follow-up.
- **Retrieval query construction** (`_build_retrieval_query`): for a genuine follow-up, the
  last couple of user turns are prepended to the **keyword/lexical** query so the referent
  resolves. A self-contained question retrieves on **its own terms**, so an earlier,
  unrelated topic can't drag retrieval onto the wrong document. The **embedding always uses
  the bare message** — bge-m3 handles multilingual queries natively, no translation.
- The resolved query then enters the hybrid retriever (stage 8).

## Key constraints & design decisions

- **On-prem only.** Confidential RDSO/Escorts documents must never be sent to an external
  API; chart vision (Qwen2.5-VL) and the answer LLM both run on local Ollama.
- **Device-aware (CPU or GPU), still on-prem.** One resolver (`app/core/device.py`,
  `DEVICE=auto|cpu|cuda`) decides GPU-vs-CPU for *every* torch-backed model — Docling/
  TableFormer, PaddleOCR, BLIP, the cross-encoder reranker — so they all agree. `auto`
  uses CUDA when actually available, else CPU; a CPU-only host runs unchanged. PaddleOCR
  is resolved independently (`paddle_use_gpu`) because PaddlePaddle ships separate CPU/GPU
  wheels. GPU is opt-in via `docker-compose.gpu.yml`; enabling it moves no data off-host.
- **Native-text-first.** OCR is the fallback, not the default — it's slower and lossier.
- **Distributed detection.** No monolithic layout classifier; tables, charts, and diagrams
  are each detected by the tool best suited to them, and a page can emit several types.
- **Digital vs scanned tables.** Line-based extraction fixes digital ruled tables; the
  ragged/merged and scanned cases fall back to the local vision model (gated). Verifiable
  rules-based output is never overwritten by an LLM, so exact engineering values
  (drawing numbers, quantities) aren't put at hallucination risk.
- **Hybrid retrieval over dense-only.** Averaged embeddings bury exact-token lookups (one
  row in a big table); BM25 recalls them and a cross-encoder reranker reorders the merged
  pool. All three models (BGE-M3 embed, BM25, reranker) run **on the host** — the hybrid
  step adds no cloud dependency.
- **One Ollama model resident at a time.** Ingestion runs vision and embedding as separate,
  non-overlapping phases, and chat isn't used during a sync — so holding qwen2.5vl + bge-m3
  + qwen3 together only starves host RAM and gets the vision runner OOM-killed. The Ollama
  container is pinned to `OLLAMA_MAX_LOADED_MODELS=1`, `OLLAMA_KEEP_ALIVE=30s` (unload
  promptly between phases), and `OLLAMA_NUM_PARALLEL=1` (bounds KV-cache memory, which a
  large image balloons).

## Relevant config knobs (`app/core/config.py`)

| Setting | Purpose |
|---|---|
| `device` | GPU/CPU selection for all torch-backed models (`auto`/`cpu`/`cuda`) |
| `use_docling`, `docling_do_ocr`, `docling_table_accurate`, `docling_images_scale` | Docling text/table behavior |
| `docling_formula_enrichment` | Convert equations to LaTeX (CodeFormula) |
| `ocr_engine`, `ocr_lang`, `ocr_full_page_dpi` | PaddleOCR fallback |
| `enable_chart_extraction`, `chart_vision_model`, `chart_candidate_min_drawings`, `chart_render_scale` | Chart vision pipeline (`chart_vision_model` also serves the table fallback) |
| `enable_vision_table_fallback`, `table_low_conf_empty_frac`, `table_low_conf_ragged_frac` | On-prem vision fallback for low-confidence tables |
| `enable_diagram_captioning`, `diagram_min_coverage` | Diagram handling |
| `pdf_chunk_size`, `pdf_chunk_overlap`, `table_rows_per_chunk` | Chunking |
| `dedup_*` | Deduplication thresholds |
| `embedding_model`, `embedding_dimension` | BGE-M3 / Qdrant vector size |
| `embed_num_ctx`, `embed_safety_margin_tokens`, `embed_min_chunk_tokens`, `embed_split_max_depth`, `embed_tokenizer_model` | Oversized-chunk split-and-retry (no chunk dropped) |
| `embed_page_batch_size`, `qdrant_upsert_batch_size`, `qdrant_upsert_max_retries` | Incremental page-window writes + upsert sub-batching/retry |
| `enable_bm25_search`, `enable_reranking`, `rerank_top_k`, `bm25_cache_ttl_seconds` | Hybrid retrieval (BM25 + cross-encoder reranking) |
| `vector_search_top_k`, `similarity_threshold`, `enable_grounding_check` | Dense candidate pool + answer grounding |
| `ollama_num_predict`, `ollama_num_predict_concise`, `ollama_think` | Answer length caps + reasoning toggle |
| `ollama_chat_model` | Answer LLM (local) |
```
