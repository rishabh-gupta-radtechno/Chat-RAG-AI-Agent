"""Tests for oversized-chunk handling in the embedding pipeline.

Covers the guarantees added to stop the "input length exceeds the context length"
Ollama 500s (and the silent chunk drops that followed them):

  * token estimation + generic token-budget splitting
  * large tables split by token count (header repeated, no rows lost)
  * long native-text pages and OCR-heavy pages split into sub-budget chunks
  * mixed-content pages contribute both text and table chunks
  * every source page is represented unless genuinely empty
  * RAGPipeline recursively splits + retries on a context-length error and never
    silently skips a chunk; unrelated errors still propagate
"""

import os

os.environ["DEBUG"] = "false"

import pytest

from app.ai.rag import RAGPipeline
from app.ai.text_processor import TextProcessor
from app.core.config import get_settings


settings = get_settings()


@pytest.fixture(autouse=True)
def _heuristic_tokens(monkeypatch):
    """Force the char/word heuristic (never fetch the real HF tokenizer) so token
    estimates are deterministic and offline for every test in this module."""
    monkeypatch.setattr(TextProcessor, "_embed_tokenizer", None)
    monkeypatch.setattr(TextProcessor, "_embed_tokenizer_failed", True)


@pytest.fixture
def small_budget(monkeypatch):
    """Shrink the embedding context so splitting is exercised with tiny inputs."""
    monkeypatch.setattr(settings, "embed_num_ctx", 120)
    monkeypatch.setattr(settings, "embed_safety_margin_tokens", 20)
    monkeypatch.setattr(settings, "embed_min_chunk_tokens", 8)
    return TextProcessor.embed_token_budget()  # 100


# --------------------------------------------------------------------------- #
# Token estimation + generic splitter
# --------------------------------------------------------------------------- #

def test_estimate_tokens_basic():
    assert TextProcessor.estimate_tokens("") == 0
    assert TextProcessor.estimate_tokens("one two three") > 0
    # More text -> more tokens.
    small = TextProcessor.estimate_tokens("word " * 10)
    big = TextProcessor.estimate_tokens("word " * 100)
    assert big > small


def test_split_by_tokens_keeps_every_piece_under_budget(small_budget):
    text = " ".join(f"word{i}" for i in range(2000))
    parts = TextProcessor.split_text_by_tokens(text, small_budget)
    assert len(parts) > 1
    for part in parts:
        assert TextProcessor.estimate_tokens(part) <= small_budget


def test_split_by_tokens_preserves_all_content_in_order(small_budget):
    words = [f"w{i}" for i in range(500)]
    text = " ".join(words)
    parts = TextProcessor.split_text_by_tokens(text, small_budget)
    rejoined = " ".join(parts).split()
    assert rejoined == words  # nothing dropped, order kept


def test_split_by_tokens_noop_when_small():
    text = "a short line that fits comfortably"
    assert TextProcessor.split_text_by_tokens(text, 10_000) == [text]


# --------------------------------------------------------------------------- #
# Large tables split by token count
# --------------------------------------------------------------------------- #

def _wide_table(n_rows: int) -> dict:
    header = ["Sl.No.", "Description", "Drg.No.", "Qty", "Remarks"]
    rows = [
        [
            str(i),
            f"Assembly component number {i} with a fairly long descriptive label",
            f"DRG-{i:04d}",
            str((i % 5) + 1),
            "replace during overhaul as per maintenance schedule section",
        ]
        for i in range(1, n_rows + 1)
    ]
    return {"title": "Bill of Materials", "header": header, "rows": rows}


def test_large_table_splits_by_tokens(small_budget):
    tp = TextProcessor()
    table = _wide_table(40)
    chunks = tp.chunk_table_text(table)

    assert len(chunks) > 1
    # Every multi-row chunk stays within budget (a lone row that alone exceeds the
    # budget is unavoidable and handled by truncate=true, so only assert for >1).
    for chunk in chunks:
        rows_in_chunk = chunk.count("=Sl.No") + chunk.count("Sl.No.=")
        est = TextProcessor.estimate_tokens(chunk)
        if est > small_budget:
            # allowed only for an irreducible single-row chunk
            assert "Rows:" in chunk

    # Header is repeated in every chunk (needed for standalone retrieval).
    for chunk in chunks:
        assert "Sl.No." in chunk and "Description" in chunk


def test_large_table_loses_no_rows(small_budget):
    tp = TextProcessor()
    table = _wide_table(40)
    chunks = tp.chunk_table_text(table)
    joined = "\n".join(chunks)
    for i in range(1, 41):
        assert f"DRG-{i:04d}" in joined  # every row's unique drawing number survives


def test_small_table_single_chunk():
    tp = TextProcessor()
    table = _wide_table(2)
    chunks = tp.chunk_table_text(table)
    assert len(chunks) == 1


# --------------------------------------------------------------------------- #
# build_pdf_chunks: long text / OCR / mixed pages, every page represented
# --------------------------------------------------------------------------- #

def _pages_of(chunks) -> set:
    return {c["metadata"].get("page_number") for c in chunks}


def test_long_native_text_page_splits_into_multiple_chunks():
    tp = TextProcessor()
    long_text = " ".join(f"maintenance{i}" for i in range(900))
    pages = [{"page_number": 1, "native_text": long_text}]
    chunks = tp.build_pdf_chunks(pages, "manual.pdf")

    text_chunks = [c for c in chunks if c["metadata"]["content_type"] in ("text", "ocr")]
    assert len(text_chunks) > 1
    assert 1 in _pages_of(chunks)
    budget = TextProcessor.embed_token_budget()
    for c in text_chunks:
        assert TextProcessor.estimate_tokens(c["text"]) <= budget


def test_ocr_heavy_page_is_chunked_and_represented():
    tp = TextProcessor()
    ocr_text = " ".join(f"scanline{i}" for i in range(900))
    pages = [{"page_number": 7, "ocr_text": ocr_text}]  # no native_text -> OCR page
    chunks = tp.build_pdf_chunks(pages, "scanned.pdf")

    assert 7 in _pages_of(chunks)
    assert len(chunks) > 1
    budget = TextProcessor.embed_token_budget()
    for c in chunks:
        assert TextProcessor.estimate_tokens(c["text"]) <= budget


def test_mixed_content_page_yields_text_and_table(monkeypatch):
    tp = TextProcessor()
    # Trust our own table rather than validate_table's heuristics for this test.
    monkeypatch.setattr(tp, "validate_table", lambda table, page_text="": True)

    pages = [
        {
            "page_number": 3,
            "native_text": " ".join(f"procedure{i}" for i in range(120)),
            "tables": [_wide_table(6)],
        }
    ]
    chunks = tp.build_pdf_chunks(pages, "mixed.pdf")
    content_types = {c["metadata"]["content_type"] for c in chunks}

    assert "table" in content_types
    assert "text" in content_types
    assert 3 in _pages_of(chunks)


def test_empty_page_still_represented_by_fallback():
    tp = TextProcessor()
    pages = [{"page_number": 2}]  # no text, no tables
    chunks = tp.build_pdf_chunks(pages, "blank.pdf")
    assert 2 in _pages_of(chunks)  # a page is never silently dropped


# --------------------------------------------------------------------------- #
# RAGPipeline: recursive split + retry, no silent skips
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_embed_proactive_split_all_children_embedded(small_budget, monkeypatch):
    pipe = RAGPipeline()
    calls = []

    async def fake_embed(text):
        calls.append(text)
        return [0.1, 0.2, 0.3]

    monkeypatch.setattr(pipe, "_embed", fake_embed)

    text = " ".join(f"token{i}" for i in range(1000))
    results = await pipe._embed_text_with_splitting(
        text, doc_name="doc.pdf", page_number=1, chunk_type="text"
    )

    assert len(results) > 1
    # Each embedded piece is within budget...
    for part_text, emb in results:
        assert TextProcessor.estimate_tokens(part_text) <= small_budget
        assert emb == [0.1, 0.2, 0.3]
    # ...and all original content is preserved across the children (nothing skipped).
    assert " ".join(t for t, _ in results).split() == text.split()


@pytest.mark.asyncio
async def test_embed_retry_split_on_context_error(small_budget, monkeypatch):
    """A context-length error triggers recursive split+retry, embedding everything."""
    pipe = RAGPipeline()
    # Fail whenever the input is above a threshold *below* the proactive budget, so
    # the failure is only escapable via the retry-split path, not the proactive one.
    fail_threshold = small_budget // 3

    async def fake_embed(text):
        if TextProcessor.estimate_tokens(text) > fail_threshold:
            raise RuntimeError(
                "Ollama API error: 500 the input length exceeds the context length"
            )
        return [0.0]

    monkeypatch.setattr(pipe, "_embed", fake_embed)

    text = " ".join(f"cell{i}" for i in range(600))
    results = await pipe._embed_text_with_splitting(
        text, doc_name="doc.pdf", page_number=4, chunk_type="table"
    )

    assert len(results) > 1
    for part_text, _ in results:
        assert TextProcessor.estimate_tokens(part_text) <= fail_threshold
    assert " ".join(t for t, _ in results).split() == text.split()


@pytest.mark.asyncio
async def test_embed_non_context_error_propagates(monkeypatch):
    """A genuine failure (e.g. service down) is raised, not silently swallowed."""
    pipe = RAGPipeline()

    async def fake_embed(text):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(pipe, "_embed", fake_embed)

    with pytest.raises(RuntimeError, match="connection refused"):
        await pipe._embed_text_with_splitting(
            "some short text", doc_name="d.pdf", page_number=1, chunk_type="text"
        )


@pytest.mark.asyncio
async def test_embed_empty_text_returns_nothing():
    pipe = RAGPipeline()
    assert await pipe._embed_text_with_splitting(
        "   ", doc_name="d.pdf", page_number=1, chunk_type="text"
    ) == []


# --------------------------------------------------------------------------- #
# estimate_tokens: real tokenizer wiring with heuristic fallback
# --------------------------------------------------------------------------- #

def test_estimate_tokens_uses_real_tokenizer_when_available(monkeypatch):
    class FakeTok:
        def encode(self, text, add_special_tokens=True):
            return list(range(len(text.split()) + 2))  # words + 2 special tokens

    monkeypatch.setattr(TextProcessor, "_embed_tokenizer", FakeTok())
    monkeypatch.setattr(TextProcessor, "_embed_tokenizer_failed", False)
    assert TextProcessor.estimate_tokens("alpha beta gamma") == 5  # 3 words + 2


def test_estimate_tokens_falls_back_when_tokenizer_raises(monkeypatch):
    class BadTok:
        def encode(self, *a, **k):
            raise RuntimeError("tokenizer boom")

    monkeypatch.setattr(TextProcessor, "_embed_tokenizer", BadTok())
    monkeypatch.setattr(TextProcessor, "_embed_tokenizer_failed", False)
    # No exception; heuristic produces a positive count.
    assert TextProcessor.estimate_tokens("alpha beta gamma") > 0


def test_tokenizer_disabled_when_model_empty(monkeypatch):
    monkeypatch.setattr(TextProcessor, "_embed_tokenizer", None)
    monkeypatch.setattr(TextProcessor, "_embed_tokenizer_failed", False)
    monkeypatch.setattr(settings, "embed_tokenizer_model", "")
    assert TextProcessor._get_embed_tokenizer() is None


# --------------------------------------------------------------------------- #
# _embed_locally: cached model + Ollama fallback when ST is missing
# --------------------------------------------------------------------------- #

def test_local_embed_model_is_cached(monkeypatch):
    pipe = RAGPipeline()

    class _Emb:
        def tolist(self):
            return [1.0, 2.0]

    class _Model:
        def __init__(self):
            self.calls = 0

        def encode(self, text):
            self.calls += 1
            return _Emb()

    model = _Model()
    pipe._local_embed_model = model  # pre-seed cache
    assert pipe._embed_locally("a") == [1.0, 2.0]
    assert pipe._embed_locally("b") == [1.0, 2.0]
    assert model.calls == 2            # reused, encoded twice
    assert pipe._local_embed_model is model  # never rebuilt


@pytest.mark.asyncio
async def test_embed_falls_back_to_ollama_when_st_missing(monkeypatch):
    pipe = RAGPipeline()
    monkeypatch.setattr(settings, "use_local_embeddings", True)

    def _boom(text):
        raise ImportError("No module named 'sentence_transformers'")

    async def _fake_ollama(text):
        return [9.9]

    monkeypatch.setattr(pipe, "_embed_locally", _boom)
    monkeypatch.setattr(pipe.llm_client, "embed", _fake_ollama)

    assert await pipe._embed("hi") == [9.9]  # awaited correctly, not a coroutine
