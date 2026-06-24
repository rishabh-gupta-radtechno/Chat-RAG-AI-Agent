"""
Table understanding via a local Ollama vision model (on-prem fallback).

Used only when rules-based extraction (Docling + pdfplumber/camelot) produces a
low-confidence table — ragged/merged digital tables or scanned tables. Reads the
page image and returns structured rows. Runs on-prem so confidential pages never
leave the host.
"""

from typing import Any, Dict, List

from app.ai.chart_extractor import ChartExtractor
from app.core.logging import get_logger

logger = get_logger(__name__)


class TableExtractor:
    """Turn a page image into structured tables (header + rows)."""

    PROMPT = (
        "You are reading one image of a document page that contains one or more tables.\n"
        "Extract EVERY data table exactly as printed. Preserve all cell text verbatim — "
        "especially part numbers, drawing numbers, IS codes, dimensions and quantities. "
        "Do NOT invent, complete, or correct any value; if a cell is blank use an empty "
        "string. Keep each row's cells aligned to the correct column, and give every row "
        "the same number of columns as the header.\n"
        "Return ONLY valid JSON (no prose, no markdown fences) in exactly this shape:\n"
        '{"tables": [{"title": "<caption or empty>", '
        '"header": ["col1", "col2"], '
        '"rows": [["c1", "c2"], ["c1", "c2"]]}]}\n'
        'If there are no tables, return {"tables": []}.'
    )

    def __init__(self, llm_client: Any) -> None:
        self._llm = llm_client

    async def analyze(self, image_bytes: bytes) -> List[Dict[str, Any]]:
        """Return every table found in the image; [] on error or when none found."""
        if not image_bytes:
            return []
        try:
            raw = await self._llm.vision(self.PROMPT, image_bytes)
        except Exception as exc:  # vision failure must never abort ingestion
            logger.warning("Table vision call failed: %s", exc)
            return []

        parsed = ChartExtractor._parse_json(raw)
        if not parsed:
            return []

        records: List[Dict[str, Any]] = []
        for table in parsed.get("tables", []) or []:
            if not isinstance(table, dict):
                continue
            header = [str(c).strip() for c in (table.get("header") or [])]
            rows = [
                [str(c).strip() for c in row]
                for row in (table.get("rows") or [])
                if isinstance(row, list)
            ]
            if rows:
                records.append({
                    "title": str(table.get("title") or "").strip() or "Table",
                    "header": header,
                    "rows": rows,
                })
        return records
