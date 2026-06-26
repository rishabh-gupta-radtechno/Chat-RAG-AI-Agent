"""
Chart understanding via a local Ollama vision model.

Detects whether a rendered page image is a chart (pie/bar/line), extracts its
title, categories and numeric values as structured data, and writes a short
human summary. Runs on-prem so confidential pages never leave the host.
"""

import json
import re
from typing import Any, Dict, List, Optional

from app.core.logging import get_logger

logger = get_logger(__name__)


class ChartExtractor:
    """Turn a chart image into structured data + a human summary."""

    # Shared rules so single and multi prompts extract COMPLETE, well-structured data.
    _DATA_RULES = (
        "Extract EVERY data point you can read — do not omit or summarise any slice, "
        "bar, or marker.\n"
        "Rules for the data list:\n"
        "- Each entry is ONE slice/bar/point: "
        '{"label": "<name>", "value": "<single number or percentage>"}.\n'
        "- value must be a single number or percent string, NEVER a list or array.\n"
        "- Pie: label = slice name, value = its percentage.\n"
        "- Bar/line with x-axis categories: label = the category (e.g. \"SWR\"). "
        "If there are MULTIPLE series (grouped or stacked bars), output one entry "
        'PER bar with label = "<category> - <series name>" (e.g. "SWR - Found Defective").\n'
        "- Read every x-axis tick label and every legend name; do not skip any category.\n"
    )

    PROMPT = (
        "You are analysing one image of a document/slide page.\n"
        "Decide if it primarily shows a data chart (pie, bar, or line).\n"
        + _DATA_RULES +
        "Return ONLY valid JSON (no prose, no markdown fences) in exactly this shape:\n"
        '{"is_chart": true, "chart_type": "pie|bar|line|other", '
        '"title": "<chart title or empty>", '
        '"data": [{"label": "<category or legend>", "value": "<number or percentage>"}], '
        '"summary": "<one or two plain-English sentences naming the dominant values>"}\n'
        'If the image is not a chart, return {"is_chart": false}.'
    )

    PROMPT_MULTI = (
        "You are analysing one image of a document/slide page.\n"
        "It may contain MORE THAN ONE data chart (for example two pie charts side "
        "by side, or a chart next to a table). Identify EVERY distinct data chart "
        "(pie, bar, or line) separately — do not merge them.\n"
        + _DATA_RULES +
        "Return ONLY valid JSON (no prose, no markdown fences) in exactly this shape:\n"
        '{"charts": [{"chart_type": "pie|bar|line|other", '
        '"title": "<that chart\'s title or empty>", '
        '"data": [{"label": "<category or legend>", "value": "<number or percentage>"}], '
        '"summary": "<one or two plain-English sentences naming the dominant values>"}]}\n'
        "Return one object per chart, in reading order. If there are no charts, "
        'return {"charts": []}.'
    )

    # The chart extractor only owns real data charts. If the vision model reports
    # anything else (most often "table", sometimes "other"), it's NOT a chart —
    # the chart's flat {label, value} model can't represent a table and would
    # collapse rows. Such pages belong to the table pipeline, so we drop them here.
    _CHART_TYPES = {"pie", "bar", "line"}

    PROMPT_FIGURE = (
        "You are analysing ONE image: a region of a technical document page. "
        "Classify it into exactly one kind and respond with ONLY valid JSON "
        "(no prose, no markdown fences).\n"
        "1) A DATA CHART (pie, bar, or line graph):\n"
        '   {"kind": "chart", "charts": [{"chart_type": "pie|bar|line", '
        '"title": "<title or empty>", '
        '"data": [{"label": "<category>", "value": "<number or percentage>"}], '
        '"summary": "<one or two sentences naming the dominant values>"}]}\n   '
        + _DATA_RULES +
        "2) An ENGINEERING DIAGRAM / SCHEMATIC / FIGURE (a labelled drawing, cutaway, "
        "assembly, flow or circuit diagram — NOT a data chart):\n"
        '   {"kind": "diagram", "title": "<figure title or empty>", '
        '"description": "<2 to 4 sentences describing what the figure shows: its main '
        'components and how they connect or function>"}\n'
        "3) Mainly a table, plain text, or nothing visual:\n"
        '   {"kind": "none"}'
    )

    def __init__(self, llm_client: Any) -> None:
        self._llm = llm_client

    @classmethod
    def _normalize_chart(cls, chart: Any) -> Optional[Dict[str, Any]]:
        """Normalize one model chart object, or None if it isn't a real chart."""
        if not isinstance(chart, dict) or chart.get("is_chart") is False:
            return None
        chart_type = str(chart.get("chart_type") or "other").strip().lower()
        if chart_type not in cls._CHART_TYPES:
            return None  # tables / "other" belong to other pipelines, not charts
        data = chart.get("data")
        record = {
            "chart_type": chart_type,
            "title": str(chart.get("title") or "").strip(),
            "structured_data": data if isinstance(data, list) else [],
            "summary": str(chart.get("summary") or "").strip(),
        }
        return record if (record["structured_data"] or record["summary"]) else None

    async def analyze(self, image_bytes: bytes) -> Optional[Dict[str, Any]]:
        """Return a chart record, or None if the image is not a chart / on error."""
        if not image_bytes:
            return None
        try:
            raw = await self._llm.vision(self.PROMPT, image_bytes)
        except Exception as exc:  # vision failure must never abort ingestion
            logger.warning("Chart vision call failed: %s", exc)
            return None

        parsed = self._parse_json(raw)
        if not parsed or not parsed.get("is_chart"):
            return None
        return self._normalize_chart(parsed)

    async def analyze_many(self, image_bytes: bytes) -> List[Dict[str, Any]]:
        """Return every chart in the image (handles multiple charts on one page).

        A page may render as a single image (e.g. a flattened slide with two pie
        charts that has no separable vector regions), so we ask the model for a
        list and emit one record per chart. Best-effort: returns [] on error or
        when the image holds no charts.
        """
        if not image_bytes:
            return []
        try:
            raw = await self._llm.vision(self.PROMPT_MULTI, image_bytes)
        except Exception as exc:  # vision failure must never abort ingestion
            logger.warning("Chart vision call failed: %s", exc)
            return []

        parsed = self._parse_json(raw)
        if not parsed:
            return []
        charts = parsed.get("charts")
        if charts is None:
            # Model replied with a single-chart object instead of a list.
            charts = [parsed] if parsed.get("is_chart") else []

        records: List[Dict[str, Any]] = []
        for chart in charts:
            record = self._normalize_chart(chart)
            if record:
                records.append(record)
        return records

    async def analyze_region(self, image_bytes: bytes) -> Dict[str, Any]:
        """Classify and read one figure region in a SINGLE vision call.

        Returns ``{"charts": [...records...], "diagram": {"title","description"}|None}``.
        A data chart yields structured chart records; an engineering diagram/schematic
        yields a written description (far more useful for retrieval than a generic
        caption); tables/plain text yield neither. A region is a chart XOR a diagram.
        """
        result: Dict[str, Any] = {"charts": [], "diagram": None}
        if not image_bytes:
            return result
        try:
            raw = await self._llm.vision(self.PROMPT_FIGURE, image_bytes)
        except Exception as exc:  # vision failure must never abort ingestion
            logger.warning("Figure vision call failed: %s", exc)
            return result

        parsed = self._parse_json(raw)
        if not parsed:
            return result

        for chart in parsed.get("charts") or []:
            record = self._normalize_chart(chart)
            if record:
                result["charts"].append(record)

        if not result["charts"]:
            description = str(parsed.get("description") or "").strip()
            if str(parsed.get("kind") or "").strip().lower() == "diagram" and description:
                result["diagram"] = {
                    "title": str(parsed.get("title") or "").strip(),
                    "description": description,
                }
        return result

    @staticmethod
    def _parse_json(raw: str) -> Optional[Dict[str, Any]]:
        """Parse the model's reply into a dict, tolerating markdown fences/prose."""
        if not raw:
            return None
        text = raw.strip()
        # Strip ```json ... ``` fences if present.
        fence = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
        if fence:
            text = fence.group(1).strip()
        # Otherwise take the outermost JSON object.
        if not text.startswith("{"):
            start, end = text.find("{"), text.rfind("}")
            if start == -1 or end == -1 or end <= start:
                return None
            text = text[start : end + 1]
        try:
            value = json.loads(text)
            return value if isinstance(value, dict) else None
        except (ValueError, TypeError):
            logger.debug("Could not parse chart JSON: %s", raw[:200])
            return None
