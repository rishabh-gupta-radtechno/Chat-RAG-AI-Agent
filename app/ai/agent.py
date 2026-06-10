"""
Lightweight RAG answer agent.
"""

import re
from enum import Enum
from typing import Optional

from app.ai.llm import OllamaClient
from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


class AgentAction(Enum):
    """Agent actions."""

    RETRIEVE = "retrieve"
    REASON = "reason"
    RESPOND = "respond"


class ReActAgent:
    """RAG-based question answering agent."""

    def __init__(self, rag_pipeline):
        self.rag_pipeline = rag_pipeline
        self.llm_client = OllamaClient()

    async def process(
        self,
        question: str,
        documents: list[dict],
    ) -> dict:
        """Process question using ReAct pattern with hallucination guard."""
        try:
            logger.info("="*70)
            logger.info("STARTING RAG AGENT PROCESS")
            logger.info("="*70)
            logger.info("Question: %s", question)
            logger.info("Documents retrieved: %d", len(documents))
            logger.info("="*70)

            # First: Check if retrieval is grounded in question entities
            from app.ai.rag import RAGPipeline
            is_grounded, grounding_msg = RAGPipeline.validate_retrieval_grounding(question, documents)
            logger.info(f"Retrieval grounding check: {grounding_msg}")
            
            state = {
                "question": question,
                "documents": documents,
                "observation": self._format_context(documents),
                "answer": "",
                "grounded": is_grounded,
            }

            # If retrieval is not grounded, return limited answer without hallucination
            if not is_grounded:
                logger.warning("Retrieved documents not grounded in question entities; limiting answer")
                state["answer"] = f"I couldn't find specific information about the requested topic in the available documents."
                return {
                    "answer": state.get("answer", ""),
                    "thinking": (
                        f"Retrieved {len(documents)} documents but none contain the specific "
                        f"entities from the question. Preventing hallucination by not generating answer."
                    ),
                }

            extracted_answer = self._extract_procedure_answer(question, documents)
            if extracted_answer:
                logger.info("Extracted direct procedure answer from documents")
                state["answer"] = extracted_answer
            else:
                logger.info("No direct procedure answer found, using LLM to generate response")
                state = await self._respond(state)

            logger.info("="*70)
            logger.info("AGENT PROCESS COMPLETE")
            logger.info("="*70)
            
            return {
                "answer": state.get("answer", ""),
                "thinking": (
                f"Used {min(len(state.get('documents', [])), settings.rag_context_docs)} "
                    f"of {len(documents)} retrieved document chunks."
                ),
            }

        except Exception as e:
            logger.error(f"Error in ReAct agent: {e}")
            raise

    def _format_context(self, documents: list[dict], max_chars: Optional[int] = None) -> str:
        """Format retrieved chunks while keeping the prompt small enough for local LLMs."""
        context_parts = []
        total_chars = 0
        char_limit = max_chars or settings.rag_context_max_chars
        
        logger.info("Formatting context from %d documents (max %d docs, %d char limit)", 
                   len(documents), settings.rag_context_docs, char_limit)

        for index, doc in enumerate(documents[: settings.rag_context_docs], start=1):
            chunk_text = (doc.get("chunk_text") or "").strip()
            if not chunk_text:
                logger.debug("Skipping empty chunk at index %d", index)
                continue

            remaining_chars = char_limit - total_chars
            if remaining_chars <= 0:
                logger.info("Reached character limit, stopping context formatting")
                break

            filename = doc.get("filename", "Unknown")
            page_number = doc.get("page_number", 0)
            document_page_number = doc.get("document_page_number")
            content_type = doc.get("content_type", "text")
            chunk = chunk_text[:remaining_chars]
            page_label = f"page {page_number}"
            if document_page_number and document_page_number != page_number:
                page_label = f"page {page_number}, document page {document_page_number}"
            context_part = f"Source {index} ({filename}, {page_label}, {content_type}):\n{chunk}"
            context_parts.append(context_part)
            total_chars += len(chunk)
            
            logger.debug("Added source %d: %s (%s) - chunk_length=%d", 
                        index, filename, page_label, len(chunk))

        context = "\n\n".join(context_parts)
        logger.info(
            "Formatted context from %s documents: total_chars=%s, formatted_context_length=%s",
            len(documents),
            total_chars,
            context,
            len(context),
        )
        return context

    def _extract_procedure_answer(self, question: str, documents: list[dict]) -> str:
        """Extract obvious procedure steps directly from manual text."""
        question_terms = set(self._important_terms(question))
        if not question_terms:
            return ""

        exact_section = self._extract_exact_section_answer(question, documents)
        if exact_section:
            return exact_section

        for document in documents:
            chunk_text = (document.get("chunk_text") or "").strip()
            if not chunk_text:
                continue

            normalized_chunk = self._normalize_text(chunk_text)
            matched_terms = [term for term in question_terms if term in normalized_chunk]
            required_terms = 1 if len(question_terms) == 1 else max(2, len(question_terms) - 1)
            if len(matched_terms) < required_terms:
                continue

            section_text = self._section_text_after_heading(chunk_text, question_terms)
            steps = self._procedure_steps(section_text)
            if steps:
                heading = " ".join(term.capitalize() for term in self._important_terms(question))
                numbered_steps = "\n".join(
                    f"{index}. {step}" for index, step in enumerate(steps, start=1)
                )
                return f"{heading}:\n{numbered_steps}"

        return ""

    def _extract_exact_section_answer(self, question: str, documents: list[dict]) -> str:
        """Return a matching manual section directly, preserving wording as much as possible."""
        query_terms = self._important_terms(question)
        if not query_terms:
            return ""

        # This path is intended for heading-style questions such as "REASSEMBLING".
        if len(query_terms) > 3:
            return ""

        ordered_chunks = sorted(
            documents,
            key=lambda doc: (
                str(doc.get("filename") or doc.get("file_name") or ""),
                int(doc.get("page_number") or 0),
                int(doc.get("chunk_index") or 0),
            ),
        )

        best_section = ""
        best_score = -1
        for document in ordered_chunks:
            text = self._document_text(document)
            if not text:
                continue

            start = self._find_best_section_heading(text, query_terms)
            if start < 0:
                continue

            section = text[start:]
            stop = self._find_next_section_heading(section)
            if stop > 0:
                section = section[:stop]

            cleaned_section = self._clean_extracted_section(section)
            if len(cleaned_section.split()) < 12:
                continue

            score = self._procedure_signal_score(section)
            content_type = str(document.get("content_type") or "").lower()
            if "ocr" in content_type:
                score -= 1

            if score > best_score:
                best_score = score
                best_section = cleaned_section

        combined = "\n".join(
            self._document_text(document)
            for document in ordered_chunks
            if self._document_text(document)
        )
        combined_section = ""
        combined_score = -1
        if combined.strip():
            start = self._find_best_section_heading(combined, query_terms)
            if start >= 0:
                section = combined[start:]
                stop = self._find_next_section_heading(section)
                if stop > 0:
                    section = section[:stop]
                cleaned_section = self._clean_extracted_section(section)
                if len(cleaned_section.split()) >= 12:
                    combined_section = cleaned_section
                    combined_score = self._procedure_signal_score(section)

        if combined_section and combined_score >= best_score:
            return combined_section
        return best_section

    @staticmethod
    def _document_text(document: dict) -> str:
        return (
            document.get("chunk_text")
            or document.get("diagram_description")
            or document.get("description")
            or ""
        ).strip()

    def _find_best_section_heading(self, text: str, query_terms: list[str]) -> int:
        candidates = self._find_section_headings(text, query_terms)
        if not candidates:
            return -1

        best_offset = candidates[0]
        best_score = -1
        for offset in candidates:
            preview = self._section_preview(text, offset)
            score = self._procedure_signal_score(preview)
            if score > best_score:
                best_offset = offset
                best_score = score

        return best_offset

    def _section_preview(self, text: str, offset: int) -> str:
        preview = text[offset : offset + 2500]
        stop = self._find_next_section_heading(preview)
        if stop > 0:
            return preview[:stop]
        return preview

    def _find_section_headings(self, text: str, query_terms: list[str]) -> list[int]:
        lines = text.splitlines()
        offset = 0
        offsets = []
        for line in lines:
            if self._line_matches_heading(line, query_terms):
                offsets.append(offset)
            offset += len(line) + 1
        return offsets

    @staticmethod
    def _procedure_signal_score(text: str) -> int:
        command_pattern = re.compile(
            r"\b(Clamp|Place|Lower|Screw|Drive|Fix|Insert|Release|Remove|Unscrew|Fit)\b",
            flags=re.IGNORECASE,
        )
        figure_score = len(re.findall(r"\bFig\.?\s*\d+", text, flags=re.IGNORECASE))
        return len(command_pattern.findall(text)) + figure_score

    def _line_matches_heading(self, line: str, query_terms: list[str]) -> bool:
        normalized_line = self._normalize_heading_text(line)
        if not normalized_line:
            return False

        matches = sum(
            1
            for term in query_terms
            if self._normalize_heading_text(term) in normalized_line
        )
        if matches == len(query_terms):
            return True

        # OCR often splits or mutates headings: "REASSENBL ING" for "REASSEMBLING".
        if len(query_terms) == 1:
            compact_query = self._normalize_heading_text(query_terms[0])
            return self._is_close_heading(compact_query, normalized_line)

        return False

    @staticmethod
    def _normalize_heading_text(text: str) -> str:
        normalized = re.sub(r"[^a-z0-9]+", "", text.lower())
        return normalized.replace("reassenbling", "reassembling")

    @staticmethod
    def _is_close_heading(query: str, normalized_line: str) -> bool:
        if query in normalized_line:
            return True
        words = re.findall(r"[a-z0-9]+", normalized_line)
        return any(ReActAgent._levenshtein_distance(query, word) <= 2 for word in words)

    @staticmethod
    def _levenshtein_distance(left: str, right: str) -> int:
        if abs(len(left) - len(right)) > 2:
            return 3
        previous = list(range(len(right) + 1))
        for i, left_char in enumerate(left, start=1):
            current = [i]
            for j, right_char in enumerate(right, start=1):
                current.append(
                    min(
                        previous[j] + 1,
                        current[j - 1] + 1,
                        previous[j - 1] + (left_char != right_char),
                    )
                )
            previous = current
        return previous[-1]

    @staticmethod
    def _find_next_section_heading(text: str) -> int:
        matches = list(
            re.finditer(
                r"(?m)^\s*(?:\d+\.\d+|[A-Za-z]\.)\s+[A-Z][A-Z0-9 ,/&().-]{4,}\s*$|^\s*(?!FIG(?:URE)?|NOTE)[A-Z0-9 ,/&().-]{8,}\s*$",
                text,
            )
        )
        return matches[1].start() if len(matches) > 1 else -1

    @staticmethod
    def _clean_extracted_section(text: str) -> str:
        lines = []
        for raw_line in text.splitlines():
            line = re.sub(r"\s+", " ", raw_line).strip()
            if not line:
                continue
            if re.fullmatch(r"\d{6,}", line):
                continue
            lines.append(line)

        # Remove duplicate lines while preserving order (helps with OCR / chunk overlap)
        seen = set()
        deduped_lines = []
        for l in lines:
            key = l.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped_lines.append(l)

        cleaned = "\n".join(deduped_lines)
        # Collapse long gaps and normalize spacing
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

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

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join(re.findall(r"[a-z0-9]+", text.lower()))

    def _section_text_after_heading(self, text: str, question_terms: set[str]) -> str:
        normalized_text = self._normalize_text(text)
        positions = [
            normalized_text.find(term)
            for term in question_terms
            if normalized_text.find(term) >= 0
        ]
        if not positions:
            return text

        # Map approximately back to original text by searching the earliest matched term.
        earliest_term = min(
            question_terms,
            key=lambda term: normalized_text.find(term)
            if normalized_text.find(term) >= 0
            else len(normalized_text),
        )
        original_position = text.lower().find(earliest_term)
        if original_position < 0:
            return text

        return text[original_position:]

    @staticmethod
    def _procedure_steps(text: str) -> list[str]:
        command_pattern = re.compile(
            r"\b(Remove|Unscrew|Press|Use|Lift|Extract|Take out|Check|Clean|Renew|Fit|Clamp|Place|Lower|Screw|Drive|Fix|Insert|Release)\b",
        )

        matches = list(command_pattern.finditer(text))
        steps = []
        for index, match in enumerate(matches):
            start = match.start()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            step = text[start:end].strip(" .,-")
            step = re.sub(r"\s+", " ", step)

            if len(step) < 8:
                continue
            if re.match(r"^\d+(\s+\d+)*$", step):
                continue
            steps.append(step)
            if len(steps) >= 8:
                break

        # Deduplicate near-duplicate steps while preserving order
        deduped = []
        seen_norm = set()
        for s in steps:
            norm = re.sub(r"[^a-z0-9]+", "", s.lower())
            if norm in seen_norm:
                continue
            seen_norm.add(norm)
            deduped.append(s)

        return deduped

    async def _respond(self, state: dict) -> dict:
        """Respond step: Generate final answer."""
        try:
            if state.get("observation"):
                logger.info("="*50)
                logger.info("CONTEXT FOR QUESTION:")
                logger.info("="*50)
                logger.info("Question: %s", state['question'])
                logger.info("-"*50)
                logger.info("Retrieved Context:\n%s", state.get('observation', ''))
                logger.info("-"*50)
                
                # Fix #4: Stronger System Prompt
                prompt = f"""
You are a technical document assistant.

Answer using ONLY the provided context.
Do NOT use outside knowledge.

Ignore OCR noise, repeated headers, footers, page titles, and document metadata.

If information is spread across multiple sections, combine it into a single coherent answer.

If the context describes a procedure:
- Present it as numbered steps.
- Include all relevant values, limits, pressures, dimensions, and part numbers.

If technical terms appear misspelled due to OCR, use the most likely correct technical spelling.

Do not copy raw OCR text.
Rewrite it into clear professional language.

If the answer cannot be determined from the provided context, respond:
"The documents do not provide enough information."

Question: {state['question']}

Context:
{state.get('observation', '')}"""
                
                logger.info("="*50)
                logger.info("FULL PROMPT SENT TO LLM:")
                logger.info("="*50)
                logger.info("\n%s", prompt)
                logger.info("="*50)
            else:
                prompt = f"""The user asked: {state['question']}

No relevant uploaded document context was found. Say that the uploaded documents do not provide enough information."""
                logger.info("No context found. Using fallback prompt.")
                logger.info("Prompt: %s", prompt)

            answer = await self.llm_client.generate(
                prompt,
                system="You are a careful RAG assistant. Do not invent facts.",
                temperature=0.2,
                top_p=0.9,
            )
            state["answer"] = answer
            logger.info("="*50)
            logger.info("Generated answer: %s", answer)
            logger.info("="*50)

            return state

        except Exception:
            logger.exception("Error in respond step")
            state["answer"] = (
                "Unable to generate an answer because the local Ollama model timed out. "
                "Try again after the model finishes loading, reduce retrieved context, or use a smaller chat model."
            )
            return state
