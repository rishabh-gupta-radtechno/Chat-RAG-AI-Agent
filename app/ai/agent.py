"""
Lightweight RAG answer agent.
"""

import re
from enum import Enum

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
        """Process question using ReAct pattern."""
        try:
            logger.info("Starting RAG agent for question: %s", question)

            state = {
                "question": question,
                "documents": documents,
                "observation": self._format_context(documents),
                "answer": "",
            }

            extracted_answer = self._extract_procedure_answer(question, documents)
            if extracted_answer:
                state["answer"] = extracted_answer
            else:
                state = await self._respond(state)

            return {
                "answer": state.get("answer", ""),
                "thinking": (
                    f"Used {min(len(documents), settings.rag_context_docs)} "
                    f"of {len(documents)} retrieved document chunks."
                ),
            }

        except Exception as e:
            logger.error(f"Error in ReAct agent: {e}")
            raise

    def _format_context(self, documents: list[dict]) -> str:
        """Format retrieved chunks while keeping the prompt small enough for local LLMs."""
        context_parts = []
        total_chars = 0

        for index, doc in enumerate(documents[: settings.rag_context_docs], start=1):
            chunk_text = (doc.get("chunk_text") or "").strip()
            if not chunk_text:
                continue

            remaining_chars = settings.rag_context_max_chars - total_chars
            if remaining_chars <= 0:
                break

            filename = doc.get("filename", "Unknown")
            chunk = chunk_text[:remaining_chars]
            context_part = f"Source {index} ({filename}):\n{chunk}"
            context_parts.append(context_part)
            total_chars += len(chunk)

        context = "\n\n".join(context_parts)
        logger.info(
            "Observation generated from %s documents, context_chars=%s",
            len(documents),
            len(context),
        )
        return context

    def _extract_procedure_answer(self, question: str, documents: list[dict]) -> str:
        """Extract obvious procedure steps directly from manual text."""
        question_terms = set(self._important_terms(question))
        if not question_terms:
            return ""

        for document in documents:
            chunk_text = (document.get("chunk_text") or "").strip()
            if not chunk_text:
                continue

            normalized_chunk = self._normalize_text(chunk_text)
            matched_terms = [term for term in question_terms if term in normalized_chunk]
            if len(matched_terms) < max(2, len(question_terms) - 1):
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

    @staticmethod
    def _important_terms(text: str) -> list[str]:
        stop_words = {
            "a", "an", "and", "are", "for", "how", "in", "is", "of", "on",
            "or", "the", "to", "what", "when", "where", "which", "with",
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
            r"\b(Remove|Unscrew|Press|Use|Lift|Extract|Take out|Check|Clean|Renew|Fit)\b",
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

        return steps

    async def _respond(self, state: dict) -> dict:
        """Respond step: Generate final answer."""
        try:
            if state.get("observation"):
                prompt = f"""Answer the question using only the provided context.
The question may contain grammar mistakes. Match the important technical terms.
If the context contains a section heading that matches the question, summarize the steps under that section.
Only say the uploaded documents do not provide enough information when no relevant section or details are present.
Keep the answer direct and concise, using numbered steps when the context describes a procedure.

Question: {state['question']}

Context:
{state.get('observation', '')}"""
            else:
                prompt = f"""The user asked: {state['question']}

No relevant uploaded document context was found. Say that the uploaded documents do not provide enough information."""

            answer = await self.llm_client.generate(
                prompt,
                system="You are a careful RAG assistant. Do not invent facts.",
                temperature=0.2,
                top_p=0.9,
            )
            state["answer"] = answer
            logger.info(f"Generated answer: {answer[:100]}...")

            return state

        except Exception:
            logger.exception("Error in respond step")
            state["answer"] = (
                "Unable to generate an answer because the local Ollama model timed out. "
                "Try again after the model finishes loading, reduce retrieved context, or use a smaller chat model."
            )
            return state
