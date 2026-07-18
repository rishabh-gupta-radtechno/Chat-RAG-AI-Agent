"""
Chat and RAG service with LLM integration.
"""
from datetime import datetime

import json
import re
import uuid
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.rag import RAGPipeline
from app.ai.agent import ReActAgent
from app.ai.llm import OllamaClient
from app.core.config import get_settings
from app.core.logging import get_logger
from app.repositories.chat import ChatHistoryRepository
from app.schemas import (
    ChatResponse,
    ConversationChatResponse,
    ConversationSummaryResponse,
    ConversationTurnResponse,
    DiagramReference,
    SourceReference,
)

logger = get_logger(__name__)
settings = get_settings()


class ChatService:
    """Chat service with RAG and agent integration."""

    def __init__(self, session: AsyncSession):
        self.session = session
        self.chat_repo = ChatHistoryRepository(session)
        self.rag_pipeline = RAGPipeline()
        self.agent = ReActAgent(self.rag_pipeline)
        self.llm_client = OllamaClient()

    async def ask_question(
        self,
        user_id: uuid.UUID,
        question: str,
    ) -> ChatResponse:
        """Process a question with RAG and agent."""
        response = await self.chat(user_id=user_id, message=question)
        return ChatResponse(
            answer=response.answer,
            sources=response.sources,
            diagrams=response.diagrams,
            model=response.model,
            # thinking=response.thinking,
        )

    async def chat(
        self,
        user_id: uuid.UUID,
        message: str,
        conversation_id: Optional[uuid.UUID] = None,
    ) -> ConversationChatResponse:
        """Process a multi-turn chat message with RAG and conversation memory."""
        logger.info(f"Processing chat message from user {user_id}: {message}")

        try:
            conversation_id = conversation_id or uuid.uuid4()

            history = await self.chat_repo.get_recent_by_conversation(
                user_id=user_id,
                conversation_id=conversation_id,
                limit=6,
            )

            retrieval_query, embed_texts = await self._build_retrieval_query(message, history)

            # Retrieve relevant documents
            documents = await self.rag_pipeline.retrieve(retrieval_query, user_id=user_id, embed_queries=embed_texts)
            logger.info(f"Retrieved {len(documents)} documents")
            diagram_user_id = None if any(
                doc.get("retrieval_scope") == "global_fallback" for doc in documents
            ) else str(user_id)
            diagrams = await self.rag_pipeline.vector_db.get_diagrams_for_sources(
                documents,
                user_id=diagram_user_id,
            )
            logger.info(f"Retrieved {len(diagrams)} related diagrams")

            # Process with conversation-aware RAG
            response = await self._generate_conversation_answer(
                message=message,
                documents=documents,
                diagrams=diagrams,
                history=history,
            )
            logger.info("Agent generated response")

            # Extract sources from documents
            sources = [
                SourceReference(
                    filename=doc.get("filename", "Unknown"),
                    filepath=doc.get("filepath"),
                    file_id=uuid.UUID(doc.get("file_id", "00000000-0000-0000-0000-000000000000")),
                    chunk_index=doc.get("chunk_index", 0),
                    relevance_score=doc.get("relevance_score", 0.0),
                    page_number=doc.get("page_number", 0) or 0,
                    document_page_number=doc.get("document_page_number"),
                    content_type=doc.get("content_type"),
                    excerpt=doc.get("chunk_text", ""),  # TEMP: full chunk text
                )
                for doc in documents
            ]
            sources = sorted(sources, key=lambda source: source.relevance_score, reverse=True)[:5]

            diagram_references = [
                DiagramReference(
                    filename=diagram.get("filename", "Unknown"),
                    file_id=uuid.UUID(diagram.get("file_id", "00000000-0000-0000-0000-000000000000")),
                    page_number=diagram.get("page_number", 0) or 0,
                    document_page_number=diagram.get("document_page_number"),
                    image_index=diagram.get("image_index"),
                    description=diagram.get("diagram_description") or diagram.get("chunk_text", ""),
                    image_url=diagram.get("image_url"),
                    relevance_score=diagram.get("relevance_score", 0.0),
                )
                for diagram in diagrams
            ]

            if response.get("answer", "").strip() == self._not_found_answer(message):
                sources = []
                diagram_references = []

            # Store chat history
            await self.chat_repo.create_with_sources(
                user_id=user_id,
                conversation_id=conversation_id,
                question=message,
                answer=response.get("answer", ""),
                sources=[s.model_dump(mode="json") for s in sources],
                diagrams=[d.model_dump(mode="json") for d in diagram_references],
                model=settings.ollama_chat_model,
            )
            await self.chat_repo.commit()

            return ConversationChatResponse(
                conversation_id=conversation_id,
                answer=response.get("answer", ""),
                sources=sources,
                diagrams=diagram_references,
                model=settings.ollama_chat_model,
                # thinking=response.get("thinking"),
            )

        except Exception as e:
            logger.error(f"Error processing question: {e}")
            raise

    # Markers that a message leans on the previous turn for its meaning.
    _ANAPHORA_RE = re.compile(
        r"\b(it|its|it's|that|this|these|those|them|they|their|same|the above|above)\b",
        re.IGNORECASE,
    )
    _CONTINUATION_RE = re.compile(
        r"^\s*(and|also|what about|then|so|ok|okay|but)\b",
        re.IGNORECASE,
    )
    _DEVANAGARI_WORD_RE = re.compile(r"[ऀ-ॿ]+")

    @classmethod
    def _is_followup(cls, message: str) -> bool:
        """True when retrieval should lean on prior turns to resolve the message.

        A follow-up carries little standalone content: it has almost no content
        terms, opens with a continuation word ("and…", "what about…"), or uses an
        anaphor ("it/that/this") with little else. A self-contained question such
        as "what is the principle of operation?" is NOT a follow-up, so its
        retrieval stays independent of the previous (possibly unrelated) topic and
        can land in any document.

        _important_terms() only recognises [a-z0-9] words, so a Devanagari question
        has near-zero "terms" by that count alone — every Hindi question, however
        detailed and self-contained, would otherwise look like a bare anaphoric
        follow-up and get the previous (possibly unrelated) turns spliced into its
        retrieval query. Count Devanagari word runs alongside the ASCII terms so a
        substantive Hindi question is recognised as self-contained too.
        """
        text = (message or "").strip()
        if not text:
            return False
        content_term_count = len(cls._important_terms(text)) + len(cls._DEVANAGARI_WORD_RE.findall(text))
        if content_term_count <= 1:
            return True
        if cls._CONTINUATION_RE.search(text):
            return True
        if cls._ANAPHORA_RE.search(text) and content_term_count <= 3:
            return True
        return False

    async def _build_retrieval_query(self, message: str, history: list) -> tuple[str, list[str]]:
        """Build (keyword_query, embed_texts) for retrieval.

        Conversation history is prepended to the keyword/lexical query ONLY for
        genuine follow-up questions (see _is_followup); a self-contained question
        retrieves on its own terms so an earlier, unrelated topic cannot drag
        retrieval onto the wrong document.

        A Devanagari question is translated to English before retrieval. bge-m3
        embeds Hindi natively, but the LEXICAL half of retrieval cannot: the corpus
        is English and _keyword_terms() keeps only [a-z0-9], so a Hindi query reduces
        to [] — which returns zero keyword hits AND zeroes lexical_score for every
        document — while BM25 tokenizes to the stray digits alone ('23', '79'), which
        matches noise rather than content. Section/heading rescues are likewise blind
        to Devanagari. So the keyword_query is always the English text.

        The EMBEDDING half is different: rather than embed the English translation
        alone, a Hindi question is embedded in BOTH languages (English translation +
        original Hindi) and each is searched, then merged. bge-m3 aligns Hindi and
        English in one space, so the Hindi vector recovers meaning a lossy/awkward
        translation drops (and vice versa) — two shots at dense recall instead of one.
        English questions embed once (no translation call). Answer generation still
        runs on the original message, so replies stay in Hindi.
        """
        query = message
        if history and self._is_followup(message):
            prior_questions = [
                turn.question.strip()
                for turn in history[-2:]
                if getattr(turn, "question", "").strip()
            ]
            if prior_questions:
                logger.info(
                    "Follow-up detected; expanding retrieval query with %d prior turn(s)",
                    len(prior_questions),
                )
                # Expand BOTH the keyword query and the embedding: a bare anaphoric
                # follow-up ("how does it do that?") embeds too generically and pulls
                # any manual's "how it works" content, so ground the vector search in
                # the conversation's subject too.
                query = "\n".join(prior_questions + [message])

        if not self._contains_devanagari(query):
            return query, [query]

        english = await self._translate_query_for_retrieval(query)
        # Lexical search runs on English only (corpus + tokenizer are English).
        # Semantic search runs on both languages when the translation actually
        # differs from the original; if translation failed (returned the Hindi
        # unchanged) this collapses to a single Hindi embedding — the same input
        # BM25/keyword see, so nothing regresses.
        embed_texts = [english]
        if english.strip() != query.strip():
            embed_texts.append(query)
        return english, embed_texts

    async def _translate_query_for_retrieval(self, message: str) -> str:
        """Translate Hindi/Devanagari questions to English for retrieval.

        Uploaded manuals are indexed in English, while users may ask in Hindi.
        Keep answer generation on the original message, but retrieve with an
        English version so vector and keyword search can match the manual text.
        """
        if not self._contains_devanagari(message):
            return message

        try:
            prompt = f"""Translate this question to English for document retrieval.
Preserve technical terms, part names, abbreviations, model numbers, and numeric values exactly.
Return only one concise English query. Do not answer the question.

Question:
{message}"""
            translated = await self.llm_client.generate(
                prompt,
                system="You translate search queries into English. Return only the translated query.",
                temperature=0.0,
                top_p=0.8,
                num_predict=64,
            )
            translated = re.sub(r"\s+", " ", translated or "").strip().strip('"')
            if not translated:
                return message
            logger.info("Translated retrieval query: %s", translated)
            return translated
        except Exception:
            logger.exception("Failed to translate Hindi retrieval query")
            return message

    async def _generate_conversation_answer(
        self,
        message: str,
        documents: list[dict],
        diagrams: list[dict],
        history: list,
    ) -> dict:
        """Generate an answer with retrieved docs and recent conversation context."""
        terms_source = message

        extraction_documents = documents + [
            {
                **diagram,
                "chunk_text": diagram.get("diagram_description") or diagram.get("chunk_text", ""),
                "content_type": diagram.get("content_type", "diagram"),
            }
            for diagram in diagrams
        ]
        extracted_answer = self.agent._extract_procedure_answer(terms_source, extraction_documents)
        if extracted_answer:
            source_pages = self._source_page_summary(documents)
            diagram_pages = self._diagram_page_summary(diagrams)
            if source_pages:
                extracted_answer = f"{extracted_answer}\n\nSources: {source_pages}."
            if diagram_pages:
                extracted_answer = f"{extracted_answer}\nDiagrams: {diagram_pages}."
            extracted_answer = await self._localize_answer(message, extracted_answer)
            return {
                "answer": extracted_answer,
                "thinking": f"Used {min(len(documents), settings.rag_context_docs)} of {len(documents)} retrieved document chunks and {len(diagrams)} related diagrams.",
            }

        direct_answer = self._extract_direct_answer(terms_source, documents, diagrams)
        if direct_answer:
            direct_answer = await self._localize_answer(message, direct_answer)
            return {
                "answer": direct_answer,
                "thinking": f"Used direct matching text from {len(documents)} retrieved document chunks and {len(diagrams)} related diagrams.",
            }

        if not self._has_sufficient_evidence(terms_source, documents):
            return {
                "answer": self._not_found_answer(message),
                "thinking": f"Found {len(documents)} retrieved chunks, but none provided enough direct evidence.",
            }

        is_hindi = self._contains_devanagari(message)
        # For Hindi, use a tighter context window to reduce LLM processing time.
        # Hindi Devanagari uses more tokens per word than English, so allow more output tokens.
        context = self.agent._format_context(
            documents,
            max_chars=3000 if is_hindi else settings.rag_context_max_chars,
        )
        diagram_context = self._format_diagram_context(diagrams)
        history_lines = []
        for turn in history:
            history_lines.append(f"User: {turn.question}")
            history_lines.append(f"Assistant: {turn.answer}")
        history_text = "\n".join(history_lines) if history_lines else "No previous conversation."

        # A factual lookup / count question wants a single value back, not the
        # parts table it was pulled from. Local models (qwen3:8b) tend to mirror
        # whatever they see in context, so give these a short, strict prompt AND a
        # hard output cap — the cap is what actually stops a full-table dump when
        # the model ignores the instruction.
        #
        # The cap is applied ONLY when the evidence points at a single source
        # manual (_confident_single_source). When several manuals could each
        # answer, we fall through to the detailed, uncapped prompt so the answer
        # can give each manual's value with its own reference instead of being
        # truncated after the first one.
        if not self._wants_detailed_answer(message) and self._confident_single_source(documents):
            prompt = f"""Answer the user's latest message using ONLY the retrieved context.
Give ONLY the specific fact asked for — the exact value, number, name, count, or single table cell — using as few words as possible (aim for well under 300 characters). Do not restate the question or add a preamble.
Reproduce every value EXACTLY as written in the context (numbers, part/item/identification numbers, labels, units); never round, convert, drop, or invent a value.
Do NOT reproduce tables. Do NOT list other rows, parts, or components the question did not ask about. Do NOT add background, assembly steps, or a bill of materials.
Cite the page, and the table name or section when the source is labelled with one, e.g. (table "Page 38 table 2", page 38).
If the specific detail is not present in the context, say plainly that the documents do not specify it. Do not use outside knowledge.
Respond in the same language the user used; keep English technical terms, abbreviations, measurements, and numbers exactly as written.
For a follow-up, resolve references like 'it' or 'that part' to the subject established earlier in the conversation.

Recent conversation:
{history_text}

Latest user message:
{message}

Retrieved context:
{context}

Related diagrams:
{diagram_context}"""

            answer = await self.llm_client.generate(
                prompt,
                system="You are a careful RAG assistant. Answer the exact question asked in one short sentence using only the retrieved context. Do not reproduce tables or list unrelated parts. Do not invent facts.",
                temperature=0.2,
                top_p=0.9,
                num_predict=settings.ollama_num_predict_concise,
            )

            if self._contains_devanagari(message) and not self._contains_devanagari(answer):
                answer = await self._localize_answer(message, answer)

            return {
                "answer": answer,
                "thinking": f"Used {min(len(documents), settings.rag_context_docs)} of {len(documents)} retrieved document chunks, {len(diagrams)} related diagrams, and {len(history)} prior turns (concise lookup).",
            }

        prompt = f"""Answer the user's latest message using the retrieved context and recent conversation.
Respond in the same language the user used. If the user wrote in Hindi, answer in Hindi (Devanagari script) and keep any English technical terms, abbreviations, measurements, and numbers from the source exactly as written.
For a follow-up, resolve references like 'it', 'that', or 'the above part' to the specific subject established earlier in the conversation, and answer about THAT subject. Use only the source(s) that describe that subject; if the retrieved sources are about a different device or manual, say the documents do not cover it rather than answering from an unrelated source.
Answer with page numbers for important facts, using short citations like "(page 3)".
When a source is labeled with a section, cite that section with the page, e.g. "(section 3.1 Main Valve, page 6)".
When a fact comes from a table, read the exact cell value and cite the table by its name and page, e.g. "(table \"DV Specifications\", page 4)".
Prefer exact wording from the context for definitions, names, numbers, limits, and procedures.
Reproduce every technical value EXACTLY as written in the context — numbers, measurements, tolerances, units, dimensions, part/item numbers and their labels (e.g. "3.8±0.1 kg/cm²", "585mm", "nut 35", "spindle 23"). Never round, convert, drop, or invent a value; if a value is not in the context, do not state one.
Each source is labelled with the manual (filename) it came from. When the question is about a specific component or piece of equipment, prefer the source whose manual and section most specifically match it (e.g. a section titled for that exact component) over a generic mention in a different manual.
When the answer differs across manuals, OR the question is general (not tied to one manual) and several manuals each cover it, do NOT merge or silently pick one — give each relevant manual's answer separately, labelled with its filename and page, e.g. "C3W2_D_V_manual.pdf (page 12): ...; Faiveley_make_LSD_DCV_PRV_DV_CPB_CR_IC_Manual.pdf (page 127): ...", so the reader can choose which applies.
If related diagrams are available, include a short "Diagrams" line with their page numbers.
If the specific detail the question asks for (a material, grade, chemical composition, value, dimension, etc.) is not stated in the context, say plainly that the documents do not specify it. Do NOT pad the answer by summarizing tables, listing components, or dumping a bill of materials that does not answer the question. You may point to the closest available identifier (e.g. a part or drawing number) and note where the detail would normally be found (an engineering drawing or a referenced specification), as long as those appear in the context.
Do not use outside knowledge.
Keep the answer direct and concise: answer a simple factual question in one or two sentences and stop. Add operating detail (pistons, springs, valve seats, step-by-step behaviour) only when the question explicitly asks how something works or for a procedure.

Recent conversation:
{history_text}

Latest user message:
{message}

Retrieved context:
{context}

Related diagrams:
{diagram_context}"""

        answer = await self.llm_client.generate(
            prompt,
            system="You are a careful RAG assistant. Always respond in the same language as the user's question. Use the retrieved documents as the factual source of truth. Do not invent facts.",
            temperature=0.2,
            top_p=0.9,
        )

        # Grounding guard: if the generated answer's words don't overlap the
        # retrieved context, it's likely a hallucination — return "not found"
        # rather than a confident wrong answer. Logged so a false rejection is
        # diagnosable; toggle via settings.enable_grounding_check.
        if settings.enable_grounding_check and not self._validate_answer_grounding(answer, documents):
            logger.warning(
                "Answer failed grounding check (low overlap with retrieved context); "
                "replacing with not-found. query=%r answer_preview=%r",
                message[:120], answer[:160],
            )
            answer = self._not_found_answer(message)

        if self._contains_devanagari(message) and not self._contains_devanagari(answer):
            # Model ignored the language instruction — translate as fallback
            answer = await self._localize_answer(message, answer)

        return {
            "answer": answer,
            "thinking": f"Used {min(len(documents), settings.rag_context_docs)} of {len(documents)} retrieved document chunks, {len(diagrams)} related diagrams, and {len(history)} prior turns.",
        }

    async def _localize_answer(self, message: str, answer: str) -> str:
        """Return Hindi/Devanagari answers for Hindi/Devanagari questions."""
        if not answer or not self._contains_devanagari(message):
            return answer
        if self._contains_devanagari(answer):
            return answer

        logger.info("Localizing answer to Hindi because the user message is Devanagari")
        try:
            prompt = f"""Translate this answer to Hindi using Devanagari script.

Rules:
- Any word that appears in English in the source must stay in English exactly as written — do NOT transliterate English words into Devanagari.
- This applies to: technical terms, proper nouns, abbreviations, acronyms, measurements, units, numbers, file names, and page references.
- Do not add new facts. Do not remove citations or source lines.
- Return only the translated Hindi answer. The output must contain Devanagari characters.

Answer:
{answer}"""
            translated = await self.llm_client.generate(
                prompt,
                system="You translate technical answers into Hindi Devanagari. English technical terms, abbreviations, and proper nouns must stay in English exactly as written — never transliterate them into Devanagari script.",
                temperature=0.0,
                top_p=0.8,
                num_predict=1024,
            )
            translated = (translated or "").strip()
            if translated and self._contains_devanagari(translated):
                return translated

            logger.warning("Hindi localization returned non-Devanagari output; retrying")
            retry_prompt = f"""हिंदी में देवनागरी लिपि का उपयोग करके नीचे दिए गए उत्तर का अनुवाद करें।
महत्वपूर्ण: जो शब्द source में English में हैं, उन्हें English में ही रखें — उन्हें देवनागरी में मत लिखें।
इसमें शामिल हैं: technical terms, abbreviations, acronyms, measurements, units, numbers, file names, page references।
केवल हिंदी अनुवाद लौटाएं।

Answer:
{answer}"""
            translated = await self.llm_client.generate(
                retry_prompt,
                system="Translate the answer into Hindi Devanagari. English technical terms and abbreviations must remain in English exactly. Do not transliterate any English word.",
                temperature=0.0,
                top_p=0.8,
                num_predict=1024,
            )
            translated = (translated or "").strip()
            if translated and self._contains_devanagari(translated):
                return translated
        except Exception:
            logger.exception("Failed to translate answer to Hindi")

        logger.warning("Returning original answer because Hindi localization failed")
        return answer

    @staticmethod
    def _not_found_answer(message: str) -> str:
        """Return a sensible not-found message in the user's language."""
        if ChatService._contains_devanagari(message):
            return "माफ़ करें, अपलोड किए गए दस्तावेज़ों में आपके इस सवाल का जवाब नहीं मिला। कृपया संबंधित दस्तावेज़ अपलोड करें या अलग शब्दों में पूछें।"
        return "The uploaded documents do not provide enough information to answer this question."

    @staticmethod
    def _excerpt(text: str, max_chars: int = 240) -> str:
        text = re.sub(r"\s+", " ", (text or "")).strip()
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 3].rstrip() + "..."

    def _format_diagram_context(self, diagrams: list[dict]) -> str:
        if not diagrams:
            return "No related diagrams found."

        lines = []
        for index, diagram in enumerate(diagrams, start=1):
            filename = diagram.get("filename", "Unknown")
            page_number = diagram.get("page_number", 0)
            image_index = diagram.get("image_index")
            description = self._excerpt(
                diagram.get("diagram_description") or diagram.get("chunk_text", ""),
                max_chars=400,
            )
            image_label = f", image {image_index}" if image_index is not None else ""
            page_label = self._format_page_reference(page_number, diagram.get("document_page_number"))
            lines.append(f"Diagram {index} ({filename}, {page_label}{image_label}): {description}")

        return "\n".join(lines)

    # Questions that want an explanation, not a single extracted value/sentence.
    _DESCRIPTIVE_RE = re.compile(
        r"\b(principle|explain|describe|overview|mechanism|purpose\s+of|function\s+of)\b"
        r"|how\s+(?:does|do|is|are|it)\b",
        re.IGNORECASE,
    )

    @classmethod
    def _is_descriptive_question(cls, message: str) -> bool:
        """True for explanatory questions ('principle of operation', 'how does X work')."""
        return bool(cls._DESCRIPTIVE_RE.search(message or ""))

    # A question naming a measurable quantity wants a single value; one that only
    # names a component ("Manually Operated Quick Release Portion") wants its whole
    # section, which the LLM must assemble.
    _VALUE_NOUN_RE = re.compile(
        r"\b(speed|pressure|rate|limit|value|distance|dimension|temperature|torque|"
        r"interval|time|weight|size|capacity|tolerance|clearance|gap|diameter|length|"
        r"width|height|volume|force|load|quantity|number|range|frequency|voltage|"
        r"current|power|angle|thickness|depth|stroke|setting|reading|level|ratio|"
        r"percentage)\b",
        re.IGNORECASE,
    )

    @classmethod
    def _has_value_noun(cls, message: str) -> bool:
        """True when the question asks for a measurable value rather than a description."""
        return bool(cls._VALUE_NOUN_RE.search(message or ""))

    # Single-answer lookups: a count ("how many bolts"), a quantity ("how much
    # clearance"), or one field pulled from a parts table ("what is the
    # Identification No. for X", "the description for Greysham Item No Y"). These
    # want ONE value back, not the table they came from — even when a procedural
    # stem such as "assembly" happens to appear in the sentence ("...required for
    # the assembly?"), which would otherwise be misread as a procedure request.
    _LOOKUP_RE = re.compile(
        r"\bhow\s+many\b|\bhow\s+much\b"
        r"|\b(identification|item|part|greysham|drawing|reference|serial|catalogue|catalog)\s*"
        r"(no\.?|number|code)\b"
        r"|\bpart\s*(no\.?|number)\b"
        r"|\bdescription\s+(for|of)\b",
        re.IGNORECASE,
    )

    @classmethod
    def _is_lookup_question(cls, message: str) -> bool:
        """True for a single-value count/quantity or table-field lookup question."""
        return bool(cls._LOOKUP_RE.search(message or ""))

    # An explicit request to SEE a table or a full listing ("list all the parts",
    # "show me the full table", "give the complete spare parts list", "bill of
    # materials"). This is a deliberate ask for the whole table, so it must never
    # be truncated to a one-line answer — it outranks the lookup/concise rule
    # below. A bare mention of "table" is intentionally NOT enough (e.g. "in the
    # table ... what is the Identification No. for X" is still a one-cell lookup).
    _TABLE_LIST_RE = re.compile(
        # Imperative verb "list ..." only (NOT the noun in "spare parts list"),
        # recognised by the determiner/quantifier that follows it.
        r"\blist\s+(?:all|the|out|down|every|each|of|me)\b"
        r"|\ball\s+(?:the\s+|of\s+the\s+)?(?:parts|items|components|rows|entries|spare\s+parts)\b"
        r"|\bevery\s+(?:part|item|component|row)\b"
        r"|\b(?:full|complete|entire|whole)\s+(?:spare\s+)?(?:parts?\s+)?(?:table|list|bill)\b"
        r"|\bbill\s+of\s+materials\b"
        r"|\b(?:show|give|display|provide)\s+(?:me\s+|us\s+)?(?:a\s+|the\s+)?"
        r"(?:full\s+|complete\s+|entire\s+|whole\s+)?(?:spare\s+)?(?:parts?\s+)?(?:table|list)\b"
        r"|\bwhat\s+are\s+all\b",
        re.IGNORECASE,
    )

    @classmethod
    def _wants_table_or_list(cls, message: str) -> bool:
        """True when the user explicitly asks to see a table or full listing."""
        return bool(cls._TABLE_LIST_RE.search(message or ""))

    @classmethod
    def _wants_detailed_answer(cls, message: str) -> bool:
        """True when the question warrants a full multi-line / table / step answer.

        Priority order:
        1. An explicit "show me the table / list everything" request always wins —
           the user asked to see the whole table, so never truncate it.
        2. A single-value count or field lookup is concise, even if it shares a
           keyword with the descriptive/procedural classes (e.g. "assembly").
        3. Otherwise, only genuinely descriptive ("how does X work", "principle of
           operation") or procedural ("how to dismantle") questions want detail.
        """
        if cls._wants_table_or_list(message):
            return True
        if cls._is_lookup_question(message):
            return False
        return cls._is_descriptive_question(message) or ReActAgent._is_procedural_question(message)

    def _extract_direct_answer(self, message: str, documents: list[dict], diagrams: list[dict]) -> str:
        query_terms = self._important_terms(message)
        if not query_terms:
            return ""

        # A descriptive OR procedural question wants a synthesized, complete answer;
        # a single extracted sentence would start mid-context and drop steps, so
        # defer both to the LLM.
        if self._is_descriptive_question(message) or ReActAgent._is_procedural_question(message):
            return ""
        # This path answers single-value factual questions ("what is the SPEED of
        # X"). A definitional question that only names a component ("what is the
        # Manually Operated Quick Release Portion?") has no value noun and its answer
        # is a whole section — defer it to the LLM instead of returning a lone
        # heading sentence from whichever manual happened to rank first.
        if not self._has_value_noun(message):
            return ""

        best_document = None
        best_score = 0.0
        for document in documents:
            chunk_text = document.get("chunk_text", "")
            normalized = self._normalize_text(chunk_text)
            matched_terms = sum(1 for term in query_terms if term in normalized)
            score = matched_terms + float(document.get("lexical_score") or 0.0)
            if score > best_score:
                best_score = score
                best_document = document

        if not best_document or best_score < max(1, len(query_terms) - 1):
            return ""

        chunk_text = re.sub(r"\s+", " ", best_document.get("chunk_text", "")).strip()
        if not chunk_text:
            return ""

        sentences = re.split(r"(?<=[.!?])\s+", chunk_text)
        normalized_query = " ".join(query_terms)
        selected = []

        for sentence in sentences:
            normalized_sentence = self._normalize_text(sentence)
            if normalized_query in normalized_sentence and re.search(r"\b(is|are|used|set|designed|connected)\b", sentence, re.IGNORECASE):
                selected.append(sentence.strip())
                break

        if not selected:
            for sentence in sentences:
                normalized_sentence = self._normalize_text(sentence)
                matched_terms = sum(1 for term in query_terms if term in normalized_sentence)
                if matched_terms >= max(1, len(query_terms) - 1) and len(sentence.split()) > 8:
                    selected.append(sentence.strip())
                    break

        if not selected:
            return ""

        page_number = best_document.get("page_number", 0) or 0
        document_page_number = best_document.get("document_page_number")
        answer_text = self._strip_leading_heading(selected[0], query_terms)
        page_reference = self._format_page_reference(page_number, document_page_number)
        answer = f"{answer_text} ({page_reference})"

        diagram_pages = self._diagram_page_summary(diagrams)
        if page_number:
            answer = f"{answer}\n\nSources: {page_reference}."
        if diagram_pages:
            answer = f"{answer}\nDiagrams: {diagram_pages}."
        return answer

    @staticmethod
    def _contains_devanagari(text: str) -> bool:
        return bool(re.search(r"[\u0900-\u097F]", text or ""))

    @staticmethod
    def _strip_leading_heading(text: str, query_terms: list[str]) -> str:
        query_phrase = r"\s+".join(re.escape(term) for term in query_terms)
        stripped = re.sub(
            rf"^\s*\d+(\.\d+)*\s+{query_phrase}\s+",
            "",
            text,
            flags=re.IGNORECASE,
        ).strip()
        return stripped or text

    @staticmethod
    def _format_page_reference(page_number: int, document_page_number: Optional[int] = None) -> str:
        if document_page_number and document_page_number != page_number:
            return f"page {page_number} / document page {document_page_number}"
        return f"page {page_number}"

    @staticmethod
    def _important_terms(text: str) -> list[str]:
        stop_words = {
            "a", "an", "and", "are", "for", "how", "in", "is", "of", "on",
            "or", "the", "to", "what", "when", "where", "which", "with",
            "tell", "about", "explain", "describe", "give",
            # Demonstratives are never content terms; keeping them inflates the
            # term count and skews the direct-answer / evidence thresholds.
            "this", "that", "these", "those", "its",
        }
        return [
            term
            for term in re.findall(r"[a-z0-9]+", text.lower())
            if len(term) > 2 and term not in stop_words
        ]

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join(re.findall(r"[a-z0-9]+", text.lower()))

    @staticmethod
    def _confident_single_source(documents: list[dict], score_ratio: float = 0.75) -> bool:
        """True when the retrieved evidence clearly points at ONE source manual.

        Gates the concise answer cap. If two or more manuals score comparably —
        each could hold the answer — this returns False so the answer takes the
        uncapped detailed path and can present each manual's value with its own
        reference, rather than being truncated after the first source.

        "Comparable" means within ``score_ratio`` of the top relevance score, so a
        weak incidental match in another manual does not force the detailed path.
        """
        scored = [
            (str(doc.get("filename") or ""), float(doc.get("relevance_score") or 0.0))
            for doc in documents
            if doc.get("filename")
        ]
        if not scored:
            return False
        top = max(score for _, score in scored)
        if top <= 0:
            # No usable relevance scores — treat an ambiguous multi-manual
            # retrieval as multi-source (no cap) by falling back to file count.
            return len({filename for filename, _ in scored}) <= 1
        cutoff = top * score_ratio
        strong_sources = {filename for filename, score in scored if score >= cutoff}
        return len(strong_sources) <= 1

    @staticmethod
    def _source_page_summary(documents: list[dict]) -> str:
        pages = sorted({
            int(doc.get("page_number"))
            for doc in documents
            if doc.get("page_number")
        })
        return ", ".join(f"page {page}" for page in pages)

    @staticmethod
    def _diagram_page_summary(diagrams: list[dict]) -> str:
        pages = sorted({
            int(diagram.get("page_number"))
            for diagram in diagrams
            if diagram.get("page_number")
        })
        return ", ".join(f"page {page}" for page in pages)

    async def get_chat_history(
        self,
        user_id: uuid.UUID,
        skip: int = 0,
        limit: int = 50,
    ) -> list:
        """Get chat history for user."""
        histories = await self.chat_repo.get_by_user(user_id, skip, limit)
        result = []
        for h in histories:
            try:
                sources = json.loads(h.sources) if h.sources else []
            except json.JSONDecodeError:
                sources = []
            try:
                diagrams = json.loads(h.diagrams) if h.diagrams else []
            except json.JSONDecodeError:
                diagrams = []

            result.append({
                "id": h.id,
                "conversation_id": h.conversation_id,
                "question": h.question,
                "answer": h.answer,
                "sources": [SourceReference(**source) for source in sources],
                "diagrams": [DiagramReference(**diagram) for diagram in diagrams],
                "created_at": h.created_at,
            })
        return result

    async def get_conversation_history(
        self,
        user_id: uuid.UUID,
        conversation_id: uuid.UUID,
        limit: int = 100,
    ) -> list[ConversationTurnResponse]:
        """Get ordered turns for a conversation."""
        histories = await self.chat_repo.get_by_conversation(user_id, conversation_id, limit)
        result = []
        for history in histories:
            try:
                sources = json.loads(history.sources) if history.sources else []
            except json.JSONDecodeError:
                sources = []
            try:
                diagrams = json.loads(history.diagrams) if history.diagrams else []
            except json.JSONDecodeError:
                diagrams = []

            result.append(
                ConversationTurnResponse(
                    id=history.id,
                    conversation_id=history.conversation_id,
                    question=history.question,
                    answer=history.answer,
                    sources=[SourceReference(**source) for source in sources],
                    diagrams=[DiagramReference(**diagram) for diagram in diagrams],
                    model=history.model,
                    created_at=history.created_at,
                )
            )
        return result

    async def list_conversations(
        self,
        user_id: uuid.UUID,
        limit: int = 50,
    ) -> list[ConversationSummaryResponse]:
        """List conversation summaries for a user."""
        histories = await self.chat_repo.list_conversations(user_id, limit)
        return [
            ConversationSummaryResponse(
                conversation_id=history.conversation_id,
                last_question=history.question,
                last_answer=history.answer,
                model=history.model,
                created_at=history.created_at,
            )
            for history in histories
        ]

    def _validate_answer_grounding(self, answer: str, documents: list[dict]) -> bool:
        """Validate that the answer is grounded in the retrieved documents."""
        if not documents:
            return False

        answer_lower = answer.lower()
        combined_text = " ".join([doc.get("chunk_text", "") for doc in documents]).lower()
        if not combined_text.strip():
            return False

        if "do not provide enough information" in answer_lower:
            return True

        # Extract English words from the answer for overlap check.
        # For Hindi answers the model may output very few English words —
        # if none are found, fall back to checking the documents have any
        # content at all (non-empty retrieval is a weak but safe signal).
        key_phrases = [
            word
            for word in re.findall(r"[a-z0-9]+", answer_lower)
            if len(word) > 4
        ]
        if not key_phrases:
            return bool(combined_text.strip())

        matches = sum(1 for phrase in key_phrases if phrase in combined_text)
        return matches / len(key_phrases) >= 0.2

    def _has_sufficient_evidence(self, message: str, documents: list[dict]) -> bool:
        if not documents:
            return False

        query_terms = self._important_terms(message)
        if not query_terms:
            return True

        for document in documents:
            chunk_text = document.get("chunk_text", "")
            lexical_score = float(document.get("lexical_score") or 0.0)
            relevance_score = float(document.get("relevance_score") or 0.0)
            normalized = self._normalize_text(chunk_text)
            matched_terms = sum(1 for term in set(query_terms) if term in normalized)
            if matched_terms >= max(1, min(3, len(set(query_terms)))):
                return True
            if lexical_score >= 0.45 or relevance_score >= settings.similarity_threshold:
                return True

        return False
    
    async def list_all_conversations(
        self,
        limit: int = 100,
        user_name: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> list[dict]:
        """List all conversation summaries across the system (all users)."""
        rows = await self.chat_repo.list_all_conversations(limit, user_name=user_name, start_date=start_date, end_date=end_date)
        return [
            {
                "user": {
                    "id": row.ChatHistory.user_id,
                    "name": row.name,
                },
                "conversation_id": row.ChatHistory.conversation_id,
                "conversation_title": row.first_question[:50] + "..." if row.first_question and len(row.first_question) > 50 else row.first_question,
                "first_question": row.first_question,
                "first_answer": row.first_answer,
                "last_question": row.ChatHistory.question,
                "last_answer": row.ChatHistory.answer,
                "model": row.ChatHistory.model,
                "startdate": row.start_date,
                "last_activity": row.ChatHistory.created_at,
            }
            for row in rows
        ]
 
    async def get_conversation_history_by_id(
        self,
        conversation_id: uuid.UUID,
        limit: int = 100,
    ) -> list[ConversationTurnResponse]:
        """Get ordered turns for a conversation."""
        histories = await self.chat_repo.get_by_conversation_history(conversation_id, limit)
        result = []
        for history in histories:
            try:
                sources = json.loads(history.sources) if history.sources else []
            except json.JSONDecodeError:
                sources = []
            try:
                diagrams = json.loads(history.diagrams) if history.diagrams else []
            except json.JSONDecodeError:
                diagrams = []
 
            result.append(
                ConversationTurnResponse(
                    id=history.id,
                    conversation_id=history.conversation_id,
                    question=history.question,
                    answer=history.answer,
                    sources=[SourceReference(**source) for source in sources],
                    diagrams=[DiagramReference(**diagram) for diagram in diagrams],
                    model=history.model,
                    created_at=history.created_at,
                )
            )
        return result
