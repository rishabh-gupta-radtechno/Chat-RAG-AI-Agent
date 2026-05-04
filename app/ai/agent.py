"""
ReAct Agent using LangGraph for orchestration.
"""

from typing import Optional, Any
from enum import Enum

from app.ai.llm import OllamaClient
from app.core.logging import get_logger

logger = get_logger(__name__)


class AgentAction(Enum):
    """Agent actions."""

    RETRIEVE = "retrieve"
    REASON = "reason"
    RESPOND = "respond"


class ReActAgent:
    """ReAct (Reason + Act) Agent for RAG-based question answering."""

    def __init__(self, rag_pipeline):
        self.rag_pipeline = rag_pipeline
        self.llm_client = OllamaClient()
        self.max_iterations = 5

    async def process(
        self,
        question: str,
        documents: list[dict],
    ) -> dict:
        """Process question using ReAct pattern."""
        try:
            logger.info(f"Starting ReAct agent for question: {question}")

            # Initialize state
            state = {
                "question": question,
                "documents": documents,
                "thought": "",
                "action": "",
                "observation": "",
                "answer": "",
                "iteration": 0,
            }

            # Iterate through ReAct loop
            while state["iteration"] < self.max_iterations:
                state["iteration"] += 1

                # Step 1: Thought
                state = await self._thought(state)

                # Step 2: Action
                state = await self._action(state)

                # Step 3: Observation
                state = await self._observation(state)

                # Check if we should stop
                if state.get("is_final"):
                    break

            # Generate final response
            state = await self._respond(state)

            return {
                "answer": state.get("answer", ""),
                "thinking": f"Iterations: {state['iteration']}\nThought: {state.get('thought', '')}",
            }

        except Exception as e:
            logger.error(f"Error in ReAct agent: {e}")
            raise

    async def _thought(self, state: dict) -> dict:
        """Thought step: Reason about the question."""
        try:
            context = "\n".join([f"- {doc.get('chunk_text', '')}" for doc in state["documents"][:3]])

            prompt = f"""Given the question and relevant documents, think through how to answer it.

Question: {state['question']}

Relevant Documents:
{context}

Provide a concise thought on how to answer this question."""

            thought = await self.llm_client.generate(prompt)
            state["thought"] = thought
            logger.info(f"Thought: {thought}")

            return state

        except Exception as e:
            logger.error(f"Error in thought step: {e}")
            state["thought"] = "Error generating thought"
            return state

    async def _action(self, state: dict) -> dict:
        """Action step: Decide what action to take."""
        try:
            # For now, we're using the retrieved documents
            state["action"] = "use_retrieved_documents"
            logger.info(f"Action: {state['action']}")
            return state

        except Exception as e:
            logger.error(f"Error in action step: {e}")
            return state

    async def _observation(self, state: dict) -> dict:
        """Observation step: Process the action results."""
        try:
            # Combine documents into context
            context = "\n".join([
                f"Document {i+1} ({doc.get('filename', 'Unknown')}): {doc.get('chunk_text', '')}"
                for i, doc in enumerate(state["documents"][:5])
            ])

            state["observation"] = context
            state["is_final"] = True  # For now, we consider it final after one iteration

            logger.info(f"Observation generated from {len(state['documents'])} documents")
            return state

        except Exception as e:
            logger.error(f"Error in observation step: {e}")
            state["is_final"] = True
            return state

    async def _respond(self, state: dict) -> dict:
        """Respond step: Generate final answer."""
        try:
            prompt = f"""Based on the following question and context, provide a comprehensive answer.

Question: {state['question']}

Context:
{state.get('observation', '')}

Thought Process:
{state.get('thought', '')}

Please provide a clear, concise answer that directly addresses the question."""

            answer = await self.llm_client.generate(prompt)
            state["answer"] = answer
            logger.info(f"Generated answer: {answer[:100]}...")

            return state

        except Exception as e:
            logger.error(f"Error in respond step: {e}")
            state["answer"] = "Unable to generate answer. Please try again."
            return state
