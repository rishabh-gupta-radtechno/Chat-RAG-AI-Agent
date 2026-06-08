"""
Ollama LLM client for local model inference.
"""

import re
import httpx
import time
from typing import Optional

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


class OllamaClient:
    """Ollama LLM client."""

    def __init__(self):
        self.base_url = settings.ollama_base_url
        self.model = settings.ollama_chat_model
        self.embedding_model = settings.ollama_embedding_model
        self.embeddings_path = settings.ollama_embeddings_path
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                settings.ollama_timeout_seconds,
                connect=10.0,
                read=settings.ollama_timeout_seconds,
                write=30.0,
                pool=10.0,
            )
        )

    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        num_predict: Optional[int] = None,
    ) -> str:
        """Generate text using Ollama."""
        try:
            started_at = time.perf_counter()
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

            logger.info(
                "Ollama chat request model=%s prompt_chars=%s",
                self.model,
                len(prompt),
            )
            response = await self.client.post(
                f"{self.base_url.rstrip('/')}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "options": {
                        "temperature": temperature,
                        "top_p": top_p,
                        "num_ctx": settings.ollama_num_ctx,
                        "num_predict": num_predict or settings.ollama_num_predict,
                    },
                    "keep_alive": "10m",
                    "stream": False,
                },
            )

            if response.status_code != 200:
                raise RuntimeError(
                    f"Ollama API error: {response.status_code} {response.text}"
                )

            result = response.json()
            logger.info(
                "Ollama chat completed model=%s duration_seconds=%.2f",
                self.model,
                time.perf_counter() - started_at,
            )
            content = result.get("message", {}).get("content", "")
            # Strip thinking blocks emitted by reasoning models (e.g. qwen3)
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            return content

        except httpx.ReadTimeout as e:
            raise TimeoutError(
                f"Ollama model {self.model} did not respond within "
                f"{settings.ollama_timeout_seconds} seconds"
            ) from e
        except Exception:
            logger.exception("Error generating text with Ollama model %s", self.model)
            raise

    async def embed(self, text: str) -> list[float]:
        """Generate embeddings using Ollama."""
        path = self.embeddings_path or "/api/embeddings"
        if not path.startswith("/"):
            path = f"/{path}"

        url = f"{self.base_url.rstrip('/')}{path}"
        payload = {"model": self.embedding_model}

        if path == "/api/embed":
            payload["input"] = text
        else:
            payload["prompt"] = text

        response = await self.client.post(url, json=payload)

        logger.info("Ollama embed request %s %s", response.request.method, response.url)
        logger.info("Ollama embed status %s", response.status_code)
        logger.debug("Ollama embed response: %s", response.text)

        if response.status_code != 200:
            raise RuntimeError(f"Ollama API error: {response.status_code} {response.text}")

        data = response.json()
        if isinstance(data.get("embedding"), list):
            return data["embedding"]
        if data.get("embeddings"):
            return data["embeddings"][0]

        raise RuntimeError(f"Unexpected Ollama response format: {data}")

    async def health_check(self) -> bool:
        """Check if Ollama is healthy."""
        try:
            response = await self.client.get(f"{self.base_url}/api/tags")
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            return False

    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()
