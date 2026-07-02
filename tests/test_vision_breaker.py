import asyncio
import os

os.environ["DEBUG"] = "false"

from app.ai.llm import OllamaClient
from app.core.config import get_settings


class _Resp:
    def __init__(self, status):
        self.status_code = status
        self.text = "boom"

    def json(self):
        return {"response": "ok"}


def test_vision_circuit_breaker_disables_after_failures(monkeypatch):
    threshold = get_settings().vision_max_consecutive_failures
    client = OllamaClient()
    calls = {"n": 0}

    async def fake_post(*args, **kwargs):
        calls["n"] += 1
        return _Resp(500)  # always fails ("unexpected EOF" class of error)

    monkeypatch.setattr(client.client, "post", fake_post)

    async def run():
        for _ in range(threshold):
            try:
                await client.vision("p", b"img")
            except Exception:
                pass
        assert calls["n"] == threshold
        assert client._vision_disabled is True

        # Further calls short-circuit: no new HTTP request is made.
        try:
            await client.vision("p", b"img")
        except Exception:
            pass
        assert calls["n"] == threshold  # unchanged — breaker skipped the call

        # Reset re-enables it (as done per document).
        client.reset_vision_breaker()
        try:
            await client.vision("p", b"img")
        except Exception:
            pass
        assert calls["n"] == threshold + 1

    asyncio.run(run())


def test_vision_success_resets_failure_count(monkeypatch):
    client = OllamaClient()
    sequence = [500, 200]  # fail, then succeed

    async def fake_post(*args, **kwargs):
        return _Resp(sequence.pop(0))

    monkeypatch.setattr(client.client, "post", fake_post)

    async def run():
        try:
            await client.vision("p", b"img")
        except Exception:
            pass
        assert client._vision_failures == 1
        out = await client.vision("p", b"img")
        assert out == "ok"
        assert client._vision_failures == 0 and client._vision_disabled is False

    asyncio.run(run())
