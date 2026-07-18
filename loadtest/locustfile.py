"""
Optional Locust load test for the Chat RAG AI Agent chat API.

Locust gives you a live web dashboard (charts, RPS, percentiles, failures) on top
of the same endpoints the standalone load_test.py drives. Use this when you want
interactive control of the user count during a run.

Install + run:
    pip install locust
    locust -f loadtest/locustfile.py --host http://localhost:8000

Then open http://localhost:8089 and set the number of users / spawn rate.

Auth: set credentials via environment variables before launching, e.g.
    LOADTEST_EMAIL=you@example.com LOADTEST_PASSWORD=secret \
        locust -f loadtest/locustfile.py --host http://localhost:8000
or provide a pre-issued token with LOADTEST_TOKEN to skip login.

Endpoint: defaults to /chat/message; set LOADTEST_ENDPOINT=ask for /chat/ask.

Note: RAG answers on a CPU Ollama backend are slow (tens of seconds each), so keep
the user count modest and expect low RPS — you are measuring latency under load,
not throughput.
"""

import os
import random

from locust import HttpUser, between, task

QUESTIONS = [
    "What is the principle of operation described in the manual?",
    "List the maintenance steps for the brake system.",
    "What are the dimensions and weight of the wagon body?",
    "Explain the air brake distributor valve function.",
    "What torque values are specified for the bogie bolts?",
    "Describe the suspension arrangement of the bogie.",
    "बोगी के रखरखाव की प्रक्रिया क्या है?",
]

ENDPOINT = os.getenv("LOADTEST_ENDPOINT", "message")  # "message" | "ask"


class ChatUser(HttpUser):
    # RAG answers are slow; a real user would pause between questions.
    wait_time = between(1, 5)

    def on_start(self) -> None:
        """Authenticate once per simulated user."""
        token = os.getenv("LOADTEST_TOKEN")
        if not token:
            email = os.getenv("LOADTEST_EMAIL")
            password = os.getenv("LOADTEST_PASSWORD")
            if not (email and password):
                raise RuntimeError(
                    "Set LOADTEST_TOKEN, or LOADTEST_EMAIL and LOADTEST_PASSWORD."
                )
            with self.client.post(
                "/auth/login",
                json={"email": email, "password": password},
                name="/auth/login",
                catch_response=True,
            ) as resp:
                if resp.status_code != 200:
                    resp.failure(f"login failed: {resp.status_code} {resp.text[:120]}")
                    raise RuntimeError("login failed")
                token = resp.json().get("access_token")
        self.client.headers.update({"Authorization": f"Bearer {token}"})
        self.conversation_id = None

    @task
    def chat(self) -> None:
        question = random.choice(QUESTIONS)
        if ENDPOINT == "ask":
            path, payload, name = "/chat/ask", {"question": question}, "/chat/ask"
        else:
            payload = {"message": question}
            if self.conversation_id:
                payload["conversation_id"] = self.conversation_id
            path, name = "/chat/message", "/chat/message"

        # High timeout: a single CPU LLM answer can take tens of seconds.
        with self.client.post(
            path, json=payload, name=name, timeout=300, catch_response=True
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"HTTP {resp.status_code}: {resp.text[:120]}")
                return
            try:
                body = resp.json()
            except ValueError:
                resp.failure("non-JSON response")
                return
            if ENDPOINT == "message" and body.get("conversation_id"):
                self.conversation_id = body["conversation_id"]
            if not body.get("answer"):
                resp.failure("empty answer")
