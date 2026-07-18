#!/usr/bin/env python3
"""
Async load tester for the Chat RAG AI Agent chat API.

Targets the FastAPI server (default http://localhost:8000) and drives one of the
chat endpoints under configurable concurrency:

    POST /chat/message   {"message": ..., "conversation_id": optional}
    POST /chat/ask        {"question": ...}

Because the backend answers with a local Ollama LLM (qwen3:8b) on CPU with
OLLAMA_NUM_PARALLEL=1, answer generation is effectively serialized on the server.
So this tool is built to reveal *latency under concurrency* and queue behavior
(how p95/p99 degrade as you add virtual users), not raw requests-per-second. Keep
--timeout generous; a single CPU answer can take tens of seconds to minutes.

Only dependency is httpx, which the project already uses.

Examples
--------
# 20 requests, 4 concurrent users, logging in with a real account:
python loadtest/load_test.py --email you@example.com --password secret \
    --concurrency 4 --requests 20

# Run for 2 minutes at concurrency 8 against the single-shot /chat/ask endpoint:
python loadtest/load_test.py --token "$TOKEN" --endpoint ask \
    --concurrency 8 --duration 120

# Ramp 10 users up over 30s, multi-turn conversations, save raw results:
python loadtest/load_test.py --email you@example.com --password secret \
    --concurrency 10 --ramp 30 --duration 300 --conversation \
    --out results.csv --summary-out summary.json
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import random
import statistics
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

try:
    import httpx
except ImportError:  # pragma: no cover
    sys.exit(
        "httpx is required. Install it with:  pip install httpx\n"
        "(it is already listed in the project's requirements.txt)"
    )


# --------------------------------------------------------------------------- #
# Built-in question pool (used when --questions file is not provided).
# Mixes generic + railway-manual (RDSO/Escorts wagon) style questions.
# --------------------------------------------------------------------------- #
DEFAULT_QUESTIONS = [
    "What is the principle of operation described in the manual?",
    "List the maintenance steps for the brake system.",
    "What are the dimensions and weight of the wagon body?",
    "Explain the air brake distributor valve function.",
    "What is the maximum axle load permitted?",
    "Describe the coupler assembly and its components.",
    "What torque values are specified for the bogie bolts?",
    "Summarize the safety precautions for maintenance.",
    "What is the drawing number for the side frame?",
    "How is the buffer height measured and adjusted?",
    "What lubricants are recommended for the axle boxes?",
    "Explain the loading and unloading procedure.",
    "What are the inspection intervals for wheel sets?",
    "Describe the suspension arrangement of the bogie.",
    "What is the tare weight and payload capacity?",
    "बोगी के रखरखाव की प्रक्रिया क्या है?",
    "ब्रेक सिस्टम के मुख्य घटक कौन से हैं?",
]


@dataclass
class RequestResult:
    """One completed (or failed) request."""

    worker: int
    seq: int
    endpoint: str
    started: float          # epoch seconds
    latency: float          # seconds
    status: int             # HTTP status, 0 if no response (timeout/conn error)
    ok: bool
    error: str = ""
    resp_bytes: int = 0
    answer_chars: int = 0
    question: str = ""


@dataclass
class Config:
    base_url: str
    endpoint: str           # "message" | "ask"
    concurrency: int
    requests: Optional[int]
    duration: Optional[float]
    ramp: float
    timeout: float
    think_min: float
    think_max: float
    conversation: bool
    questions: list[str]
    verbose: bool


# --------------------------------------------------------------------------- #
# Auth
# --------------------------------------------------------------------------- #
async def login(base_url: str, email: str, password: str, timeout: float) -> str:
    """Log in and return the access token."""
    url = f"{base_url.rstrip('/')}/auth/login"
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(url, json={"email": email, "password": password})
    if resp.status_code != 200:
        raise SystemExit(
            f"Login failed ({resp.status_code}) at {url}: {resp.text[:300]}"
        )
    data = resp.json()
    token = data.get("access_token")
    if not token:
        raise SystemExit(f"Login response missing access_token: {data}")
    return token


# --------------------------------------------------------------------------- #
# Shared run state
# --------------------------------------------------------------------------- #
class RunState:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.results: list[RequestResult] = []
        self.dispatched = 0          # reserved-but-maybe-in-flight count
        self.completed = 0
        self.in_flight = 0
        self.stop = asyncio.Event()
        self.start_time = 0.0

    def reserve(self) -> Optional[int]:
        """Atomically reserve the next request index, or None if we should stop.

        Safe without a lock: there is no ``await`` between the checks and the
        increment, so the single-threaded event loop can't interleave here.
        """
        if self.stop.is_set():
            return None
        if self.cfg.requests is not None and self.dispatched >= self.cfg.requests:
            return None
        if self.cfg.duration is not None and (time.monotonic() - self.start_time) >= self.cfg.duration:
            self.stop.set()
            return None
        seq = self.dispatched
        self.dispatched += 1
        return seq


# --------------------------------------------------------------------------- #
# Worker
# --------------------------------------------------------------------------- #
async def worker(wid: int, client: httpx.AsyncClient, state: RunState) -> None:
    cfg = state.cfg
    path = "/chat/message" if cfg.endpoint == "message" else "/chat/ask"
    url = f"{cfg.base_url.rstrip('/')}{path}"
    conversation_id: Optional[str] = None

    # Ramp: stagger worker start across the ramp window.
    if cfg.ramp > 0 and cfg.concurrency > 1:
        await asyncio.sleep(cfg.ramp * (wid / cfg.concurrency))

    while True:
        seq = state.reserve()
        if seq is None:
            return

        question = random.choice(cfg.questions)
        if cfg.endpoint == "message":
            payload: dict = {"message": question}
            if cfg.conversation and conversation_id:
                payload["conversation_id"] = conversation_id
        else:
            payload = {"question": question}

        state.in_flight += 1
        started = time.time()
        t0 = time.monotonic()
        status = 0
        ok = False
        error = ""
        resp_bytes = 0
        answer_chars = 0
        try:
            resp = await client.post(url, json=payload)
            status = resp.status_code
            resp_bytes = len(resp.content)
            ok = 200 <= status < 300
            if ok:
                try:
                    body = resp.json()
                    answer_chars = len(body.get("answer", "") or "")
                    if cfg.endpoint == "message" and cfg.conversation:
                        conversation_id = body.get("conversation_id") or conversation_id
                except Exception:
                    pass  # non-JSON 2xx is still counted as ok
            else:
                error = f"HTTP {status}: {resp.text[:120]}"
        except httpx.TimeoutException:
            error = f"timeout>{cfg.timeout}s"
        except httpx.HTTPError as e:
            error = f"{type(e).__name__}: {e}"
        except Exception as e:  # noqa: BLE001
            error = f"{type(e).__name__}: {e}"
        finally:
            latency = time.monotonic() - t0
            state.in_flight -= 1
            state.completed += 1
            state.results.append(
                RequestResult(
                    worker=wid, seq=seq, endpoint=cfg.endpoint, started=started,
                    latency=latency, status=status, ok=ok, error=error,
                    resp_bytes=resp_bytes, answer_chars=answer_chars, question=question,
                )
            )
            if cfg.verbose:
                tag = "ok " if ok else "ERR"
                print(
                    f"  [{tag}] w{wid:02d} #{seq} {latency:6.2f}s "
                    f"status={status} ans={answer_chars}c "
                    f"{('- ' + error) if error else ''}",
                    flush=True,
                )

        if ok and (cfg.think_min > 0 or cfg.think_max > 0):
            await asyncio.sleep(random.uniform(cfg.think_min, cfg.think_max))


# --------------------------------------------------------------------------- #
# Live progress reporter
# --------------------------------------------------------------------------- #
async def reporter(state: RunState) -> None:
    while not state.stop.is_set():
        await asyncio.sleep(2.0)
        done = state.completed
        elapsed = max(time.monotonic() - state.start_time, 1e-6)
        oks = [r.latency for r in state.results if r.ok]
        errs = sum(1 for r in state.results if not r.ok)
        p50 = _pct(oks, 50)
        p95 = _pct(oks, 95)
        rps = done / elapsed
        target = ""
        if state.cfg.requests is not None:
            target = f"/{state.cfg.requests}"
        elif state.cfg.duration is not None:
            target = f"  {elapsed:.0f}/{state.cfg.duration:.0f}s"
        print(
            f"  … {done}{target} done | in-flight {state.in_flight} | "
            f"err {errs} | {rps:.2f} req/s | p50 {p50:.1f}s p95 {p95:.1f}s",
            flush=True,
        )


# --------------------------------------------------------------------------- #
# Stats helpers
# --------------------------------------------------------------------------- #
def _pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def summarize(state: RunState, wall: float) -> dict:
    results = state.results
    oks = [r for r in results if r.ok]
    lat = [r.latency for r in oks]
    status_counts: dict[str, int] = {}
    for r in results:
        key = str(r.status) if r.status else "no-response"
        status_counts[key] = status_counts.get(key, 0) + 1
    error_counts: dict[str, int] = {}
    for r in results:
        if not r.ok:
            # collapse similar errors (strip trailing detail after ':')
            k = r.error.split(":")[0] if r.error else "unknown"
            error_counts[k] = error_counts.get(k, 0) + 1

    summary = {
        "endpoint": state.cfg.endpoint,
        "base_url": state.cfg.base_url,
        "concurrency": state.cfg.concurrency,
        "wall_seconds": round(wall, 2),
        "total_requests": len(results),
        "successful": len(oks),
        "failed": len(results) - len(oks),
        "success_rate_pct": round(100.0 * len(oks) / len(results), 2) if results else 0.0,
        "throughput_req_per_s": round(len(results) / wall, 3) if wall > 0 else 0.0,
        "throughput_ok_per_s": round(len(oks) / wall, 3) if wall > 0 else 0.0,
        "latency_seconds": {
            "min": round(min(lat), 3) if lat else 0.0,
            "mean": round(statistics.mean(lat), 3) if lat else 0.0,
            "p50": round(_pct(lat, 50), 3),
            "p90": round(_pct(lat, 90), 3),
            "p95": round(_pct(lat, 95), 3),
            "p99": round(_pct(lat, 99), 3),
            "max": round(max(lat), 3) if lat else 0.0,
        },
        "mean_answer_chars": round(statistics.mean([r.answer_chars for r in oks]), 1) if oks else 0.0,
        "status_counts": status_counts,
        "error_counts": error_counts,
    }
    return summary


def print_summary(summary: dict) -> None:
    lat = summary["latency_seconds"]
    print("\n" + "=" * 62)
    print(" LOAD TEST SUMMARY")
    print("=" * 62)
    print(f"  Endpoint            : {summary['endpoint']}  ({summary['base_url']})")
    print(f"  Concurrency         : {summary['concurrency']}")
    print(f"  Wall time           : {summary['wall_seconds']}s")
    print(f"  Total requests      : {summary['total_requests']}")
    print(f"  Successful          : {summary['successful']}  "
          f"({summary['success_rate_pct']}%)")
    print(f"  Failed              : {summary['failed']}")
    print(f"  Throughput (all)    : {summary['throughput_req_per_s']} req/s")
    print(f"  Throughput (ok)     : {summary['throughput_ok_per_s']} req/s")
    print("  Latency (successful requests):")
    print(f"      min  {lat['min']:8.2f}s     mean {lat['mean']:8.2f}s")
    print(f"      p50  {lat['p50']:8.2f}s     p90  {lat['p90']:8.2f}s")
    print(f"      p95  {lat['p95']:8.2f}s     p99  {lat['p99']:8.2f}s")
    print(f"      max  {lat['max']:8.2f}s")
    print(f"  Mean answer length  : {summary['mean_answer_chars']} chars")
    print(f"  Status codes        : {summary['status_counts']}")
    if summary["error_counts"]:
        print(f"  Errors              : {summary['error_counts']}")
    print("=" * 62)


def write_csv(path: str, results: list[RequestResult]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["seq", "worker", "endpoint", "started_epoch", "latency_s",
             "status", "ok", "resp_bytes", "answer_chars", "error", "question"]
        )
        for r in sorted(results, key=lambda x: x.started):
            writer.writerow(
                [r.seq, r.worker, r.endpoint, f"{r.started:.3f}", f"{r.latency:.3f}",
                 r.status, int(r.ok), r.resp_bytes, r.answer_chars,
                 r.error, r.question]
            )


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
async def run(cfg: Config, token: str) -> dict:
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    limits = httpx.Limits(
        max_connections=cfg.concurrency + 5,
        max_keepalive_connections=cfg.concurrency + 5,
    )
    timeout = httpx.Timeout(cfg.timeout, connect=15.0)

    state = RunState(cfg)
    async with httpx.AsyncClient(
        headers=headers, timeout=timeout, limits=limits
    ) as client:
        state.start_time = time.monotonic()
        rep = asyncio.create_task(reporter(state))
        workers = [
            asyncio.create_task(worker(i, client, state))
            for i in range(cfg.concurrency)
        ]
        # Duration guard: trip the stop event when time is up.
        if cfg.duration is not None:
            async def _deadline() -> None:
                await asyncio.sleep(cfg.duration)
                state.stop.set()
            deadline = asyncio.create_task(_deadline())
        else:
            deadline = None

        try:
            await asyncio.gather(*workers)
        except KeyboardInterrupt:  # pragma: no cover
            state.stop.set()
        finally:
            state.stop.set()
            rep.cancel()
            if deadline:
                deadline.cancel()
        wall = time.monotonic() - state.start_time

    return summarize(state, wall), state.results


def load_questions(path: Optional[str]) -> list[str]:
    if not path:
        return DEFAULT_QUESTIONS
    with open(path, encoding="utf-8") as f:
        qs = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    if not qs:
        raise SystemExit(f"No questions found in {path}")
    return qs


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Load tester for the Chat RAG AI Agent chat API.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--base-url", default=os.getenv("LOADTEST_BASE_URL", "http://localhost:8000"),
                   help="API base URL.")
    p.add_argument("--endpoint", choices=["message", "ask"], default="message",
                   help="Which chat endpoint to hit.")

    auth = p.add_argument_group("authentication")
    auth.add_argument("--token", default=os.getenv("LOADTEST_TOKEN"),
                      help="JWT access token (skips login). Or set LOADTEST_TOKEN.")
    auth.add_argument("--email", default=os.getenv("LOADTEST_EMAIL"),
                      help="Login email (used if --token not given).")
    auth.add_argument("--password", default=os.getenv("LOADTEST_PASSWORD"),
                      help="Login password.")

    load = p.add_argument_group("load shape")
    load.add_argument("--concurrency", "-c", type=int, default=4,
                      help="Number of concurrent virtual users.")
    load.add_argument("--requests", "-n", type=int, default=None,
                      help="Total number of requests to send.")
    load.add_argument("--duration", "-d", type=float, default=None,
                      help="Run for this many seconds (alternative to --requests).")
    load.add_argument("--ramp", type=float, default=0.0,
                      help="Seconds to ramp all users up over.")
    load.add_argument("--think-min", type=float, default=0.0,
                      help="Min think-time (s) between a user's requests.")
    load.add_argument("--think-max", type=float, default=0.0,
                      help="Max think-time (s) between a user's requests.")
    load.add_argument("--timeout", type=float, default=300.0,
                      help="Per-request timeout in seconds (CPU LLM is slow).")
    load.add_argument("--conversation", action="store_true",
                      help="Reuse conversation_id per user (multi-turn) on /chat/message.")

    p.add_argument("--questions", default=None,
                   help="Path to a newline-delimited questions file.")
    p.add_argument("--out", default=None, help="Write per-request results as CSV here.")
    p.add_argument("--summary-out", default=None, help="Write summary JSON here.")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Print each request as it completes.")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    if args.requests is None and args.duration is None:
        args.requests = 20  # sensible default
    if args.think_max < args.think_min:
        args.think_max = args.think_min

    cfg = Config(
        base_url=args.base_url,
        endpoint=args.endpoint,
        concurrency=max(1, args.concurrency),
        requests=args.requests,
        duration=args.duration,
        ramp=max(0.0, args.ramp),
        timeout=args.timeout,
        think_min=max(0.0, args.think_min),
        think_max=max(0.0, args.think_max),
        conversation=args.conversation,
        questions=load_questions(args.questions),
        verbose=args.verbose,
    )

    # Resolve auth.
    token = args.token
    if not token:
        if not (args.email and args.password):
            raise SystemExit(
                "Provide --token, or --email and --password to log in.\n"
                "(env: LOADTEST_TOKEN, or LOADTEST_EMAIL + LOADTEST_PASSWORD)"
            )
        print(f"Logging in as {args.email} …")
        token = asyncio.run(login(cfg.base_url, args.email, args.password, cfg.timeout))
        print("Login OK.")

    shape = (f"{cfg.requests} requests" if cfg.requests is not None
             else f"{cfg.duration:.0f}s duration")
    print(
        f"Starting load test → {cfg.base_url} /chat/{cfg.endpoint} | "
        f"concurrency={cfg.concurrency} | {shape} | timeout={cfg.timeout:.0f}s"
        + (f" | ramp={cfg.ramp:.0f}s" if cfg.ramp else "")
    )

    try:
        summary, results = asyncio.run(run(cfg, token))
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130

    print_summary(summary)

    if args.out:
        write_csv(args.out, results)
        print(f"  Per-request CSV     → {args.out}")
    if args.summary_out:
        with open(args.summary_out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"  Summary JSON        → {args.summary_out}")

    # Non-zero exit if anything failed, so it can gate CI.
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
