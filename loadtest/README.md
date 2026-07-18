# Chat API Load Testing

Tools to load test the Chat RAG AI Agent chat endpoints on `http://localhost:8000`.

Two options are included:

| File | Use it when |
|------|-------------|
| `load_test.py` | Headless runs, CI gates, scripted scenarios, CSV/JSON output. **No extra deps** (uses `httpx`, already in the project). |
| `locustfile.py` | You want a live web dashboard and interactive user-count control. Requires `pip install locust`. |

## What it hits

- `POST /chat/message` — conversation-aware: `{"message": ..., "conversation_id": optional}`
- `POST /chat/ask` — single-shot: `{"question": ...}`

Both require a JWT: `Authorization: Bearer <token>`. The tools log in via
`POST /auth/login` (`{"email", "password"}`) or accept a token directly.

## Reality check before you turn the dial up

The backend answers with a **local Ollama LLM (`qwen3:8b`) on CPU** with
`OLLAMA_NUM_PARALLEL=1`, so answer generation is effectively **serialized** on the
server. Adding virtual users does **not** multiply throughput — it grows a queue.
So:

- Expect **low RPS** and **per-request latency of tens of seconds** (sometimes minutes).
- Keep `--timeout` high (default `300s`).
- The useful signal is **how p95/p99 latency degrades as concurrency rises**, and at
  what concurrency requests start timing out — not raw req/s.
- Start small: `--concurrency 2`, then `4`, `8`, and watch the percentiles.

## Quick start (`load_test.py`)

```bash
# 20 requests, 4 concurrent users, logging in with a real account
python loadtest/load_test.py \
    --email you@example.com --password secret \
    --concurrency 4 --requests 20
```

Using a pre-issued token instead of logging in:

```bash
export LOADTEST_TOKEN="eyJhbGci..."
python loadtest/load_test.py --token "$LOADTEST_TOKEN" -c 4 -n 20
```

Run for a fixed duration against the single-shot endpoint, with a ramp:

```bash
python loadtest/load_test.py \
    --email you@example.com --password secret \
    --endpoint ask --concurrency 8 --duration 120 --ramp 20 \
    --out results.csv --summary-out summary.json
```

Multi-turn conversations (reuse `conversation_id` per user) with your own questions:

```bash
python loadtest/load_test.py \
    --email you@example.com --password secret \
    --endpoint message --conversation \
    --questions loadtest/questions.txt \
    --concurrency 6 --duration 300 -v
```

On Windows PowerShell, set env vars with `$env:LOADTEST_TOKEN = "..."`.

### Key options

| Flag | Meaning | Default |
|------|---------|---------|
| `--base-url` | API base URL | `http://localhost:8000` |
| `--endpoint` | `message` or `ask` | `message` |
| `--token` / `--email` `--password` | Auth (token skips login) | env fallbacks |
| `-c, --concurrency` | Concurrent virtual users | `4` |
| `-n, --requests` | Total requests to send | `20` (if no `--duration`) |
| `-d, --duration` | Run for N seconds instead | — |
| `--ramp` | Seconds to start all users over | `0` |
| `--think-min/--think-max` | Pause between a user's requests | `0` |
| `--timeout` | Per-request timeout (s) | `300` |
| `--conversation` | Reuse `conversation_id` per user | off |
| `--questions` | Newline-delimited questions file | built-in pool |
| `--out` | Per-request results CSV | — |
| `--summary-out` | Summary JSON | — |
| `-v, --verbose` | Print each request as it finishes | off |

Env fallbacks: `LOADTEST_BASE_URL`, `LOADTEST_TOKEN`, `LOADTEST_EMAIL`,
`LOADTEST_PASSWORD`.

### Output

Live progress every 2s, then a summary with success rate, throughput, and latency
percentiles (min/mean/p50/p90/p95/p99/max), status-code counts, and an error
breakdown. Exit code is non-zero if any request failed (handy for CI gates).

```
==============================================================
 LOAD TEST SUMMARY
==============================================================
  Endpoint            : message  (http://localhost:8000)
  Concurrency         : 4
  Wall time           : 143.2s
  Total requests      : 20
  Successful          : 20  (100.0%)
  ...
  Latency (successful requests):
      min     8.31s     mean    27.04s
      p50    25.90s     p90    41.12s
      p95    44.83s     p99    46.10s
      max    46.10s
==============================================================
```

## Quick start (Locust dashboard)

```bash
pip install locust
LOADTEST_EMAIL=you@example.com LOADTEST_PASSWORD=secret \
    locust -f loadtest/locustfile.py --host http://localhost:8000
```

Open <http://localhost:8089>, set the number of users and spawn rate, and watch the
charts. Set `LOADTEST_ENDPOINT=ask` to hit `/chat/ask`, or `LOADTEST_TOKEN` to skip
login.

Headless Locust (no UI, e.g. for CI):

```bash
LOADTEST_EMAIL=you@example.com LOADTEST_PASSWORD=secret \
    locust -f loadtest/locustfile.py --host http://localhost:8000 \
    --headless -u 8 -r 2 -t 3m --csv loadtest_run
```

## Tips

- Point `--questions` at questions that actually match your ingested documents so
  retrieval + generation are exercised realistically (empty-retrieval answers are
  faster and won't reflect real load).
- To find the server's saturation point, sweep concurrency: run the same `-n` at
  `-c 1,2,4,8,16` and compare p95. The knee is where latency explodes / timeouts start.
- Watch server-side resources (CPU, RAM, Ollama) during the run — the bottleneck is
  almost always the LLM, not the HTTP layer.
```
