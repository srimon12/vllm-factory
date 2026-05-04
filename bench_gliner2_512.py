"""
Clean throughput benchmark: deberta_gliner2 with 512-token NER texts.
4 vLLM instances on RTX A5000, round-robin load balancing.
"""

import asyncio
import os
import signal
import subprocess
import sys
import time

import aiohttp

# ── Config ──
NUM_INSTANCES = 4
PORTS = [9100, 9101, 9102, 9103]
MODEL_ID = "/tmp/gliner2-vllm"
IO_PLUGIN = "deberta_gliner2_io"
LABELS = ["person", "organization", "location", "date", "event"]
TARGET_TOKENS = 512
NUM_EXAMPLES = 512
WARMUP = 200
CONCURRENCY_LEVELS = [4, 8, 16, 32, 64, 128, 256]
COST_PER_HOUR = 0.28
GPU_UTIL_PER_INSTANCE = 0.22


def generate_512_token_texts(n: int = NUM_EXAMPLES) -> list[str]:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID)

    seed_words = (
        "Max Mustermann the lead research engineer at Siemens AG headquartered "
        "in Munich presented results on neural network optimization at the "
        "International Conference on Machine Learning held in Vienna Austria "
        "on 15 January 2025 together with colleagues from Google DeepMind and "
        "OpenAI and partners at Microsoft Research in Redmond Washington and "
        "the European Organization for Nuclear Research CERN in Geneva Switzerland"
    ).split()

    texts = []
    for i in range(n):
        words = []
        while True:
            words.extend(seed_words)
            trial = f"Sample{i} " + " ".join(words)
            toks = tok(trial, truncation=False, add_special_tokens=True)["input_ids"]
            if len(toks) >= TARGET_TOKENS:
                lo, hi = 0, len(words)
                while lo < hi:
                    mid = (lo + hi) // 2
                    trial = f"Sample{i} " + " ".join(words[:mid])
                    if len(tok(trial, truncation=False, add_special_tokens=True)["input_ids"]) < TARGET_TOKENS:
                        lo = mid + 1
                    else:
                        hi = mid
                text = f"Sample{i} " + " ".join(words[:lo])
                texts.append(text)
                break
    return texts


def build_dataset() -> list[dict]:
    texts = generate_512_token_texts()
    return [{"text": t, "labels": LABELS} for t in texts]


# ── Server lifecycle ──

def start_instance(port: int, gpu_util: float) -> subprocess.Popen:
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL_ID,
        "--io-processor-plugin", IO_PLUGIN,
        "--port", str(port),
        "--trust-remote-code",
        "--dtype", "bfloat16",
        "--enforce-eager",
        "--no-enable-prefix-caching",
        "--no-enable-chunked-prefill",
        "--gpu-memory-utilization", str(gpu_util),
        "--disable-log-stats",
        "--uvicorn-log-level", "warning",
    ]
    env = os.environ.copy()
    env["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
    log = open(f"/tmp/gliner2_bench_{port}.log", "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                            env=env, text=True, start_new_session=True)
    return proc


def wait_instance_healthy(port: int, timeout: int = 600) -> bool:
    import requests
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = requests.get(f"http://localhost:{port}/health", timeout=5)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(3)
    return False


def kill_all(procs: list[subprocess.Popen]):
    for proc in procs:
        if proc.poll() is None:
            try:
                pgid = os.getpgid(proc.pid)
                os.killpg(pgid, signal.SIGTERM)
                proc.wait(timeout=15)
            except Exception:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    proc.wait(5)
                except Exception:
                    pass
    time.sleep(2)


# ── Async load test with round-robin ──

async def send_request(session, port, data):
    body = {"model": MODEL_ID, "data": data, "task": "plugin"}
    url = f"http://localhost:{port}/pooling"
    t0 = time.perf_counter()
    async with session.post(url, json=body) as resp:
        raw = await resp.read()
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status} on :{port}: {raw[:300]}")
    return (time.perf_counter() - t0) * 1000


async def warmup_requests(dataset, n, ports):
    conn = aiohttp.TCPConnector(limit=64)
    async with aiohttp.ClientSession(connector=conn) as session:
        sem = asyncio.Semaphore(32)

        async def bounded(i):
            async with sem:
                port = ports[i % len(ports)]
                return await send_request(session, port, dataset[i % len(dataset)])

        await asyncio.gather(*[bounded(i) for i in range(n)])


async def run_saturate(dataset, concurrency, num_reqs, ports):
    conn = aiohttp.TCPConnector(limit=max(16, concurrency * 2))
    async with aiohttp.ClientSession(connector=conn) as session:
        sem = asyncio.Semaphore(concurrency)
        t0 = time.perf_counter()

        async def bounded(i):
            async with sem:
                port = ports[i % len(ports)]
                return await send_request(session, port, dataset[i % len(dataset)])

        latencies = await asyncio.gather(*[bounded(i) for i in range(num_reqs)])
        elapsed = time.perf_counter() - t0

    lat = sorted(latencies)
    n = len(lat)
    return {
        "req_per_s": num_reqs / elapsed,
        "p50_ms": lat[int(n * 0.50)],
        "p95_ms": lat[int(n * 0.95)],
        "p99_ms": lat[int(n * 0.99)],
    }


# ── Main ──

def main():
    print("=" * 65)
    print("  GLiNER2 throughput benchmark — 512-token NER texts")
    print(f"  {NUM_INSTANCES} instances  |  RTX A5000  |  vLLM Factory")
    print("=" * 65)

    # Prepare model
    print("\n[1/5] Preparing model...")
    from pathlib import Path
    if not (Path(MODEL_ID) / "config.json").exists():
        from plugins.deberta_gliner2.parity_test import phase_prepare
        phase_prepare()
    else:
        print("  Model dir exists.")

    # Generate dataset
    print("\n[2/5] Generating 512-token dataset...")
    dataset = build_dataset()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    sample_lens = [len(tok(d["text"], truncation=False, add_special_tokens=True)["input_ids"])
                   for d in dataset[:5]]
    print(f"  {len(dataset)} examples, token lengths (first 5): {sample_lens}")

    # Start instances
    print(f"\n[3/5] Starting {NUM_INSTANCES} vLLM instances "
          f"(gpu_util={GPU_UTIL_PER_INSTANCE} each)...")
    procs = []
    for port in PORTS:
        print(f"  Launching instance on port {port}...")
        procs.append(start_instance(port, GPU_UTIL_PER_INSTANCE))

    try:
        for port in PORTS:
            print(f"  Waiting for :{port}...", end=" ", flush=True)
            if not wait_instance_healthy(port):
                print("FAILED")
                with open(f"/tmp/gliner2_bench_{port}.log") as f:
                    print(f.read()[-2000:])
                raise RuntimeError(f"Instance on port {port} failed to start")
            print("healthy")

        # Warmup
        print(f"\n[4/5] Warmup ({WARMUP} requests across all instances)...")
        asyncio.run(warmup_requests(dataset, WARMUP, PORTS))
        print("  Done.")

        # Throughput sweep
        print(f"\n[5/5] Throughput sweep ({NUM_EXAMPLES} req per level, "
              f"round-robin across {NUM_INSTANCES} instances)")
        print(f"{'Concurrency':>12} {'req/s':>10} {'p50 ms':>10} "
              f"{'p95 ms':>10} {'p99 ms':>10}")
        print("-" * 58)

        best_rps = 0
        best_c = 0
        for c in CONCURRENCY_LEVELS:
            result = asyncio.run(run_saturate(dataset, c, NUM_EXAMPLES, PORTS))
            rps = result["req_per_s"]
            print(f"{c:>12} {rps:>10.1f} {result['p50_ms']:>10.1f} "
                  f"{result['p95_ms']:>10.1f} {result['p99_ms']:>10.1f}")
            if rps > best_rps:
                best_rps = rps
                best_c = c

        print("-" * 58)
        print(f"\n  PEAK: {best_rps:.1f} req/s at concurrency={best_c}")
        print(f"  Setup: {NUM_INSTANCES} instances, {TARGET_TOKENS} tokens/request")

        docs_per_hour = best_rps * 3600
        tokens_per_hour = docs_per_hour * TARGET_TOKENS
        cost_per_1m_docs = (COST_PER_HOUR / docs_per_hour) * 1_000_000
        cost_per_1m_tokens = (COST_PER_HOUR / tokens_per_hour) * 1_000_000

        print(f"\n  {'='*50}")
        print(f"  Unit Economics (RTX A5000 @ ${COST_PER_HOUR}/hr)")
        print(f"  {'='*50}")
        print(f"  {best_rps:.0f} req/s (1 req = 1 page ≈ {TARGET_TOKENS} tokens)")
        print(f"  ~{docs_per_hour:,.0f} pages/hour")
        print(f"  ~${cost_per_1m_docs:.2f} per 1M pages")
        print(f"  ~${cost_per_1m_tokens:.5f} per 1M input tokens")

    finally:
        print("\nShutting down all instances...")
        kill_all(procs)
        print("Done.")


if __name__ == "__main__":
    os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
    main()
