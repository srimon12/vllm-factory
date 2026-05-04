#!/usr/bin/env python3
"""
PR #4 Acceptance Test — GLiNER2 labels + schema via vLLM serve

Sends the exact examples from https://github.com/fastino-ai/GLiNER2
through the live vLLM HTTP /pooling endpoint and validates output
structure and content.

Usage:
    python test_pr4_gliner2_acceptance.py
"""

import asyncio
import json
import os
import signal
import subprocess
import sys
import time

import aiohttp

MODEL_DIR = "/tmp/gliner2-vllm"
MODEL_NAME = MODEL_DIR
IO_PLUGIN = "deberta_gliner2_io"
PORT = 9200
BASE_URL = f"http://localhost:{PORT}"
GPU_UTIL = "0.60"

PASSED = 0
FAILED = 0


def header(name: str):
    print(f"\n{'─'*60}\n  {name}\n{'─'*60}")


def check(name: str, condition: bool, detail: str = ""):
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  PASS  {name}")
    else:
        FAILED += 1
        print(f"  FAIL  {name}  {detail}")


async def post_pooling(session: aiohttp.ClientSession, data: dict) -> dict:
    body = {"model": MODEL_NAME, "data": data, "task": "plugin"}
    async with session.post(f"{BASE_URL}/pooling", json=body, timeout=aiohttp.ClientTimeout(total=30)) as resp:
        if resp.status != 200:
            text = await resp.text()
            raise RuntimeError(f"HTTP {resp.status}: {text[:500]}")
        envelope = await resp.json()
        return envelope.get("data", envelope)


# ── Test cases ────────────────────────────────────────────────


async def test_1_entities_labels_list(session):
    """Entity extraction via labels (list) — GLiNER2 README §1"""
    header("Test 1: Entity extraction via labels (list)")
    text = "Apple CEO Tim Cook announced iPhone 15 in Cupertino yesterday."
    resp = await post_pooling(session, {
        "text": text,
        "labels": ["company", "person", "product", "location"],
    })
    print(f"  Response: {json.dumps(resp, indent=2, default=str)}")
    ents = resp.get("entities", {})
    check("has 'entities' key", "entities" in resp)
    check("person found", len(ents.get("person", [])) > 0)
    check("'Tim Cook' in person", any("Tim Cook" in str(e) for e in ents.get("person", [])))
    check("company found", len(ents.get("company", [])) > 0)
    check("'Apple' in company", any("Apple" in str(e) for e in ents.get("company", [])))


async def test_2_entities_labels_dict(session):
    """Entity extraction via labels (dict with descriptions) — GLiNER2 README §1"""
    header("Test 2: Entity extraction via labels (dict w/ descriptions)")
    text = "Patient received 400mg ibuprofen for severe headache at 2 PM."
    resp = await post_pooling(session, {
        "text": text,
        "labels": {
            "medication": "Names of drugs, medications, or pharmaceutical substances",
            "dosage": "Specific amounts like '400mg', '2 tablets', or '5ml'",
            "symptom": "Medical symptoms, conditions, or patient complaints",
            "time": "Time references like '2 PM', 'morning', or 'after lunch'",
        },
    })
    print(f"  Response: {json.dumps(resp, indent=2, default=str)}")
    ents = resp.get("entities", {})
    check("has 'entities' key", "entities" in resp)
    check("medication found", len(ents.get("medication", [])) > 0)
    check("'ibuprofen' in medication", any("ibuprofen" in str(e).lower() for e in ents.get("medication", [])))
    return resp


async def test_3_entities_schema(session):
    """Entity extraction via schema (same data) — GLiNER2 README §1"""
    header("Test 3: Entity extraction via schema (same text as test 2)")
    text = "Patient received 400mg ibuprofen for severe headache at 2 PM."
    resp = await post_pooling(session, {
        "text": text,
        "schema": {
            "entities": {
                "medication": "Names of drugs, medications, or pharmaceutical substances",
                "dosage": "Specific amounts like '400mg', '2 tablets', or '5ml'",
                "symptom": "Medical symptoms, conditions, or patient complaints",
                "time": "Time references like '2 PM', 'morning', or 'after lunch'",
            }
        },
    })
    print(f"  Response: {json.dumps(resp, indent=2, default=str)}")
    ents = resp.get("entities", {})
    check("has 'entities' key", "entities" in resp)
    check("medication found", len(ents.get("medication", [])) > 0)
    check("'ibuprofen' in medication", any("ibuprofen" in str(e).lower() for e in ents.get("medication", [])))


async def test_4_classification(session):
    """Classification via schema — GLiNER2 README §2"""
    header("Test 4: Classification via schema")
    text = "This laptop has amazing performance but terrible battery life!"
    resp = await post_pooling(session, {
        "text": text,
        "schema": {
            "classifications": [{
                "task": "sentiment",
                "labels": ["positive", "negative", "neutral"],
            }]
        },
    })
    print(f"  Response: {json.dumps(resp, indent=2, default=str)}")
    check("has 'sentiment' key", "sentiment" in resp)
    label = resp.get("sentiment", "")
    check("sentiment is valid label", label in ("positive", "negative", "neutral"),
          f"got '{label}'")


async def test_5_relations(session):
    """Relation extraction via schema — GLiNER2 README §4"""
    header("Test 5: Relation extraction via schema")
    text = "John works for Apple Inc. and lives in San Francisco. Apple Inc. is located in Cupertino."
    resp = await post_pooling(session, {
        "text": text,
        "schema": {
            "relations": ["works_for", "lives_in", "located_in"],
        },
    })
    print(f"  Response: {json.dumps(resp, indent=2, default=str)}")
    check("has 'relation_extraction' key", "relation_extraction" in resp)
    rels = resp.get("relation_extraction", {})
    check("at least one relation type returned", len(rels) > 0)


async def test_6_structured_extraction(session):
    """Structured data extraction via schema — GLiNER2 README §3"""
    header("Test 6: Structured extraction via schema")
    text = "iPhone 15 Pro Max with 256GB storage, A17 Pro chip, priced at $1199. Available in titanium and black colors."
    resp = await post_pooling(session, {
        "text": text,
        "schema": {
            "structures": {
                "product": {
                    "fields": [
                        {"name": "name", "dtype": "str", "description": "Full product name and model"},
                        {"name": "storage", "dtype": "str", "description": "Storage capacity"},
                        {"name": "processor", "dtype": "str", "description": "Chip or processor"},
                        {"name": "price", "dtype": "str", "description": "Product price"},
                    ]
                }
            }
        },
    })
    print(f"  Response: {json.dumps(resp, indent=2, default=str)}")
    check("has 'product' key", "product" in resp)
    products = resp.get("product", [])
    check("product is a list", isinstance(products, list))
    if products:
        first = products[0]
        check("first product has 'name' field", "name" in first)


async def test_7_mixed_schema(session):
    """Mixed multi-task schema — GLiNER2 README §5"""
    header("Test 7: Mixed multi-task schema")
    text = (
        "Apple CEO Tim Cook unveiled the revolutionary iPhone 15 Pro for $999. "
        "The device features an A17 Pro chip and titanium design. "
        "Tim Cook works for Apple, which is located in Cupertino."
    )
    resp = await post_pooling(session, {
        "text": text,
        "schema": {
            "entities": {
                "person": "Names of people, executives, or individuals",
                "company": "Organization, corporation, or business names",
                "product": "Products, services, or offerings mentioned",
            },
            "classifications": [
                {"task": "sentiment", "labels": ["positive", "negative", "neutral"]},
                {"task": "category", "labels": ["technology", "business", "finance", "healthcare"]},
            ],
            "relations": ["works_for", "located_in"],
            "structures": {
                "product_info": {
                    "fields": [
                        {"name": "name", "dtype": "str"},
                        {"name": "price", "dtype": "str"},
                    ]
                }
            },
        },
    })
    print(f"  Response: {json.dumps(resp, indent=2, default=str)}")
    check("has 'entities' key", "entities" in resp)
    check("has 'sentiment' key", "sentiment" in resp)
    check("has 'category' key", "category" in resp)
    check("has 'relation_extraction' key", "relation_extraction" in resp)
    check("has 'product_info' key", "product_info" in resp)


async def test_8_confidence_and_spans(session):
    """include_confidence + include_spans flags — GLiNER2 README §1"""
    header("Test 8: include_confidence + include_spans")
    text = "Apple CEO Tim Cook announced iPhone 15 in Cupertino yesterday."
    resp = await post_pooling(session, {
        "text": text,
        "labels": ["company", "person", "product", "location"],
        "include_confidence": True,
        "include_spans": True,
    })
    print(f"  Response: {json.dumps(resp, indent=2, default=str)}")
    ents = resp.get("entities", {})
    check("has 'entities' key", "entities" in resp)
    found_rich = False
    for label, items in ents.items():
        for item in items:
            if isinstance(item, dict) and "text" in item:
                found_rich = True
                check(f"'{label}' item has 'confidence'", "confidence" in item)
                check(f"'{label}' item has 'start'", "start" in item)
                check(f"'{label}' item has 'end'", "end" in item)
                break
        if found_rich:
            break
    check("at least one rich entity record found", found_rich)


# ── Server lifecycle ──────────────────────────────────────────


def ensure_model_dir():
    config_path = os.path.join(MODEL_DIR, "config.json")
    if os.path.exists(config_path):
        print(f"Model dir exists at {MODEL_DIR}")
        return
    print("Model dir not found — running parity_test.py --prepare ...")
    r = subprocess.run(
        [sys.executable, "plugins/deberta_gliner2/parity_test.py", "--prepare"],
        cwd="/workspace/vllm-factory",
    )
    if r.returncode != 0:
        print("FATAL: model preparation failed")
        sys.exit(1)


def start_server() -> subprocess.Popen:
    env = os.environ.copy()
    env["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL_DIR,
        "--io-processor-plugin", IO_PLUGIN,
        "--port", str(PORT),
        "--trust-remote-code",
        "--dtype", "bfloat16",
        "--enforce-eager",
        "--no-enable-prefix-caching",
        "--no-enable-chunked-prefill",
        "--gpu-memory-utilization", GPU_UTIL,
        "--disable-log-stats",
        "--uvicorn-log-level", "warning",
    ]
    log = open("/tmp/pr4_acceptance_server.log", "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                            env=env, text=True, start_new_session=True)
    print(f"Server PID={proc.pid} starting on port {PORT}...")
    return proc


async def wait_for_server(timeout: int = 180):
    t0 = time.time()
    async with aiohttp.ClientSession() as session:
        while time.time() - t0 < timeout:
            try:
                async with session.get(f"{BASE_URL}/health", timeout=aiohttp.ClientTimeout(total=2)) as r:
                    if r.status == 200:
                        elapsed = time.time() - t0
                        print(f"Server healthy after {elapsed:.0f}s")
                        return
            except Exception:
                pass
            await asyncio.sleep(3)
    raise TimeoutError(f"Server not healthy after {timeout}s")


def kill_server(proc: subprocess.Popen):
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        pass
    proc.wait(timeout=10)
    print("Server stopped.")


# ── Main ──────────────────────────────────────────────────────


async def run_tests():
    async with aiohttp.ClientSession() as session:
        await test_1_entities_labels_list(session)
        await test_2_entities_labels_dict(session)
        await test_3_entities_schema(session)
        await test_4_classification(session)
        await test_5_relations(session)
        await test_6_structured_extraction(session)
        await test_7_mixed_schema(session)
        await test_8_confidence_and_spans(session)


def main():
    global PASSED, FAILED

    ensure_model_dir()

    proc = start_server()
    try:
        asyncio.run(wait_for_server())
        asyncio.run(run_tests())
    except Exception as exc:
        print(f"\nFATAL: {exc}")
        FAILED += 1
    finally:
        kill_server(proc)

    print(f"\n{'='*60}")
    print(f"  RESULTS: {PASSED} passed, {FAILED} failed")
    print(f"{'='*60}")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
