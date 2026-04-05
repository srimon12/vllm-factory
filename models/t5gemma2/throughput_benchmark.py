#!/usr/bin/env python3
"""T5Gemma2 batch throughput benchmark.

Measures end-to-end forward throughput at batch sizes 1, 8, 16, 32, 64
using the same text inputs from the parity reference file.

Uses CUDA event timing, excludes Triton JIT warmup.

Usage:
    python models/t5gemma2/throughput_benchmark.py
"""

from __future__ import annotations

import gc
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

MODEL_NAME = "google/t5gemma-2-270m-270m"
REF_FILE = "/tmp/t5gemma2-reference.pt"
WARMUP_ITERS = 10
BENCH_ITERS = 50
BATCH_SIZES = [1, 8, 16, 32, 64]


def init_vllm():
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    from vllm.config import CompilationConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed.parallel_state import (
        ensure_model_parallel_initialized,
        init_distributed_environment,
    )
    vllm_config = VllmConfig(compilation_config=CompilationConfig())
    ctx = set_current_vllm_config(vllm_config)
    ctx.__enter__()
    init_distributed_environment(world_size=1, rank=0, local_rank=0, distributed_init_method="env://")
    ensure_model_parallel_initialized(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    return ctx


def load_model(config):
    from models.t5gemma2.parity_test import _load_weights_from_hub
    from models.t5gemma2.t5gemma2_model import T5Gemma2ForConditionalGeneration
    model = T5Gemma2ForConditionalGeneration(config)
    model.eval()
    model.load_weights(_load_weights_from_hub(MODEL_NAME))
    model = model.to("cuda").float()
    return model


def tile_to_batch(ref_block: dict, batch_size: int, device: torch.device):
    """Replicate the 2-sample reference inputs up to batch_size."""
    input_ids = ref_block["input_ids"].to(device)
    attention_mask = ref_block["attention_mask"].to(device)
    dec_ids = ref_block["decoder_input_ids"].to(device)
    dec_mask = ref_block["decoder_attention_mask"].to(device)

    base_bs = input_ids.shape[0]
    repeats = (batch_size + base_bs - 1) // base_bs

    input_ids = input_ids.repeat(repeats, 1)[:batch_size]
    attention_mask = attention_mask.repeat(repeats, 1)[:batch_size]
    dec_ids = dec_ids.repeat(repeats, 1)[:batch_size]
    dec_mask = dec_mask.repeat(repeats, 1)[:batch_size]

    return input_ids, attention_mask, dec_ids, dec_mask


def run_forward(model, input_ids, attention_mask, dec_ids, dec_mask):
    enc_out = model.get_encoder_outputs(input_ids, attention_mask=attention_mask)
    dec_out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        decoder_input_ids=dec_ids,
        decoder_attention_mask=dec_mask,
    )
    logits = model.compute_logits(dec_out)
    return logits


def benchmark_batch(model, input_ids, attention_mask, dec_ids, dec_mask):
    for _ in range(WARMUP_ITERS):
        with torch.no_grad():
            run_forward(model, input_ids, attention_mask, dec_ids, dec_mask)
        torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start_event.record()
    for _ in range(BENCH_ITERS):
        with torch.no_grad():
            run_forward(model, input_ids, attention_mask, dec_ids, dec_mask)
    end_event.record()
    torch.cuda.synchronize()

    elapsed_ms = start_event.elapsed_time(end_event)
    avg_ms = elapsed_ms / BENCH_ITERS
    return avg_ms


def main():
    print("=" * 80)
    print("T5Gemma2 Batch Throughput Benchmark")
    print("=" * 80)
    print(f"  Model:       {MODEL_NAME}")
    print(f"  Warmup:      {WARMUP_ITERS} iters")
    print(f"  Bench:       {BENCH_ITERS} iters")
    print(f"  Batch sizes: {BATCH_SIZES}")

    ctx = init_vllm()
    device = torch.device("cuda")

    ref_data = torch.load(REF_FILE, map_location="cpu", weights_only=False)
    from models.t5gemma2.parity_test import _load_config_from_ref
    config = _load_config_from_ref(ref_data)

    enc_seq = ref_data["text"]["input_ids"].shape[1]
    dec_seq = ref_data["text"]["decoder_input_ids"].shape[1]
    print(f"  Encoder seq:  {enc_seq}")
    print(f"  Decoder seq:  {dec_seq}")

    # Ensure optimized path with flash + norm_rope (embed off by default)
    os.environ.pop("VLLM_FACTORY_T5GEMMA2_REFERENCE_PATH", None)
    os.environ.pop("T5GEMMA2_NO_FLASH", None)
    os.environ.pop("T5GEMMA2_NO_FUSED_NORM_ROPE", None)

    configs = [
        ("optimized (flash+norm_rope)", False),
        ("baseline  (no kernels)",      True),
    ]

    all_results = {}

    for config_label, use_baseline in configs:
        print(f"\n{'=' * 80}")
        print(f"  Config: {config_label}")
        print(f"{'=' * 80}")

        if use_baseline:
            os.environ["T5GEMMA2_NO_FLASH"] = "1"
            os.environ["T5GEMMA2_NO_FUSED_NORM_ROPE"] = "1"
        else:
            os.environ.pop("T5GEMMA2_NO_FLASH", None)
            os.environ.pop("T5GEMMA2_NO_FUSED_NORM_ROPE", None)

        model = load_model(config)
        results = []

        for bs in BATCH_SIZES:
            input_ids, attention_mask, dec_ids, dec_mask = tile_to_batch(
                ref_data["text"], bs, device
            )
            try:
                avg_ms = benchmark_batch(model, input_ids, attention_mask, dec_ids, dec_mask)
                throughput = bs / (avg_ms / 1000.0)
                per_sample_ms = avg_ms / bs
                results.append({
                    "bs": bs,
                    "avg_ms": avg_ms,
                    "throughput": throughput,
                    "per_sample_ms": per_sample_ms,
                })
                print(f"  bs={bs:<4d}  {avg_ms:>8.2f} ms/fwd  "
                      f"{throughput:>8.1f} samples/s  "
                      f"{per_sample_ms:>8.2f} ms/sample")
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    results.append({
                        "bs": bs,
                        "avg_ms": float("inf"),
                        "throughput": 0.0,
                        "per_sample_ms": float("inf"),
                        "oom": True,
                    })
                    print(f"  bs={bs:<4d}  OOM")
                    break
                raise

        all_results[config_label] = results

        del model
        gc.collect()
        torch.cuda.empty_cache()

    # Summary table
    print(f"\n\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print(f"  Encoder seq_len={enc_seq}  Decoder seq_len={dec_seq}")
    print()

    opt_results = all_results.get("optimized (flash+norm_rope)", [])
    base_results = all_results.get("baseline  (no kernels)", [])

    opt_by_bs = {r["bs"]: r for r in opt_results}
    base_by_bs = {r["bs"]: r for r in base_results}

    print(f"{'BS':>4} | {'Optimized':>24} | {'Baseline':>24} | {'Speedup':>8}")
    print(f"{'':>4} | {'ms/fwd':>8} {'samp/s':>8} {'ms/samp':>7} | "
          f"{'ms/fwd':>8} {'samp/s':>8} {'ms/samp':>7} | {'':>8}")
    print("─" * 80)

    for bs in BATCH_SIZES:
        o = opt_by_bs.get(bs)
        b = base_by_bs.get(bs)

        def _fmt(r):
            if r is None or r.get("oom"):
                return f"{'OOM':>8} {'—':>8} {'—':>7}"
            return f"{r['avg_ms']:>8.2f} {r['throughput']:>8.1f} {r['per_sample_ms']:>7.2f}"

        speedup = ""
        if o and b and not o.get("oom") and not b.get("oom"):
            s = b["avg_ms"] / o["avg_ms"]
            speedup = f"{s:.2f}x"

        print(f"{bs:>4} | {_fmt(o)} | {_fmt(b)} | {speedup:>8}")

    # Peak throughput
    print()
    for label in all_results:
        results = all_results[label]
        valid = [r for r in results if not r.get("oom")]
        if valid:
            best = max(valid, key=lambda r: r["throughput"])
            print(f"  Peak [{label}]: {best['throughput']:.1f} samples/s at bs={best['bs']}")

    print()
    ctx.__exit__(None, None, None)


if __name__ == "__main__":
    main()
