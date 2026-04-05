#!/usr/bin/env python3
"""Benchmark each T5Gemma2 Triton kernel individually.

For each of the 3 kernels (flash_attention, fused_qk_norm_rope, fused_embed_scale_eoi),
this script measures:
  1. Parity: max/mean diff vs the non-kernel (SDPA/sequential) path
  2. Latency: end-to-end forward time with kernel ON vs OFF

Environment variables used as toggles:
  T5GEMMA2_NO_FLASH=1        -- disable flash attention kernel
  T5GEMMA2_NO_FUSED_NORM_ROPE=1 -- disable fused QK-norm+RoPE kernel
  T5GEMMA2_NO_FUSED_EMBED=1  -- disable fused embed scale+EOI kernel

Usage:
    python models/t5gemma2/kernel_benchmark.py
"""

from __future__ import annotations

import gc
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

MODEL_NAME = "google/t5gemma-2-270m-270m"

WARMUP_ITERS = 10
BENCH_ITERS = 50

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
    from models.t5gemma2.t5gemma2_model import T5Gemma2ForConditionalGeneration
    model = T5Gemma2ForConditionalGeneration(config)
    model.eval()
    from models.t5gemma2.parity_test import _load_weights_from_hub
    model.load_weights(_load_weights_from_hub(MODEL_NAME))
    model = model.to("cuda").float()
    return model


def prepare_inputs(device, ref_block):
    return (
        ref_block["input_ids"].to(device),
        ref_block["attention_mask"].to(device),
        ref_block["decoder_input_ids"].to(device),
        ref_block["decoder_attention_mask"].to(device),
    )


def set_kernel_flags(flash: bool, norm_rope: bool, embed: bool):
    for var, enabled in [
        ("T5GEMMA2_NO_FLASH", not flash),
        ("T5GEMMA2_NO_FUSED_NORM_ROPE", not norm_rope),
        ("T5GEMMA2_NO_FUSED_EMBED", not embed),
    ]:
        if enabled:
            os.environ[var] = "1"
        else:
            os.environ.pop(var, None)


def run_forward(model, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask):
    enc_out = model.get_encoder_outputs(input_ids, attention_mask=attention_mask)
    dec_out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        decoder_input_ids=decoder_input_ids,
        decoder_attention_mask=decoder_attention_mask,
    )
    logits = model.compute_logits(dec_out)
    return enc_out, dec_out, logits


def benchmark_latency(model, input_ids, attention_mask, dec_ids, dec_mask):
    # Warmup: triggers Triton JIT + fills CUDA caches.
    # Note: the parity forward already ran before this, so JIT is likely done,
    # but we still warm up to stabilize GPU clocks and memory allocators.
    for _ in range(WARMUP_ITERS):
        with torch.no_grad():
            run_forward(model, input_ids, attention_mask, dec_ids, dec_mask)
        torch.cuda.synchronize()

    # Use CUDA events for precise GPU-side timing (immune to host scheduling jitter)
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
    return elapsed_ms / BENCH_ITERS / 1000.0  # return seconds


def compare(name, ref, actual, mask=None):
    r, a = ref.float(), actual.float()
    if mask is not None:
        m = mask.view(*mask.shape, *([1] * (r.dim() - mask.dim()))).expand_as(r)
        r, a = r[m], a[m]
    if r.numel() == 0:
        return 0.0, 0.0
    d = (r - a).abs()
    return d.max().item(), d.mean().item()


def main():
    print("=" * 80)
    print("T5Gemma2 Kernel Benchmark: Individual Contribution Analysis")
    print("=" * 80)
    print(f"  Model:       {MODEL_NAME}")
    print(f"  Warmup:      {WARMUP_ITERS} iters")
    print(f"  Bench:       {BENCH_ITERS} iters")
    print()

    ctx = init_vllm()

    ref_data = torch.load("/tmp/t5gemma2-reference.pt", map_location="cpu", weights_only=False)
    from models.t5gemma2.parity_test import _load_config_from_ref
    config = _load_config_from_ref(ref_data)
    device = torch.device("cuda")

    input_ids, attention_mask, dec_ids, dec_mask = prepare_inputs(device, ref_data["text"])
    print(f"  Encoder seq len: {input_ids.shape[1]}")
    print(f"  Decoder seq len: {dec_ids.shape[1]}")

    # Configurations to test: (label, flash, norm_rope, embed)
    configs = [
        ("ALL OFF (baseline)",     False, False, False),
        ("ALL ON  (all kernels)",  True,  True,  True),
        ("flash ONLY",             True,  False, False),
        ("norm_rope ONLY",         False, True,  False),
        ("embed ONLY",             False, False, True),
        ("ALL ON - flash",         False, True,  True),
        ("ALL ON - norm_rope",     True,  False, True),
        ("ALL ON - embed",         True,  True,  False),
    ]

    # Always ensure we're on optimized path (not reference path)
    os.environ.pop("VLLM_FACTORY_T5GEMMA2_REFERENCE_PATH", None)

    results = []

    for label, flash, norm_rope, embed in configs:
        set_kernel_flags(flash, norm_rope, embed)
        flags = f"flash={'ON' if flash else 'off':>3}  norm_rope={'ON' if norm_rope else 'off':>3}  embed={'ON' if embed else 'off':>3}"
        print(f"\n{'─' * 80}")
        print(f"  Config: {label}")
        print(f"  Flags:  {flags}")
        print(f"{'─' * 80}")

        model = load_model(config)

        # Parity vs HF reference
        with torch.no_grad():
            enc_out, dec_out, logits = run_forward(
                model, input_ids, attention_mask, dec_ids, dec_mask
            )

        ref_enc = ref_data["text"]["encoder_hidden"].to(device)
        ref_dec = ref_data["text"]["decoder_hidden"].to(device)
        ref_logits = ref_data["text"]["logits"].to(device)
        enc_mask = attention_mask.bool()
        d_mask = dec_mask.bool()

        enc_max, enc_mean = compare("enc", ref_enc, enc_out, enc_mask)
        dec_max, dec_mean = compare("dec", ref_dec, dec_out, d_mask)
        log_max, log_mean = compare("logits", ref_logits, logits, d_mask)

        print(f"  Parity vs HF ref:")
        print(f"    encoder:  max_diff={enc_max:.6e}  mean_diff={enc_mean:.6e}")
        print(f"    decoder:  max_diff={dec_max:.6e}  mean_diff={dec_mean:.6e}")
        print(f"    logits:   max_diff={log_max:.6e}  mean_diff={log_mean:.6e}")

        # Latency
        lat = benchmark_latency(model, input_ids, attention_mask, dec_ids, dec_mask)
        print(f"  Latency:    {lat*1000:.2f} ms/forward")

        results.append({
            "label": label,
            "flash": flash,
            "norm_rope": norm_rope,
            "embed": embed,
            "enc_max_diff": enc_max,
            "dec_max_diff": dec_max,
            "logit_max_diff": log_max,
            "latency_ms": lat * 1000,
        })

        del model
        gc.collect()
        torch.cuda.empty_cache()

    # Longer sequence benchmark
    print(f"\n\n{'=' * 80}")
    print("LONGER SEQUENCE BENCHMARK (synthetic)")
    print(f"{'=' * 80}")
    for enc_len, dec_len in [(128, 64), (512, 128)]:
        print(f"\n  Encoder seq_len={enc_len}, Decoder seq_len={dec_len}")
        synth_input_ids = torch.randint(100, 30000, (1, enc_len), device=device)
        synth_enc_mask = torch.ones(1, enc_len, dtype=torch.bool, device=device)
        synth_dec_ids = torch.randint(100, 30000, (1, dec_len), device=device)
        synth_dec_mask = torch.ones(1, dec_len, dtype=torch.bool, device=device)

        for label, flash, norm_rope, embed in [
            ("ALL OFF", False, False, False),
            ("ALL ON", True, True, True),
            ("flash ONLY", True, False, False),
            ("norm_rope ONLY", False, True, False),
        ]:
            set_kernel_flags(flash, norm_rope, embed)
            model = load_model(config)
            lat = benchmark_latency(model, synth_input_ids, synth_enc_mask, synth_dec_ids, synth_dec_mask)
            print(f"    {label:<20}: {lat*1000:.2f} ms")
            del model
            gc.collect()
            torch.cuda.empty_cache()

    # Summary table
    print(f"\n\n{'=' * 80}")
    print("SUMMARY TABLE")
    print(f"{'=' * 80}")
    print(f"{'Config':<28} {'Flash':>5} {'Norm':>5} {'Embed':>5} | "
          f"{'Enc max':>10} {'Dec max':>10} {'Log max':>10} | {'Latency':>10}")
    print(f"{'':<28} {'':>5} {'':>5} {'':>5} | "
          f"{'diff':>10} {'diff':>10} {'diff':>10} | {'(ms)':>10}")
    print("─" * 110)

    baseline_ms = None
    for r in results:
        if r["label"].startswith("ALL OFF"):
            baseline_ms = r["latency_ms"]
        speedup = ""
        if baseline_ms and baseline_ms > 0:
            pct = (baseline_ms - r["latency_ms"]) / baseline_ms * 100
            speedup = f" ({pct:+.1f}%)"
        print(f"{r['label']:<28} "
              f"{'ON' if r['flash'] else 'off':>5} "
              f"{'ON' if r['norm_rope'] else 'off':>5} "
              f"{'ON' if r['embed'] else 'off':>5} | "
              f"{r['enc_max_diff']:>10.2e} "
              f"{r['dec_max_diff']:>10.2e} "
              f"{r['logit_max_diff']:>10.2e} | "
              f"{r['latency_ms']:>7.2f} ms{speedup}")

    # Marginal contribution
    print(f"\n{'=' * 80}")
    print("MARGINAL CONTRIBUTION (vs ALL ON baseline)")
    print(f"{'=' * 80}")
    all_on_ms = next(r["latency_ms"] for r in results if r["label"].startswith("ALL ON "))
    for r in results:
        if r["label"].startswith("ALL ON -"):
            kernel_name = r["label"].replace("ALL ON - ", "")
            delta = r["latency_ms"] - all_on_ms
            print(f"  Removing {kernel_name:<12}: {delta:+.2f} ms ({delta/all_on_ms*100:+.1f}%) "
                  f"{'<-- NO IMPACT' if abs(delta) < 0.5 else ''}")

    print()
    ctx.__exit__(None, None, None)


if __name__ == "__main__":
    main()
