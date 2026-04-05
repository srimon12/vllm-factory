# T5Gemma2

Encoder-decoder multimodal model (vision + text) based on Gemma-2 architecture, implemented as a vLLM Factory plugin with full HuggingFace parity.

## Architecture

- **Encoder**: T5-style text encoder with optional SigLIP vision tower and multimodal projector
- **Decoder**: Gemma-2 decoder with cross-attention to encoder outputs
- **Vision**: SiglipVisionModel backbone + average-pool projector for image tokens

Supports text-only and multimodal (image + text) inputs.

## Vision Backbone: Fast vs Reference Path

The vision backbone has **two code paths** controlled by a single environment variable:

### Fast path (default)

```bash
# No env var needed -- this is the default
python your_script.py
```

Uses vLLM's fused Triton attention kernels (`MMEncoderAttention`) and `QKVParallelLinear` for maximum throughput. This is the **production default** and should be used for all latency/throughput-sensitive workloads.

The fused kernels are mathematically correct but produce slightly different floating-point results compared to HuggingFace's PyTorch SDPA implementation. After 18 encoder layers, these differences can amplify (observed max_diff ~16 on encoder hidden states, though cosine similarity remains >0.9998).

### Reference path

```bash
export VLLM_FACTORY_T5GEMMA2_REFERENCE_PATH=1
python your_script.py
```

Replaces the fused Triton attention in the SigLIP vision backbone with standard `F.scaled_dot_product_attention`. Achieves near-exact numerical parity with HuggingFace (max_diff < 0.01 on multimodal encoder hidden states). Slightly slower than the fast path.

Use the reference path when:

- You need bit-exact reproducibility against HuggingFace Transformers
- Running parity/regression tests
- A downstream task shows quality degradation with the fast path (see below)

### Which path should I use?

**Start with the fast path** (default). For most tasks -- captioning, generation, image classification with mean pooling -- the fused kernel differences are negligible and do not affect output quality.

For **per-token tasks** that depend on individual encoder hidden states (ColPALI multi-vector retrieval, dense OCR, token-level classification), the amplified differences at specific image token positions could theoretically affect results. **Test your specific task** by running evaluation with both paths:

```bash
# Run with fast path
python evaluate.py --task colpali ...

# Run with reference path
VLLM_FACTORY_T5GEMMA2_REFERENCE_PATH=1 python evaluate.py --task colpali ...
```

If your task metric (retrieval recall, accuracy, CER, etc.) is the same or within noise, the fast path is safe. If you see degradation, use the reference path for that workload.

### Summary

| | Fast path (default) | Reference path |
|---|---|---|
| **Env var** | *(none)* | `VLLM_FACTORY_T5GEMMA2_REFERENCE_PATH=1` |
| **Vision attention** | Fused Triton kernels | PyTorch SDPA |
| **Speed** | Fastest | Slightly slower |
| **HF parity (encoder)** | ~16 max_diff | <0.01 max_diff |
| **HF parity (decoder/logits)** | <0.001 | <0.001 |
| **Use for** | Production inference | Parity testing, sensitive per-token tasks |

## Parity Testing

The included `parity_test.py` verifies numerical parity against HuggingFace Transformers across both text-only and multimodal inputs.

### Running parity tests

```bash
# Phase 1: Collect HF reference outputs (requires transformers >= 5.0)
pip install 'transformers>=5.0.0'
python models/t5gemma2/parity_test.py --collect

# Phase 2: Test vLLM-factory model against references (requires vllm)
pip install vllm==0.19.0
python models/t5gemma2/parity_test.py --test
```

The test runs both code paths (reference and optimized) and reports per-component max/mean differences with pass/fail status. It also provides a per-layer decoder breakdown for debugging.

### Current parity results (reference path)

| Component | max_diff | Status |
|---|---|---|
| Text encoder hidden | 0.000246 | PASS |
| Text decoder hidden | 0.000092 | PASS |
| Text logits | 0.000151 | PASS |
| MM encoder hidden | 0.009 | PASS |
| MM decoder hidden | 0.000062 | PASS |
| Vision backbone | 0.000963 | PASS |
| MM projector | 0.000704 | PASS |

## Performance Kernels

Two custom Triton kernels are enabled by default, benchmarked to contribute meaningful end-to-end speedup. A third was evaluated and disabled (zero impact).

| Kernel | Speedup | Toggle (env var) | Default |
|---|---|---|---|
| `flash_t5gemma2_attention` | **+25%** (17ms saved) | `T5GEMMA2_NO_FLASH=1` to disable | ON |
| `fused_qk_norm_rope` | **+10%** (7ms saved) | `T5GEMMA2_NO_FUSED_NORM_ROPE=1` to disable | ON |
| `fused_embed_scale_eoi` | 0% (no impact) | `T5GEMMA2_FUSED_EMBED=1` to enable | OFF |

Benchmarked with CUDA event timing, 10 warmup + 50 measured iterations, Triton JIT excluded.

### Throughput at varying batch sizes

Measured on `google/t5gemma-2-270m-270m`, encoder seq_len=14, decoder seq_len=8, float32:

| BS | Optimized (samp/s) | Baseline (samp/s) | Speedup |
|---:|---:|---:|---:|
| 1 | 15.3 | 10.4 | 1.48x |
| 8 | 117.3 | 77.9 | 1.51x |
| 16 | 236.2 | 161.2 | 1.47x |
| 32 | 459.3 | 321.1 | 1.43x |
| 64 | 854.1 | 592.0 | 1.44x |

Peak: **854 samples/s** at bs=64 (optimized) vs 592 samples/s (baseline).

## Supported Model

- `google/t5gemma-2-270m-270m` (gated -- requires HuggingFace authentication)

## Files

| File | Description |
|---|---|
| `config.py` | Configuration classes with HF fallbacks |
| `t5gemma2_encoder.py` | Encoder backbone, SigLIP integration, reference-path SigLIP attention |
| `t5gemma2_model.py` | Decoder, full encoder-decoder model, weight loading |
| `parity_test.py` | Two-phase parity test against HuggingFace |
| `kernel_benchmark.py` | Per-kernel parity + latency benchmark |
| `throughput_benchmark.py` | Batch throughput sweep (bs=1..64) |
