# Benchmark Methodology

This document describes how vLLM Factory benchmarks are run so that published throughput claims can be independently reproduced.

## Hardware

All published benchmarks were run on:

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA RTX A5000 (24 GB VRAM) |
| Python | 3.11 |
| vLLM | 0.15.1 |
| dtype | bfloat16 |
| OS | Ubuntu 22.04, CUDA 12.x |

## Request parameters

| Parameter | Value |
|-----------|-------|
| Requests per sweep point | 500 |
| Sequence length | 512–768 tokens (varies by model family) |
| Concurrency levels | 1, 4, 8, 16, 32, 64 |
| Modes | saturate, staggered |

Each model is benchmarked at every combination of concurrency level and mode, producing 12 sweep points per model.

## Modes

**Saturate mode** fires all concurrent requests as fast as possible. This measures peak throughput — the maximum requests/second the system can sustain. It answers: "How fast can the GPU go when fully loaded?"

**Staggered mode** uses a Poisson arrival process at 85% of the saturate throughput. This simulates realistic production traffic where requests arrive at irregular intervals. It answers: "What latency will users actually experience?"

## Metrics

### Throughput factor

```
throughput_factor = vllm_req_per_s / vanilla_req_per_s
```

Measured at the same concurrency level. A factor of 8.6x means vLLM Factory serves 8.6 times more requests per second than the vanilla PyTorch baseline (e.g. SentenceTransformers, PyLate, GLiNER library) on the same GPU.

The **peak throughput factor** reported in the README is the maximum across all concurrency levels in saturate mode.

### Latency percentiles

- **p50** — median request latency in milliseconds
- **p95** — 95th percentile (tail latency under normal conditions)
- **p99** — 99th percentile (worst-case tail latency)

Measured end-to-end from HTTP request sent to response received, including network round-trip on localhost.

### Latency factor

```
latency_factor = vanilla_p50_ms / vllm_p50_ms
```

Values above 1.0 mean vLLM Factory has lower latency than the vanilla baseline at that concurrency level.

## Vanilla baselines

Each model family has a dedicated vanilla runner that uses the standard Python library for that architecture:

| Model family | Vanilla library | Runner |
|-------------|-----------------|--------|
| ColBERT / retrieval | PyLate, HF Transformers | `bench/vanilla_runners.py` |
| GLiNER NER | GLiNER library | `bench/vanilla_runners.py` |
| GLiNER2 schema extraction | GLiNER2 library | `bench/vanilla_runners.py` |
| Embeddings | SentenceTransformers | `bench/vanilla_runners.py` |
| ColPali / multimodal | sauerkrautlm-colpali | `bench/vanilla_runners.py` |
| Entity linking | GLinker library | `bench/vanilla_runners.py` |

Vanilla baselines run single-GPU, sequential inference — the standard deployment pattern for these libraries. No manual batching, no custom schedulers.

## Parity validation

Every benchmark run includes a parity check to confirm that vLLM Factory produces equivalent outputs to the reference implementation.

**Embedding and retrieval models**: element-wise cosine similarity between vLLM Factory output tensors and vanilla library output tensors. Passing threshold: cosine similarity >= 0.95.

**NER models (GLiNER, GLiNER2)**: entity recall — every entity (text span + label) found by the reference implementation must also be found by vLLM Factory. Extra entities found by vLLM Factory are acceptable. Score differences due to bfloat16 rounding are reported but not gating.

**Entity linking models**: recall on linked entities plus link target match.

Parity scores are recorded in each result JSON and summarized in the README.

## Result format

Each benchmark run produces a JSON file in `bench/results/` with the naming convention:

```
{plugin}_{model_id_slug}_{gpu_slug}_{timestamp}.json
```

Key fields:

```json
{
  "plugin": "mt5_gliner",
  "model_id": "knowledgator/gliner-x-large",
  "gpu": "NVIDIA RTX A5000",
  "seq_len": 768,
  "num_requests": 500,
  "concurrency_levels": [1, 4, 8, 16, 32, 64],
  "modes": ["saturate", "staggered"],
  "sweeps": [
    {
      "mode": "saturate",
      "concurrency": 64,
      "vllm_req_per_s": 183.0,
      "vllm_p50_ms": 344.2,
      "vanilla_req_per_s": 29.3,
      "vanilla_p50_ms": 33.4,
      "throughput_factor": 6.25
    }
  ],
  "parity_metric": "entity_recall",
  "parity_score": 1.0,
  "dtype": "bfloat16"
}
```

## Reproduction

### Full suite

```bash
python -m bench run --all
python -m bench chart --results bench/results/ --output bench/charts/
```

### Single model

```bash
python -m bench run --plugin mt5_gliner --seq-len 768 --num-requests 500 --concurrency 32
```

### Environment capture

Before running benchmarks, capture the full environment for reproducibility:

```bash
python benchmarks/environment_capture.py --output benchmarks/results/environment.json
```

This records Python version, platform, PyTorch version, vLLM version, CUDA version, and visible GPU metadata.

## Publication checklist

When citing benchmark results:

- [ ] Include the GPU model and VRAM
- [ ] Include the exact vLLM version
- [ ] State the concurrency level and mode for the claimed throughput factor
- [ ] Link the commit SHA or release tag
- [ ] Link the result JSON artifact
- [ ] State whether the pooling patch was applied (`python -m forge.patches.pooling_extra_kwargs`)
