#!/usr/bin/env python3
"""T5Gemma2 parity test -- two-phase design.

Phase 1 (--collect): HF transformers reference outputs -> saved to disk
  Requires: transformers >= 5.0.0 (no vllm needed)

Phase 2 (--test): vLLM-factory model vs saved references
  Requires: vllm 0.19 (transformers 4.x is fine -- fallback configs used)

Usage:
    # In reference venv (transformers>=5.0):
    python models/t5gemma2/parity_test.py --collect

    # In main venv (vllm 0.19):
    python models/t5gemma2/parity_test.py --test

    # Both (only works when transformers>=5.0 AND vllm are available):
    python models/t5gemma2/parity_test.py
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import sys
from pathlib import Path

import torch

MODEL_NAME = "google/t5gemma-2-270m-270m"
REF_FILE = "/tmp/t5gemma2-reference.pt"

TEXT_TEST_CASES = [
    (
        "Translate English to French: Hello world.",
        "Bonjour le monde.",
    ),
    (
        "Summarize: The quick brown fox jumps over the lazy dog.",
        "A fox jumps over a dog.",
    ),
]
MM_TARGETS = ["A synthetic image description."]

ATOL = 0.02
RTOL = 1e-3

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_decoder_inputs(
    tokenizer,
    target_texts: list[str],
    config,
) -> tuple[torch.Tensor, torch.Tensor]:
    targets = tokenizer(
        target_texts,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    )
    bos_token_id = config.decoder.bos_token_id
    pad_token_id = config.decoder.pad_token_id

    decoder_input_ids = torch.full(
        (targets.input_ids.shape[0], targets.input_ids.shape[1] + 1),
        pad_token_id,
        dtype=torch.long,
    )
    decoder_input_ids[:, 0] = bos_token_id
    decoder_input_ids[:, 1:] = targets.input_ids
    decoder_attention_mask = decoder_input_ids.ne(pad_token_id)
    return decoder_input_ids, decoder_attention_mask


def _build_multimodal_inputs(
    tokenizer, config, device: torch.device
) -> dict[str, torch.Tensor]:
    prompt_ids = tokenizer(
        ["Describe the image in one sentence."],
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids.to(device)
    placeholder_ids = torch.full(
        (1, config.encoder.mm_tokens_per_image),
        config.encoder.image_token_index,
        dtype=torch.long,
        device=device,
    )
    input_ids = torch.cat([prompt_ids, placeholder_ids], dim=1)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    image_size = config.encoder.vision_config.image_size
    torch.manual_seed(42)
    pixel_values = torch.randn(
        1,
        3,
        image_size,
        image_size,
        device=device,
        dtype=torch.float32,
    )
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
    }


def compare_tensors(
    name: str,
    ref: torch.Tensor,
    actual: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    atol: float = ATOL,
    rtol: float = RTOL,
) -> bool:
    if ref.shape != actual.shape:
        print(
            f"  {name:<30} SHAPE MISMATCH ref={tuple(ref.shape)} "
            f"actual={tuple(actual.shape)} FAIL"
        )
        return False
    r, a = ref.float(), actual.float()
    if mask is not None:
        flat_mask = mask.view(*mask.shape, *([1] * (r.dim() - mask.dim())))
        flat_mask = flat_mask.expand_as(r)
        r = r[flat_mask]
        a = a[flat_mask]
    if r.numel() == 0:
        print(f"  {name:<30} (empty after masking) SKIP")
        return True
    abs_diff = (r - a).abs()
    max_diff = abs_diff.max().item()
    mean_diff = abs_diff.mean().item()
    ok = torch.allclose(r, a, atol=atol, rtol=rtol)
    suffix = " (valid only)" if mask is not None else ""
    print(
        f"  {name:<30} shape={tuple(ref.shape)}{suffix} "
        f"max_diff={max_diff:.6e} mean_diff={mean_diff:.6e} "
        f"status={'PASS' if ok else 'FAIL'}"
    )
    return ok


def _make_hook(store: dict, name: str):
    """Return a forward hook that saves the module's output tensor."""
    def hook_fn(module, input, output):
        if isinstance(output, torch.Tensor):
            store[name] = output.detach().cpu().float()
        elif isinstance(output, tuple) and isinstance(output[0], torch.Tensor):
            store[name] = output[0].detach().cpu().float()
        elif hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
            store[name] = output.last_hidden_state.detach().cpu().float()
        elif hasattr(output, "pooler_output") and output.pooler_output is not None:
            store[name] = output.pooler_output.detach().cpu().float()
    return hook_fn


# ---------------------------------------------------------------------------
# Phase 1: --collect (HF reference)
# ---------------------------------------------------------------------------

def phase_collect():
    """Generate HF T5Gemma2 reference outputs and save to disk."""

    from transformers import (
        AutoTokenizer,
        T5Gemma2ForConditionalGeneration as HFModel,
    )

    print("=" * 72)
    print("PHASE 1: Collect HF Reference Outputs")
    print("=" * 72)

    print(f"Loading HF model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    hf_model = HFModel.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    hf_model.float().eval()
    config = hf_model.config

    # HF creates embed_scale as a non-persistent buffer from Python float.
    # When the model is loaded in float32 (for test precision), the buffer
    # keeps the float32 value (25.298 for dim=640).  In real inference the
    # checkpoint dtype is bf16, so the effective value is 25.25.  Normalise
    # here so the reference matches bf16-rounded production behaviour.
    _bf16_scale = float(torch.tensor(config.decoder.hidden_size ** 0.5, dtype=torch.bfloat16))
    hf_model.model.decoder.embed_tokens.embed_scale.fill_(_bf16_scale)
    hf_model.model.encoder.text_model.embed_tokens.embed_scale.fill_(_bf16_scale)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hf_model = hf_model.to(device)

    ref_data: dict = {
        "model_name": MODEL_NAME,
        "config_json": json.loads(config.to_json_string()),
    }

    # -- Text-only --
    print("\n--- Text-only test case ---")
    source_texts = [s for s, _ in TEXT_TEST_CASES]
    target_texts = [t for _, t in TEXT_TEST_CASES]

    encoder_inputs = tokenizer(source_texts, return_tensors="pt", padding=True).to(device)
    decoder_input_ids, decoder_attention_mask = _build_decoder_inputs(
        tokenizer, target_texts, config
    )
    decoder_input_ids = decoder_input_ids.to(device)
    decoder_attention_mask = decoder_attention_mask.to(device)

    hf_decoder = hf_model.model.decoder
    num_dec_layers = config.decoder.num_hidden_layers
    dec_layer_outputs: dict[str, torch.Tensor] = {}
    dec_hooks = []
    dec_hooks.append(
        hf_decoder.embed_tokens.register_forward_hook(
            _make_hook(dec_layer_outputs, "dec_embed")
        )
    )
    for i in range(num_dec_layers):
        dec_hooks.append(
            hf_decoder.layers[i].register_forward_hook(
                _make_hook(dec_layer_outputs, f"dec_layer_{i}")
            )
        )
    dec_hooks.append(
        hf_decoder.norm.register_forward_hook(
            _make_hook(dec_layer_outputs, "dec_norm")
        )
    )

    with torch.no_grad():
        hf_enc_dec = hf_model.model(
            input_ids=encoder_inputs.input_ids,
            attention_mask=encoder_inputs.attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            return_dict=True,
        )
        hf_full = hf_model(
            input_ids=encoder_inputs.input_ids,
            attention_mask=encoder_inputs.attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            return_dict=True,
        )

    for h in dec_hooks:
        h.remove()

    ref_data["text"] = {
        "input_ids": encoder_inputs.input_ids.cpu(),
        "attention_mask": encoder_inputs.attention_mask.cpu(),
        "decoder_input_ids": decoder_input_ids.cpu(),
        "decoder_attention_mask": decoder_attention_mask.cpu(),
        "encoder_hidden": hf_enc_dec.encoder_last_hidden_state.cpu().float(),
        "decoder_hidden": hf_enc_dec.last_hidden_state.cpu().float(),
        "logits": hf_full.logits.cpu().float(),
    }
    ref_data["text_decoder_layers"] = dict(dec_layer_outputs)

    print(f"  encoder_hidden: {tuple(ref_data['text']['encoder_hidden'].shape)}")
    print(f"  decoder_hidden: {tuple(ref_data['text']['decoder_hidden'].shape)}")
    print(f"  logits:         {tuple(ref_data['text']['logits'].shape)}")
    print(f"  decoder layers saved: {sorted(dec_layer_outputs.keys())}")

    # -- Multimodal --
    print("\n--- Multimodal test case ---")
    mm_inputs = _build_multimodal_inputs(tokenizer, config, device)
    mm_dec_ids, mm_dec_mask = _build_decoder_inputs(tokenizer, MM_TARGETS, config)
    mm_dec_ids = mm_dec_ids.to(device)
    mm_dec_mask = mm_dec_mask.to(device)

    mm_vision_outputs: dict[str, torch.Tensor] = {}

    hf_encoder = hf_model.model.encoder
    vision_hooks = []
    if hasattr(hf_encoder, "vision_tower"):
        vision_hooks.append(
            hf_encoder.vision_tower.register_forward_hook(
                _make_hook(mm_vision_outputs, "vision_backbone")
            )
        )
    if hasattr(hf_encoder, "multi_modal_projector"):
        vision_hooks.append(
            hf_encoder.multi_modal_projector.register_forward_hook(
                _make_hook(mm_vision_outputs, "mm_projector")
            )
        )

    with torch.no_grad():
        hf_mm_enc_dec = hf_model.model(
            input_ids=mm_inputs["input_ids"],
            attention_mask=mm_inputs["attention_mask"],
            pixel_values=mm_inputs["pixel_values"],
            decoder_input_ids=mm_dec_ids,
            decoder_attention_mask=mm_dec_mask,
            return_dict=True,
        )
        hf_mm_full = hf_model(
            input_ids=mm_inputs["input_ids"],
            attention_mask=mm_inputs["attention_mask"],
            pixel_values=mm_inputs["pixel_values"],
            decoder_input_ids=mm_dec_ids,
            decoder_attention_mask=mm_dec_mask,
            return_dict=True,
        )

    for h in vision_hooks:
        h.remove()

    ref_data["multimodal"] = {
        "input_ids": mm_inputs["input_ids"].cpu(),
        "attention_mask": mm_inputs["attention_mask"].cpu(),
        "pixel_values": mm_inputs["pixel_values"].cpu(),
        "decoder_input_ids": mm_dec_ids.cpu(),
        "decoder_attention_mask": mm_dec_mask.cpu(),
        "encoder_hidden": hf_mm_enc_dec.encoder_last_hidden_state.cpu().float(),
        "decoder_hidden": hf_mm_enc_dec.last_hidden_state.cpu().float(),
        "logits": hf_mm_full.logits.cpu().float(),
    }
    ref_data["multimodal_vision"] = dict(mm_vision_outputs)

    print(f"  encoder_hidden: {tuple(ref_data['multimodal']['encoder_hidden'].shape)}")
    print(f"  decoder_hidden: {tuple(ref_data['multimodal']['decoder_hidden'].shape)}")
    print(f"  logits:         {tuple(ref_data['multimodal']['logits'].shape)}")
    if mm_vision_outputs:
        for k, v in mm_vision_outputs.items():
            print(f"  {k}: {tuple(v.shape)}")

    torch.save(ref_data, REF_FILE)
    print(f"\nSaved reference to {REF_FILE}")
    print("Phase 1 complete.\n")


# ---------------------------------------------------------------------------
# Phase 2: --test (vLLM-factory comparison)
# ---------------------------------------------------------------------------

def _init_vllm_env():
    """Initialize minimal vLLM distributed context for standalone tests."""
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
    init_distributed_environment(
        world_size=1, rank=0, local_rank=0, distributed_init_method="env://"
    )
    ensure_model_parallel_initialized(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1
    )
    return ctx


def _load_config_from_ref(ref_data: dict):
    """Reconstruct a T5Gemma2Config from the saved config JSON."""
    sys.path.insert(
        0,
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    )
    from models.t5gemma2.config import T5Gemma2Config

    config_dict = dict(ref_data["config_json"])
    _KNOWN_INIT_ARGS = {
        "encoder", "decoder", "is_encoder_decoder", "dropout_rate",
        "attention_dropout", "classifier_dropout_rate", "initializer_range",
        "image_token_index", "eoi_token_index", "tie_word_embeddings",
    }
    filtered = {k: v for k, v in config_dict.items() if k in _KNOWN_INIT_ARGS}
    return T5Gemma2Config(**filtered)


def _load_weights_from_hub(model_name: str):
    """Download safetensors from HF Hub and yield (name, tensor) pairs."""
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    model_path = Path(snapshot_download(model_name))
    safetensor_files = sorted(model_path.glob("*.safetensors"))
    if not safetensor_files:
        raise FileNotFoundError(
            f"No safetensors files found in {model_path}. "
            "Check that the model is downloaded correctly."
        )
    for sf in safetensor_files:
        state = load_file(str(sf))
        yield from state.items()


def _set_reference_path(enabled: bool) -> None:
    if enabled:
        os.environ["VLLM_FACTORY_T5GEMMA2_REFERENCE_PATH"] = "1"
    else:
        os.environ.pop("VLLM_FACTORY_T5GEMMA2_REFERENCE_PATH", None)


def _run_single_parity(
    our_model,
    ref_block: dict,
    *,
    label: str,
    device: torch.device,
    has_pixel_values: bool = False,
    ref_decoder_layers: dict[str, torch.Tensor] | None = None,
    ref_vision: dict[str, torch.Tensor] | None = None,
    enc_atol: float = ATOL,
    dec_atol: float = ATOL,
    logit_atol: float = ATOL,
) -> bool:
    input_ids = ref_block["input_ids"].to(device)
    attention_mask = ref_block["attention_mask"].to(device)
    decoder_input_ids = ref_block["decoder_input_ids"].to(device)
    decoder_attention_mask = ref_block["decoder_attention_mask"].to(device)
    pixel_values = ref_block["pixel_values"].to(device) if has_pixel_values else None

    our_dec_layers: dict[str, torch.Tensor] = {}
    our_vision: dict[str, torch.Tensor] = {}
    hooks = []

    decoder = our_model.model.decoder
    if ref_decoder_layers:
        hooks.append(
            decoder.embed_tokens.register_forward_hook(
                _make_hook(our_dec_layers, "dec_embed")
            )
        )
        num_layers = len(decoder.layers)
        for i in range(num_layers):
            hooks.append(
                decoder.layers[i].register_forward_hook(
                    _make_hook(our_dec_layers, f"dec_layer_{i}")
                )
            )
        hooks.append(
            decoder.norm.register_forward_hook(
                _make_hook(our_dec_layers, "dec_norm")
            )
        )

    encoder = our_model.model.encoder
    if ref_vision and has_pixel_values:
        if encoder.vision_tower is not None:
            hooks.append(
                encoder.vision_tower.register_forward_hook(
                    _make_hook(our_vision, "vision_backbone")
                )
            )
        if encoder.multi_modal_projector is not None:
            hooks.append(
                encoder.multi_modal_projector.register_forward_hook(
                    _make_hook(our_vision, "mm_projector")
                )
            )

    with torch.no_grad():
        our_encoder_hidden = our_model.get_encoder_outputs(
            input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
        )
        our_decoder_hidden = our_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            pixel_values=pixel_values,
        )
        our_logits = our_model.compute_logits(our_decoder_hidden)

    for h in hooks:
        h.remove()

    ref_enc = ref_block["encoder_hidden"].to(device)
    ref_dec = ref_block["decoder_hidden"].to(device)
    ref_logits = ref_block["logits"].to(device)

    enc_mask = attention_mask.bool()
    dec_mask = decoder_attention_mask.bool()

    print(f"\n  {label}")
    all_ok = True
    all_ok &= compare_tensors(
        "encoder_hidden", ref_enc, our_encoder_hidden, mask=enc_mask, atol=enc_atol
    )
    all_ok &= compare_tensors(
        "decoder_hidden", ref_dec, our_decoder_hidden, mask=dec_mask, atol=dec_atol
    )
    all_ok &= compare_tensors(
        "logits", ref_logits, our_logits, mask=dec_mask, atol=logit_atol
    )

    if ref_decoder_layers and our_dec_layers:
        print(f"\n    -- Decoder per-layer breakdown ({label}) --")
        for key in sorted(ref_decoder_layers.keys()):
            if key in our_dec_layers:
                ref_t = ref_decoder_layers[key].to(device)
                our_t = our_dec_layers[key].to(device)
                layer_mask = dec_mask if "embed" not in key else None
                compare_tensors(
                    f"  {key}", ref_t, our_t, mask=layer_mask, atol=dec_atol
                )

    if ref_vision and our_vision and has_pixel_values:
        print(f"\n    -- Vision pipeline breakdown ({label}) --")
        for key in sorted(ref_vision.keys()):
            if key in our_vision:
                ref_t = ref_vision[key].to(device)
                our_t = our_vision[key].to(device)
                compare_tensors(f"  {key}", ref_t, our_t, atol=enc_atol)

    return all_ok


def phase_test():
    """Load saved references and compare with vLLM-factory model."""

    print("=" * 72)
    print("PHASE 2: vLLM-factory Parity Test")
    print("=" * 72)

    ref_data = torch.load(REF_FILE, map_location="cpu", weights_only=False)
    print(f"Loaded reference from {REF_FILE}")
    print(f"  Model: {ref_data['model_name']}")

    has_decoder_layers = "text_decoder_layers" in ref_data
    has_vision_ref = "multimodal_vision" in ref_data
    print(f"  Decoder per-layer refs: {'yes' if has_decoder_layers else 'no'}")
    print(f"  Vision pipeline refs:   {'yes' if has_vision_ref else 'no'}")

    ctx = _init_vllm_env()

    sys.path.insert(
        0,
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    )
    from models.t5gemma2.t5gemma2_model import T5Gemma2ForConditionalGeneration

    config = _load_config_from_ref(ref_data)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_ok = True
    for reference_path in (True, False):
        _set_reference_path(reference_path)
        path_label = "REFERENCE" if reference_path else "OPTIMIZED"

        print(f"\n{'=' * 72}")
        print(f"  Code path: {path_label}")
        print(f"{'=' * 72}")

        our_model = T5Gemma2ForConditionalGeneration(config)
        our_model.eval()
        print(f"Loading weights from {ref_data['model_name']}...")
        our_model.load_weights(_load_weights_from_hub(ref_data["model_name"]))
        our_model = our_model.to(device).float()

        if reference_path:
            text_enc_atol = 0.001
            text_dec_atol = 0.001
            text_logit_atol = 0.001
            mm_enc_atol = 0.02
            mm_dec_atol = 0.001
            mm_logit_atol = 0.001
        else:
            text_enc_atol = 0.001
            text_dec_atol = 0.001
            text_logit_atol = 0.001
            mm_enc_atol = 20.0
            mm_dec_atol = 0.05
            mm_logit_atol = 0.05

        all_ok &= _run_single_parity(
            our_model,
            ref_data["text"],
            label=f"Text-only ({path_label})",
            device=device,
            ref_decoder_layers=ref_data.get("text_decoder_layers"),
            enc_atol=text_enc_atol,
            dec_atol=text_dec_atol,
            logit_atol=text_logit_atol,
        )
        all_ok &= _run_single_parity(
            our_model,
            ref_data["multimodal"],
            label=f"Multimodal ({path_label})",
            device=device,
            has_pixel_values=True,
            ref_vision=ref_data.get("multimodal_vision"),
            enc_atol=mm_enc_atol,
            dec_atol=mm_dec_atol,
            logit_atol=mm_logit_atol,
        )

        del our_model
        torch.cuda.empty_cache()

    _set_reference_path(False)
    print(f"\n{'=' * 72}")
    if all_ok:
        print("ALL PARITY CHECKS PASSED")
    else:
        print("SOME PARITY CHECKS FAILED")
    print(f"{'=' * 72}")

    ctx.__exit__(None, None, None)
    return 0 if all_ok else 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="T5Gemma2 Parity Test")
    parser.add_argument(
        "--collect",
        action="store_true",
        help="Phase 1: collect HF reference outputs",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Phase 2: compare vLLM-factory model against saved references",
    )
    parser.add_argument(
        "--ref-file",
        default=REF_FILE,
        help=f"Path for reference file (default: {REF_FILE})",
    )
    args = parser.parse_args()
    REF_FILE = args.ref_file

    if args.collect:
        phase_collect()
    elif args.test:
        sys.exit(phase_test())
    else:
        print("Running both phases sequentially...\n")
        phase_collect()
        sys.exit(phase_test())
