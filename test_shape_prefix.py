from __future__ import annotations

import torch

from vllm_factory.pooling.shape_prefix import pack_shape_prefixed_tensor


def test_shape_prefix_stays_exact_for_bf16_odd_word_counts_above_256() -> None:
    scores = torch.zeros((285, 3, 3), dtype=torch.bfloat16)

    packed = pack_shape_prefixed_tensor(scores.shape, scores)

    assert packed.dtype == torch.float32
    assert [int(item.item()) for item in packed[:3]] == [285, 3, 3]
    logits = packed[3:].reshape(1, 285, 3, 3)
    assert logits.shape == (1, 285, 3, 3)


def test_bf16_shape_prefix_would_round_without_float32_packing() -> None:
    rounded_prefix = torch.tensor([285, 3, 3], dtype=torch.bfloat16)

    assert int(rounded_prefix[0].item()) == 284
