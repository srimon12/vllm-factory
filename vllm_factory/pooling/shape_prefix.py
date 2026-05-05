"""Utilities for shape-prefixed pooling payloads."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def pack_shape_prefixed_tensor(shape: Sequence[int], *payloads: torch.Tensor) -> torch.Tensor:
    """Pack integer shape metadata without bf16/fp16 rounding.

    Several GLiNER poolers flatten logits as ``[shape..., payload...]``. The
    shape prefix must remain exact for word counts above 256, where bf16 cannot
    represent every integer.
    """

    device = payloads[0].device if payloads else None
    prefix = torch.tensor([int(item) for item in shape], device=device, dtype=torch.float32)
    flattened = [payload.reshape(-1).to(dtype=torch.float32) for payload in payloads]
    return torch.cat([prefix, *flattened]) if flattened else prefix
