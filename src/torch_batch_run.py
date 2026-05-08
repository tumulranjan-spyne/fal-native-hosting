"""Shared TorchScript forward for batched numpy inputs (stacked on axis 0)."""
from __future__ import annotations

import numpy as np


def run_torch_model_batched(model, merged_inputs: list[np.ndarray], batch_size: int) -> dict:
    import torch

    with torch.no_grad():
        tensor_inputs = []
        for inp in merged_inputs:
            if inp.dtype == np.float64:
                inp = inp.astype(np.float32)
            tensor_inputs.append(torch.from_numpy(inp).cuda())
        outputs = model(*tensor_inputs)
        if isinstance(outputs, tuple) or isinstance(outputs, list):
            outputs_np = [out.cpu().numpy() for out in outputs]
        elif isinstance(outputs, dict):
            outputs_np = {k: v.cpu().numpy() for k, v in outputs.items()}
        else:
            outputs_np = [outputs.cpu().numpy()]
    return {"outputs": outputs_np, "batch_size": batch_size}
