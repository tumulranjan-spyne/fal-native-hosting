"""FastAPI app for colmap/warping — imported inside the fal worker only."""
import threading

from src.utils import create_app, unpack_request, pack_response
from src.batch_payload import samples_to_stacked_inputs
from src.torch_batch_run import run_torch_model_batched
import os
import numpy as np
from fastapi import Request, Response

app = create_app("Heavy Colmap/Warping Microservice")

models = {}
_models_lock = threading.Lock()


def load_models():
    global models
    if models:
        return
    with _models_lock:
        if models:
            return
        import torch

        print("Loading Colmap/Warping models into memory...")

        banner_path = "/data/triton-repos/main/banner_detection/1/model.pt"
        if os.path.exists(banner_path):
            models["banner_detection"] = torch.jit.load(banner_path).cuda().eval()

        flare_path = "/data/triton-repos/heavy/bg_independent_flare/1/model.pt"
        if os.path.exists(flare_path):
            models["bg_independent_flare"] = torch.jit.load(flare_path).cuda().eval()

        print(f"Loaded models: {list(models.keys())}")


@app.post("/infer/{model_name}")
async def infer(model_name: str, request: Request):
    load_models()
    if model_name not in models:
        return Response(
            status_code=404,
            content=(
                f"Model {model_name} not found. "
                "(Note: vanilla colmap requires custom backend integration)"
            ),
        )

    payload = await unpack_request(request)
    samples = payload.get("samples")

    if (
        isinstance(samples, list)
        and len(samples) > 0
        and all(isinstance(s, dict) and "inputs" in s for s in samples)
    ):
        try:
            merged, B = samples_to_stacked_inputs(samples)
            body = run_torch_model_batched(models[model_name], merged, B)
            return pack_response(body)
        except ValueError as ve:
            return Response(status_code=400, content=str(ve))
        except Exception as e:
            import traceback

            traceback.print_exc()
            return Response(status_code=500, content=str(e))

    inputs = payload.get("inputs", [])

    try:
        import torch

        model = models[model_name]
        with torch.no_grad():
            tensor_inputs = []
            for inp in inputs:
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
        return pack_response({"outputs": outputs_np})

    except Exception as e:
        import traceback

        traceback.print_exc()
        return Response(status_code=500, content=str(e))
