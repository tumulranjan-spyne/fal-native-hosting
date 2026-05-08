"""FastAPI app for preprocessing — imported inside the fal worker only (see preprocessing.py)."""
import threading

from src.utils import create_app, unpack_request, pack_response
import os
import numpy as np
from fastapi import Request, Response

app = create_app("Fast Pre-processing Microservice")

models = {}
_models_lock = threading.Lock()

# Triton model names from the pipeline -> internal keys
_MODEL_ALIASES = {
    "auto-classification-angle_classification": "angle",
    "auto-classification-roi": "roi",
}


def _resolve_model(model_name: str) -> str:
    return _MODEL_ALIASES.get(model_name, model_name)


def load_models():
    """Load ONNX and PyTorch models into memory (lazy; does not block /health)."""
    global models
    if models:
        return
    with _models_lock:
        if models:
            return
        # Defer onnxruntime/torch until first infer — top-level import blocked fal SETUP / health.
        print("Loading models into memory...")

        angle_path = "/data/triton-repos/main/auto-classification-angle_classification/1/model.onnx"
        if os.path.exists(angle_path):
            import onnxruntime as ort

            models["angle"] = ort.InferenceSession(
                angle_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            )

        roi_path = "/data/triton-repos/main/auto-classification-roi/1/model.onnx"
        if os.path.exists(roi_path):
            import onnxruntime as ort

            models["roi"] = ort.InferenceSession(
                roi_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            )

        netvlad_path = "/data/triton-repos/main/netvlad/1/model.pt"
        if os.path.exists(netvlad_path):
            import torch

            models["netvlad"] = torch.jit.load(netvlad_path).cuda()
            models["netvlad"].eval()

        print(f"Loaded models: {list(models.keys())}")


@app.post("/infer/{model_name}")
async def infer(model_name: str, request: Request):
    load_models()
    key = _resolve_model(model_name)
    if key not in models:
        return Response(
            status_code=404,
            content=f"Model {model_name} not found or failed to load.",
        )

    payload = await unpack_request(request)
    inputs = payload.get("inputs", [])

    try:
        if key in ["angle", "roi"]:
            session = models[key]
            input_name = session.get_inputs()[0].name
            input_data = inputs[0]
            if input_data.dtype != np.float32:
                input_data = input_data.astype(np.float32)
            outputs = session.run(None, {input_name: input_data})
            return pack_response({"outputs": outputs})

        elif key == "netvlad":
            import torch

            model = models[key]
            input_data = inputs[0]
            with torch.no_grad():
                tensor_input = torch.from_numpy(input_data).cuda()
                outputs = model(tensor_input)
                if isinstance(outputs, tuple) or isinstance(outputs, list):
                    outputs = [out.cpu().numpy() for out in outputs]
                else:
                    outputs = [outputs.cpu().numpy()]
            return pack_response({"outputs": outputs})

    except Exception as e:
        import traceback

        traceback.print_exc()
        return Response(status_code=500, content=str(e))
