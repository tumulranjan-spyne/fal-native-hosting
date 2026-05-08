"""FastAPI app for postprocessing — imported inside the fal worker only."""
import threading

from src.utils import create_app, unpack_request, pack_response
import os
import numpy as np
from fastapi import Request, Response

app = create_app("Fast Post-processing Microservice")

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

        print("Loading Post-processing models into memory...")

        car_type_path = "/data/triton-repos/main/car_type_classifier/1/model.pt"
        if os.path.exists(car_type_path):
            models["car_type_classifier"] = torch.jit.load(car_type_path).cuda().eval()

        plate_path = "/data/triton-repos/main/Mvanet_plate_segmentation/1/model.pt"
        if os.path.exists(plate_path):
            models["Mvanet_plate_segmentation"] = torch.jit.load(plate_path).cuda().eval()

        print(f"Loaded models: {list(models.keys())}")


@app.post("/infer/{model_name}")
async def infer(model_name: str, request: Request):
    load_models()
    if model_name not in models:
        return Response(status_code=404, content=f"Model {model_name} not found.")

    payload = await unpack_request(request)
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
