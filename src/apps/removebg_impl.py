"""FastAPI app for removebg — imported inside the fal worker only."""
import threading

from src.utils import create_app, unpack_request, pack_response
from src.batch_payload import samples_to_stacked_inputs
from src.torch_batch_run import run_torch_model_batched
import os
import numpy as np
from fastapi import Request, Response

app = create_app("Heavy RemoveBG Microservice")

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

        print("Loading RemoveBG models into memory...")

        yolo_path = "/data/triton-repos/main/DetectorYoloV5/3/model.pt"
        if os.path.exists(yolo_path):
            models["DetectorYoloV5"] = torch.jit.load(yolo_path).cuda().eval()

        acconet_path = "/data/triton-repos/main/Acconet_segmentation/14/model.pt"
        if os.path.exists(acconet_path):
            models["Acconet_segmentation"] = torch.jit.load(acconet_path).cuda().eval()

        for ver in ["1", "2", "4"]:
            insp_path = f"/data/triton-repos/main/Inspyrenet_segmentation/{ver}/model.pt"
            if os.path.exists(insp_path):
                models[f"Inspyrenet_segmentation_v{ver}"] = (
                    torch.jit.load(insp_path).cuda().eval()
                )

        mvanet_path = "/data/triton-repos/main/Mvanet_segmentation/1/model.pt"
        if os.path.exists(mvanet_path):
            models["Mvanet_segmentation"] = torch.jit.load(mvanet_path).cuda().eval()

        pgnet_path = "/data/triton-repos/main/PGNet_segmentation/1/model.pt"
        if os.path.exists(pgnet_path):
            models["PGNet_segmentation"] = torch.jit.load(pgnet_path).cuda().eval()

        esrgan_path = "/data/triton-repos/main/real_esrgan/1/model.pt"
        if os.path.exists(esrgan_path):
            models["real_esrgan"] = torch.jit.load(esrgan_path).cuda().eval()

        print(f"Loaded models: {list(models.keys())}")


@app.post("/infer/{model_name}")
async def infer(model_name: str, request: Request):
    load_models()
    version = request.query_params.get("version", "1")
    target_model = (
        f"{model_name}_v{version}"
        if f"{model_name}_v{version}" in models
        else model_name
    )

    if target_model not in models:
        return Response(status_code=404, content=f"Model {target_model} not found.")

    payload = await unpack_request(request)
    samples = payload.get("samples")

    if (
        isinstance(samples, list)
        and len(samples) > 0
        and all(isinstance(s, dict) and "inputs" in s for s in samples)
    ):
        try:
            merged, B = samples_to_stacked_inputs(samples)
            body = run_torch_model_batched(models[target_model], merged, B)
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

        model = models[target_model]
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
