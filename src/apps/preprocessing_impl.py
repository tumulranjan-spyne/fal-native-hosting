"""FastAPI app for preprocessing — imported inside the fal worker only (see preprocessing.py)."""
import threading

from src.utils import create_app, unpack_request, pack_response
from src.batch_payload import samples_to_stacked_inputs
import os
import numpy as np
from fastapi import Request, Response

app = create_app("Fast Pre-processing Microservice")

models = {}
# First-input numpy dtype per ONNX model; resolved at load from the ONNX file when possible.
model_input_dtypes = {}
_models_lock = threading.Lock()

# Triton model names from the pipeline -> internal keys
_MODEL_ALIASES = {
    "auto-classification-angle_classification": "angle",
    "auto-classification-roi": "roi",
}


def _resolve_model(model_name: str) -> str:
    return _MODEL_ALIASES.get(model_name, model_name)


def _value_info_inputs(model):
    """Graph inputs that are not weights (initializers); order matches typical feed order."""
    ini = {x.name for x in model.graph.initializer}
    return [x for x in model.graph.input if x.name not in ini]


def _input_np_dtype_from_onnx_path(model_path: str, preferred_input_name: str | None = None):
    """Resolve first fed input dtype from ONNX (not initializer rows in graph.input)."""
    try:
        import onnx
        from onnx import TensorProto

        model = onnx.load(model_path)
        value_inputs = _value_info_inputs(model)
        if not value_inputs:
            return None
        candidates = value_inputs
        if preferred_input_name:
            by_name = [x for x in value_inputs if x.name == preferred_input_name]
            if by_name:
                candidates = by_name
        elem_type = candidates[0].type.tensor_type.elem_type
        if not elem_type:
            return None
        mapping = {
            TensorProto.FLOAT: np.float32,
            TensorProto.UINT8: np.uint8,
            TensorProto.INT8: np.int8,
            TensorProto.UINT16: np.uint16,
            TensorProto.INT16: np.int16,
            TensorProto.INT32: np.int32,
            TensorProto.INT64: np.int64,
            TensorProto.DOUBLE: np.float64,
            TensorProto.FLOAT16: np.float16,
        }
        return mapping.get(elem_type)
    except Exception:
        return None


def _numpy_dtype_for_onnx_type(type_str: str):
    """Map onnxruntime NodeArg.type (e.g. tensor(uint8)) to numpy dtype."""
    s = str(type_str or "").lower()
    if "uint8" in s:
        return np.uint8
    if "uint16" in s:
        return np.uint16
    if "int8" in s:
        return np.int8
    if "int32" in s:
        return np.int32
    if "int64" in s:
        return np.int64
    if "float16" in s or "fp16" in s:
        return np.float16
    if "float64" in s or "double" in s:
        return np.float64
    if "float" in s:
        return np.float32
    return np.float32


def _coerce_input_to_dtype(arr: np.ndarray, target: np.dtype) -> np.ndarray:
    if arr.dtype == target:
        return arr
    if target == np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            return np.clip(np.rint(arr), 0, 255).astype(np.uint8)
        return np.clip(arr, 0, 255).astype(np.uint8)
    return arr.astype(target)


def load_models():
    """Load ONNX and PyTorch models into memory (lazy; does not block /health)."""
    global models, model_input_dtypes
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
            angle_in0 = models["angle"].get_inputs()[0].name
            dt = _input_np_dtype_from_onnx_path(
                angle_path, preferred_input_name=angle_in0
            )
            if dt is not None:
                model_input_dtypes["angle"] = dt

        roi_path = "/data/triton-repos/main/auto-classification-roi/1/model.onnx"
        if os.path.exists(roi_path):
            import onnxruntime as ort

            models["roi"] = ort.InferenceSession(
                roi_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            )
            roi_in0 = models["roi"].get_inputs()[0].name
            dt = _input_np_dtype_from_onnx_path(roi_path, preferred_input_name=roi_in0)
            if dt is not None:
                model_input_dtypes["roi"] = dt

        netvlad_path = "/data/triton-repos/main/netvlad/1/model.pt"
        if os.path.exists(netvlad_path):
            import torch

            models["netvlad"] = torch.jit.load(netvlad_path).cuda()
            models["netvlad"].eval()

        print(f"Loaded models: {list(models.keys())}")


def _infer_preprocessing_batched(key: str, merged_inputs: list, B: int):
    """Run one forward pass on batch-stacked inputs; return msgpack-ready dict."""
    if key in ["angle", "roi"]:
        session = models[key]
        inp0 = session.get_inputs()[0]
        input_name = inp0.name
        input_data = merged_inputs[0]
        target_dt = model_input_dtypes.get(key) or _numpy_dtype_for_onnx_type(
            inp0.type
        )
        if key in ("angle", "roi") and target_dt == np.float32:
            target_dt = np.uint8
        input_data = _coerce_input_to_dtype(input_data, target_dt)
        input_data = np.ascontiguousarray(input_data)
        outputs = session.run(None, {input_name: input_data})
        return {"outputs": outputs, "batch_size": B}

    if key == "netvlad":
        import torch

        model = models[key]
        input_data = merged_inputs[0]
        with torch.no_grad():
            tensor_input = torch.from_numpy(input_data).cuda()
            outputs = model(tensor_input)
            if isinstance(outputs, tuple) or isinstance(outputs, list):
                outputs = [out.cpu().numpy() for out in outputs]
            else:
                outputs = [outputs.cpu().numpy()]
        return {"outputs": outputs, "batch_size": B}

    raise ValueError(f"batched infer not supported for key={key}")


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
    samples = payload.get("samples")

    if (
        isinstance(samples, list)
        and len(samples) > 0
        and all(isinstance(s, dict) and "inputs" in s for s in samples)
    ):
        try:
            merged, B = samples_to_stacked_inputs(samples)
        except ValueError as ve:
            return Response(status_code=400, content=str(ve))
        try:
            body = _infer_preprocessing_batched(key, merged, B)
            return pack_response(body)
        except Exception as e:
            import traceback

            traceback.print_exc()
            return Response(status_code=500, content=str(e))

    inputs = payload.get("inputs", [])

    try:
        if key in ["angle", "roi"]:
            session = models[key]
            inp0 = session.get_inputs()[0]
            input_name = inp0.name
            input_data = np.asarray(inputs[0])
            target_dt = model_input_dtypes.get(key) or _numpy_dtype_for_onnx_type(
                inp0.type
            )
            # Same classifiers as prod Triton (UINT8). If ONNX/ORT still say float32, force uint8.
            if key in ("angle", "roi") and target_dt == np.float32:
                target_dt = np.uint8
            input_data = _coerce_input_to_dtype(input_data, target_dt)
            input_data = np.ascontiguousarray(input_data)
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
