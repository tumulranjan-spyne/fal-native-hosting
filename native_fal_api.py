import os
import time
import httpx
import msgpack
import msgpack_numpy as m
import numpy as np
from urllib.parse import urlparse

m.patch()

# Map models to their respective microservice endpoints
MODEL_ROUTING = {
    # Pre-processing
    "auto-classification-angle_classification": "FAL_ENDPOINT_PREPROCESS",
    "auto-classification-roi": "FAL_ENDPOINT_PREPROCESS",
    "netvlad": "FAL_ENDPOINT_PREPROCESS",
    
    # RemoveBG
    "DetectorYoloV5": "FAL_ENDPOINT_REMOVEBG",
    "Acconet_segmentation": "FAL_ENDPOINT_REMOVEBG",
    "Inspyrenet_segmentation": "FAL_ENDPOINT_REMOVEBG",
    "Mvanet_segmentation": "FAL_ENDPOINT_REMOVEBG",
    "PGNet_segmentation": "FAL_ENDPOINT_REMOVEBG",
    "real_esrgan": "FAL_ENDPOINT_REMOVEBG",
    
    # Colmap/Warping
    "banner_detection": "FAL_ENDPOINT_COLMAP",
    "bg_independent_flare": "FAL_ENDPOINT_COLMAP",
    "vanilla": "FAL_ENDPOINT_COLMAP", # Note: requires custom backend integration in the app
    
    # Post-processing
    "car_type_classifier": "FAL_ENDPOINT_POSTPROCESS",
    "Mvanet_plate_segmentation": "FAL_ENDPOINT_POSTPROCESS"
}

def _get_endpoint(model_name: str) -> str:
    env_var = MODEL_ROUTING.get(model_name)
    if not env_var:
        # Fallback to removebg if unknown, or handle dynamically
        env_var = "FAL_ENDPOINT_REMOVEBG" 
        
    endpoint = os.environ.get(env_var)
    if not endpoint:
        raise ValueError(f"Missing environment variable {env_var} for model {model_name}")
        
    return endpoint.rstrip('/')

def fal_infer(
    model_input,
    model_name,
    model_version="1",
    dtype="UINT8",
    input_names=None,
    output_names=None,
):
    """
    Drop-in replacement for fal_api.py's fal_infer.
    Uses msgpack to serialize numpy arrays directly to the FastAPI microservices.
    """
    endpoint = _get_endpoint(model_name)
    fal_key = os.environ.get("FAL_KEY")
    
    if not fal_key:
        raise ValueError("FAL_KEY environment variable is not set")
        
    # Prepare inputs
    # If model_input is a single numpy array, wrap it in a list for consistency
    if isinstance(model_input, np.ndarray):
        inputs = [model_input]
    elif isinstance(model_input, list) or isinstance(model_input, tuple):
        inputs = list(model_input)
    else:
        raise ValueError(f"Unsupported input type: {type(model_input)}")
        
    # Serialize with msgpack
    payload = msgpack.packb({"inputs": inputs})
    
    url = f"{endpoint}/infer/{model_name}?version={model_version}"
    
    headers = {
        "Authorization": f"Key {fal_key}",
        "Content-Type": "application/x-msgpack",
        "Accept": "application/x-msgpack"
    }
    
    # Send request
    try:
        # Use a longer timeout for heavy models
        timeout = 60.0 if "REMOVEBG" in endpoint or "COLMAP" in endpoint else 15.0
        
        with httpx.Client(timeout=timeout) as client:
            response = client.post(url, content=payload, headers=headers)
            
        if response.status_code != 200:
            raise Exception(f"Inference failed with status {response.status_code}: {response.text}")
            
        # Deserialize response
        result = msgpack.unpackb(response.content)
        outputs = result.get("outputs", [])
        
        # Match the return signature of the original fal_infer / triton_api
        # Usually returns the first output array
        if isinstance(outputs, list) and len(outputs) > 0:
            return outputs[0]
        elif isinstance(outputs, dict):
            # If it's a dict, return the values or the first value
            return list(outputs.values())[0]
        return outputs
        
    except Exception as e:
        # Wrap in BaseExceptionError if your pipeline expects it
        # from Utils.error_template import BaseExceptionError
        # raise BaseExceptionError(message=f"Native Fal inference error for {model_name}", error_code=500, e=e)
        raise e
