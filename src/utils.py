import msgpack
import msgpack_numpy as m
from fastapi import FastAPI, Request, Response

m.patch()

def create_app(title: str):
    app = FastAPI(title=title)
    
    @app.get("/health")
    def health():
        return {"status": "ok"}
        
    return app

async def unpack_request(request: Request):
    """Unpack msgpack numpy arrays from request body."""
    body = await request.body()
    return msgpack.unpackb(body)

def pack_response(data: dict):
    """Pack dictionary with numpy arrays into msgpack response."""
    packed = msgpack.packb(data)
    return Response(content=packed, media_type="application/x-msgpack")
