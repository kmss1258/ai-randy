import asyncio
import os
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response

from app.pipeline import CropConfig, build_face_app, build_rembg_session, load_config, process_bytes

app = FastAPI()

_config: CropConfig | None = None
_face_app = None
_rembg_session = None
_semaphore: asyncio.Semaphore | None = None


@app.on_event("startup")
def startup() -> None:
    global _config, _face_app, _rembg_session, _semaphore
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("NVIDIA_VISIBLE_DEVICES", "0")
    _config = load_config()
    _face_app = build_face_app(_config)
    _rembg_session = build_rembg_session(_config)
    max_concurrent = int(os.getenv("MAX_CONCURRENT", "20"))
    if max_concurrent < 1:
        max_concurrent = 1
    _semaphore = asyncio.Semaphore(max_concurrent)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/crop")
async def crop_image(file: UploadFile = File(...)) -> Response:
    if _config is None or _face_app is None or _rembg_session is None or _semaphore is None:
        raise HTTPException(status_code=500, detail="Service not initialized")
    try:
        async with _semaphore:
            content = await file.read()
            result = process_bytes(content, _face_app, _rembg_session, _config)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Processing failed") from exc
    return Response(content=result, media_type="image/png")
