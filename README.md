# Background removal + upper-body crop API

FastAPI service that removes background, detects the largest face, and crops a square ROI (2x face size) with face center at configurable height. Output is PNG with alpha.

## Run (GPU 0 only)

```bash
docker compose up --build
```

Service listens on `http://localhost:19167`.

## API

`POST /crop` (multipart form-data)

Example:

```bash
curl -s -X POST "http://localhost:19167/crop" \
  -F "file=@bg-crop/bg_org.jpg" \
  -o bg-crop/output.png
```

## Local verification (without Docker)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/verify.py bg-crop/bg_org.jpg bg-crop/output.png --allow-cpu
python scripts/verify.py bg-crop/bg_org.jpg bg-crop/output_debug.png --allow-cpu --debug-roi
# outputs: bg-crop/output_debug_01.png, _02.png, _03.png
```

## Configuration

Environment variables:

- `MAX_DET_SIDE` (default 1600)
- `REMBG_MAX_SIDE` (default 1600)
- `ROI_EXPAND` (default 2.0)
- `FACE_Y_RATIO` (default 0.428571)
- `DET_SIZE` (default 640)
- `REMBG_MODEL` (default u2netp)
- `FACE_MODEL` (default buffalo_l)
- `ALLOW_CPU_FALLBACK` (default 0)
- `MAX_CONCURRENT` (default 20)
