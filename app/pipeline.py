import io
import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import onnxruntime as ort
from insightface.app import FaceAnalysis
from PIL import Image, ImageDraw, UnidentifiedImageError
from PIL.Image import Image as PilImage
from rembg import new_session, remove

from app.crop_math import compute_crop_box


@dataclass
class CropConfig:
    max_det_side: int = 1600
    rembg_max_side: int = 1600
    roi_expand: float = 2.0
    face_y_ratio: float = 3.0 / 7.0
    det_size: int = 640
    rembg_model: str = "u2netp"
    face_model: str = "buffalo_l"
    allow_cpu_fallback: bool = False
    alpha_matting: bool = True
    alpha_fg: int = 270
    alpha_bg: int = 20
    alpha_erode: int = 11


def load_config() -> CropConfig:
    return CropConfig(
        max_det_side=int(os.getenv("MAX_DET_SIDE", "1600")),
        rembg_max_side=int(os.getenv("REMBG_MAX_SIDE", "1600")),
        roi_expand=float(os.getenv("ROI_EXPAND", "2.0")),
        face_y_ratio=float(os.getenv("FACE_Y_RATIO", "0.428571")),
        det_size=int(os.getenv("DET_SIZE", "640")),
        rembg_model=os.getenv("REMBG_MODEL", "u2netp"),
        face_model=os.getenv("FACE_MODEL", "buffalo_l"),
        allow_cpu_fallback=os.getenv("ALLOW_CPU_FALLBACK", "0") == "1",
        alpha_matting=os.getenv("ALPHA_MATTING", "1") == "1",
        alpha_fg=int(os.getenv("ALPHA_FG", "270")),
        alpha_bg=int(os.getenv("ALPHA_BG", "20")),
        alpha_erode=int(os.getenv("ALPHA_ERODE", "11")),
    )


def _available_providers() -> List[str]:
    return ort.get_available_providers()


def _select_providers(allow_cpu_fallback: bool) -> List[str]:
    available = _available_providers()
    providers = [p for p in ["CUDAExecutionProvider", "CPUExecutionProvider"] if p in available]
    if "CUDAExecutionProvider" not in providers and not allow_cpu_fallback:
        raise RuntimeError("CUDAExecutionProvider not available. Set ALLOW_CPU_FALLBACK=1 to continue on CPU.")
    return providers or ["CPUExecutionProvider"]


def _select_ctx_id(providers: list) -> int:
    return 0 if "CUDAExecutionProvider" in providers else -1


def build_face_app(config: CropConfig) -> FaceAnalysis:
    providers = _select_providers(config.allow_cpu_fallback)
    face_app = FaceAnalysis(name=config.face_model, providers=providers)
    face_app.prepare(ctx_id=_select_ctx_id(providers), det_size=(config.det_size, config.det_size))
    return face_app


def build_rembg_session(config: CropConfig):
    return new_session(config.rembg_model)


def _resize_for_detection(image: Image.Image, max_side: int) -> Tuple[Image.Image, float]:
    if max_side <= 0:
        return image, 1.0
    width, height = image.size
    long_side = max(width, height)
    if long_side <= max_side:
        return image, 1.0
    scale = long_side / max_side
    new_size = (int(round(width / scale)), int(round(height / scale)))
    return image.resize(new_size, Image.Resampling.LANCZOS), scale


def _resize_for_rembg(image: Image.Image, max_side: int) -> Tuple[Image.Image, float]:
    if max_side <= 0:
        return image, 1.0
    width, height = image.size
    long_side = max(width, height)
    if long_side <= max_side:
        return image, 1.0
    scale = long_side / max_side
    new_size = (int(round(width / scale)), int(round(height / scale)))
    return image.resize(new_size, Image.Resampling.LANCZOS), scale


def _image_to_bgr(image: Image.Image) -> np.ndarray:
    rgb = np.array(image.convert("RGB"))
    return rgb[:, :, ::-1].copy()


def process_image(
    image: Image.Image,
    face_app: FaceAnalysis,
    rembg_session,
    config: CropConfig,
) -> Image.Image:
    det_image, scale = _resize_for_detection(image, config.max_det_side)
    det_bgr = _image_to_bgr(det_image)
    faces = face_app.get(det_bgr)
    if not faces:
        raise ValueError("No face detected")

    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    bbox = face.bbox.astype(float)
    if scale != 1.0:
        bbox *= scale

    rembg_input, rembg_scale = _resize_for_rembg(image, config.rembg_max_side)
    rembg_rgba = remove(
        rembg_input,
        session=rembg_session,
        alpha_matting=config.alpha_matting,
        alpha_matting_foreground_threshold=config.alpha_fg,
        alpha_matting_background_threshold=config.alpha_bg,
        alpha_matting_erode_size=config.alpha_erode,
    )
    if not isinstance(rembg_rgba, PilImage):
        rembg_rgba = Image.open(io.BytesIO(rembg_rgba)).convert("RGBA")
    if rembg_scale != 1.0:
        alpha = rembg_rgba.getchannel("A").resize(image.size, Image.Resampling.LANCZOS)
        rgba = image.convert("RGBA")
        rgba.putalpha(alpha)
    else:
        rgba = rembg_rgba.convert("RGBA")

    bbox_center_x = (bbox[0] + bbox[2]) / 2.0
    bbox_center_y = (bbox[1] + bbox[3]) / 2.0
    bbox_width = bbox[2] - bbox[0]
    bbox_height = bbox[3] - bbox[1]
    
    side = max(bbox_width, bbox_height) * config.roi_expand
    
    left = bbox_center_x - side / 2.0
    top = bbox_center_y - side * config.face_y_ratio
    right = left + side
    bottom = top + side

    img_width, img_height = rgba.size
    left = max(0.0, left)
    top = max(0.0, top)
    right = min(float(img_width), right)
    bottom = min(float(img_height), bottom)

    crop_box = (
        int(round(left)),
        int(round(top)),
        int(round(right)),
        int(round(bottom)),
    )

    cropped = rgba.crop(crop_box)
    if cropped.width == 0 or cropped.height == 0:
        raise ValueError("Invalid crop region")
    return cropped


def process_image_debug_steps(
    image: Image.Image,
    face_app: FaceAnalysis,
    rembg_session,
    config: CropConfig,
) -> Tuple[Image.Image, List[Tuple[str, Image.Image]]]:
    steps = []
    
    det_image, scale = _resize_for_detection(image, config.max_det_side)
    det_bgr = _image_to_bgr(det_image)
    faces = face_app.get(det_bgr)
    if not faces:
        raise ValueError("No face detected")

    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    bbox = face.bbox.astype(float)
    kps = face.kps.astype(float)
    if scale != 1.0:
        bbox *= scale
        kps *= scale

    step01 = image.convert("RGB").copy()
    draw01 = ImageDraw.Draw(step01)
    draw01.rectangle(
        (bbox[0], bbox[1], bbox[2], bbox[3]), outline=(0, 255, 0), width=3
    )
    for point in kps:
        x, y = float(point[0]), float(point[1])
        radius = 3.0
        draw01.ellipse(
            (x - radius, y - radius, x + radius, y + radius), outline=(0, 255, 0), width=2
        )
    steps.append(("01", step01))

    rembg_input, rembg_scale = _resize_for_rembg(image, config.rembg_max_side)
    rembg_rgba = remove(
        rembg_input,
        session=rembg_session,
        alpha_matting=config.alpha_matting,
        alpha_matting_foreground_threshold=config.alpha_fg,
        alpha_matting_background_threshold=config.alpha_bg,
        alpha_matting_erode_size=config.alpha_erode,
    )
    if not isinstance(rembg_rgba, PilImage):
        rembg_rgba = Image.open(io.BytesIO(rembg_rgba)).convert("RGBA")
    if rembg_scale != 1.0:
        alpha = rembg_rgba.getchannel("A").resize(image.size, Image.Resampling.LANCZOS)
        rgba = image.convert("RGBA")
        rgba.putalpha(alpha)
    else:
        rgba = rembg_rgba.convert("RGBA")
    step02 = rgba.copy()
    steps.append(("02", step02))

    bbox_center_x = (bbox[0] + bbox[2]) / 2.0
    bbox_center_y = (bbox[1] + bbox[3]) / 2.0
    bbox_width = bbox[2] - bbox[0]
    bbox_height = bbox[3] - bbox[1]
    
    side = max(bbox_width, bbox_height) * config.roi_expand
    
    left = bbox_center_x - side / 2.0
    top = bbox_center_y - side * config.face_y_ratio
    right = left + side
    bottom = top + side

    img_width, img_height = rgba.size
    left = max(0.0, left)
    top = max(0.0, top)
    right = min(float(img_width), right)
    bottom = min(float(img_height), bottom)

    crop_box = (
        int(round(left)),
        int(round(top)),
        int(round(right)),
        int(round(bottom)),
    )

    cropped = rgba.crop(crop_box)
    
    step03 = cropped.copy()
    draw03 = ImageDraw.Draw(step03)
    face_left = bbox[0] - crop_box[0]
    face_top = bbox[1] - crop_box[1]
    face_right = bbox[2] - crop_box[0]
    face_bottom = bbox[3] - crop_box[1]
    draw03.rectangle(
        (face_left, face_top, face_right, face_bottom),
        outline=(0, 255, 0, 255),
        width=3,
    )
    steps.append(("03", step03))

    return cropped, steps


def process_bytes(
    image_bytes: bytes,
    face_app: FaceAnalysis,
    rembg_session,
    config: CropConfig,
) -> bytes:
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError("Invalid image") from exc
    cropped = process_image(image, face_app, rembg_session, config)
    output = io.BytesIO()
    cropped.save(output, format="PNG")
    return output.getvalue()
