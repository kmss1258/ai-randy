from typing import Tuple


def compute_crop_box(
    face_center: Tuple[float, float],
    face_size: Tuple[float, float],
    image_size: Tuple[int, int],
    roi_expand: float,
    face_y_ratio: float,
) -> Tuple[int, int, int, int]:
    width, height = image_size
    base = max(face_size)
    side = base * roi_expand
    max_side = min(width, height)
    if side > max_side:
        side = float(max_side)
    cx, cy = face_center
    left = cx - side / 2.0
    top = cy - side * face_y_ratio
    right = left + side
    bottom = top + side

    left = max(0.0, left)
    top = max(0.0, top)
    right = min(float(width), right)
    bottom = min(float(height), bottom)

    return int(round(left)), int(round(top)), int(round(right)), int(round(bottom))
