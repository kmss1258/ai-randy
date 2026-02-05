import argparse
import os
from pathlib import Path

from PIL import Image

from app.pipeline import build_face_app, build_rembg_session, load_config, process_image, process_image_debug_steps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--debug-roi", action="store_true")
    args = parser.parse_args()

    if args.allow_cpu:
        os.environ["ALLOW_CPU_FALLBACK"] = "1"

    config = load_config()
    face_app = build_face_app(config)
    rembg_session = build_rembg_session(config)

    image = Image.open(args.input).convert("RGB")
    if args.debug_roi:
        _, steps = process_image_debug_steps(image, face_app, rembg_session, config)
        base = args.output
        stem = base.stem
        suffix = base.suffix or ".png"
        for label, step in steps:
            step_path = base.with_name(f"{stem}_{label}{suffix}")
            step.save(step_path, format="PNG")
    else:
        result = process_image(image, face_app, rembg_session, config)
        result.save(args.output, format="PNG")


if __name__ == "__main__":
    main()
