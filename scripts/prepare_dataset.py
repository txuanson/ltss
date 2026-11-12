"""Prepare image pairs according to Section 1.2 of the project specification.

Crops input images to a 16:9 aspect ratio (center crop) and resizes them to:
- Small : 1280 x 720  (Fast development and debugging)
- Medium: 1920 x 1080 (Main benchmark for the speedup table)
- Large : 3840 x 2160 (Shows GPU scaling)
"""

import argparse
from pathlib import Path
from typing import Dict, Tuple

from PIL import Image


RESOLUTIONS: Dict[str, Tuple[int, int]] = {
    "small": (1280, 720),
    "medium": (1920, 1080),
    "large": (3840, 2160),
}


def center_crop_16_9(img: Image.Image) -> Image.Image:
    """Center-crop a PIL image to a 16:9 aspect ratio."""
    w, h = img.size
    target_ratio = 16.0 / 9.0
    current_ratio = w / float(h)

    if abs(current_ratio - target_ratio) < 1e-3:
        return img

    if current_ratio > target_ratio:
        # Image is wider than 16:9 -> crop width
        new_w = int(round(h * target_ratio))
        left = (w - new_w) // 2
        return img.crop((left, 0, left + new_w, h))
    else:
        # Image is taller than 16:9 -> crop height
        new_h = int(round(w / target_ratio))
        top = (h - new_h) // 2
        return img.crop((0, top, w, top + new_h))


def prepare_pair(left_path: Path, right_path: Path, out_base_dir: Path) -> None:
    """Generate Small, Medium, and Large 16:9 image pairs."""
    img_a = center_crop_16_9(Image.open(left_path).convert("RGB"))
    img_b = center_crop_16_9(Image.open(right_path).convert("RGB"))

    for name, (tw, th) in RESOLUTIONS.items():
        sub_dir = out_base_dir / name
        sub_dir.mkdir(parents=True, exist_ok=True)
        res_a = img_a.resize((tw, th), Image.Resampling.LANCZOS)
        res_b = img_b.resize((tw, th), Image.Resampling.LANCZOS)
        out_a = sub_dir / left_path.name
        out_b = sub_dir / right_path.name
        res_a.save(out_a, quality=95)
        res_b.save(out_b, quality=95)
        print(f"[{name.upper():<6}] Saved {tw}x{th} pair to {sub_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Crop image pair to 16:9 and resize to Small/Medium/Large benchmark sets."
    )
    parser.add_argument("--left", default="data/input/1.jpg", help="Left input image")
    parser.add_argument("--right", default="data/input/2.jpg", help="Right input image")
    parser.add_argument(
        "--out-dir",
        default="data/input_resized",
        help="Output base directory for resized sets",
    )
    args = parser.parse_args()

    prepare_pair(Path(args.left), Path(args.right), Path(args.out_dir))


if __name__ == "__main__":
    main()
