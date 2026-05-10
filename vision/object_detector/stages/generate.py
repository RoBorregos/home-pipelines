"""
Dataset generation stage: composite fg objects on backgrounds → YOLO dataset.

Input:  {workdir}/DS_res/{ClassName}/   (cropped segmented PNGs)
        {workdir}/bg/                    (background images)
Output: {output_folder}/train|valid|test/images+labels/
        {output_folder}/data.yaml

CLI usage:
    python -m stages.generate /path/to/workdir /path/to/output [--n 15000]
"""

import logging
import math
import os
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, UnidentifiedImageError

logger = logging.getLogger(__name__)

# ── Augmentation constants ────────────────────────────────────────────────────

BRIGHTNESS_FACTOR = 0.3
CONTRAST_FACTOR   = 0.2
SATURATION_FACTOR = 0.3
HUE_FACTOR        = 0.05
BLUR_FACTOR       = 0.25
NOISE_PROBABILITY = 0.5
NOISE_FACTOR      = 0.15
BLOB_COUNT_MIN    = 0
BLOB_COUNT_MAX    = 2
BLOB_SIZE         = (15, 150)
BLOB_PROBABILITY  = 0.15
QUALITY_FACTOR    = 0.25


# ── Augmentation functions ────────────────────────────────────────────────────

def _adjust_brightness(img: Image.Image) -> Image.Image:
    return ImageEnhance.Brightness(img).enhance(random.uniform(1 - BRIGHTNESS_FACTOR, 1 + BRIGHTNESS_FACTOR))

def _adjust_contrast(img: Image.Image) -> Image.Image:
    return ImageEnhance.Contrast(img).enhance(random.uniform(1 - CONTRAST_FACTOR, 1 + CONTRAST_FACTOR))

def _adjust_saturation(img: Image.Image) -> Image.Image:
    return ImageEnhance.Color(img).enhance(random.uniform(1 - SATURATION_FACTOR, 1 + SATURATION_FACTOR))

def _adjust_hue(img: Image.Image) -> Image.Image:
    factor = random.uniform(1 - HUE_FACTOR, 1 + HUE_FACTOR)
    hsv = img.convert("HSV")
    h, s, v = hsv.split()
    h = h.point(lambda p: (p + int(factor * 255)) % 256)
    return Image.merge("HSV", (h, s, v)).convert("RGB")

def _add_blur(img: Image.Image) -> Image.Image:
    radius = random.uniform(0, BLUR_FACTOR)
    return img.filter(ImageFilter.GaussianBlur(radius=radius)) if radius > 0 else img

def _add_noise(img: Image.Image) -> Image.Image:
    if not random.random() < NOISE_PROBABILITY:
        return img
    factor = random.uniform(0, NOISE_FACTOR)
    arr = np.array(img, dtype=np.float32)
    noise = np.random.normal(0, factor * 255, arr.shape[:2] + (3,)).astype(np.float32)
    arr[..., :3] = np.clip(arr[..., :3] + noise, 0, 255)
    return Image.fromarray(arr.astype(np.uint8), mode=img.mode)

def _add_blobs(img: Image.Image) -> Image.Image:
    if random.random() > BLOB_PROBABILITY:
        return img
    draw = ImageDraw.Draw(img)
    for _ in range(random.randint(BLOB_COUNT_MIN, BLOB_COUNT_MAX)):
        cx, cy = random.randint(0, img.width), random.randint(0, img.height)
        r = random.randint(*BLOB_SIZE)
        n = random.randint(8, 16)
        step = 2 * math.pi / n
        pts = [
            (
                cx + r * (1 + random.uniform(-0.6, 0.6)) * math.cos(i * step + random.uniform(-0.4, 0.4) * step),
                cy + r * (1 + random.uniform(-0.6, 0.6)) * math.sin(i * step + random.uniform(-0.4, 0.4) * step),
            )
            for i in range(n)
        ]
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), random.randint(100, 255))
        draw.polygon(pts, fill=color)
    return img

def _alter_quality(img: Image.Image) -> Image.Image:
    q = random.uniform(QUALITY_FACTOR, 1)
    frame = np.array(img)
    _, enc = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(q * 100)])
    return Image.fromarray(cv2.imdecode(enc, cv2.IMREAD_COLOR))

def _augment(img: Image.Image) -> Image.Image:
    fns = [_adjust_brightness, _adjust_contrast, _adjust_saturation,
           _add_blobs, _adjust_hue, _add_blur, _add_noise, _alter_quality]
    random.shuffle(fns)
    for fn in fns:
        img = fn(img)
    return img


# ── Placement helper ──────────────────────────────────────────────────────────

def _try_place(bg: Image.Image, fg: Image.Image, occupied: np.ndarray, max_overlap_pct: float = 25.0):
    """Returns (x, y) of a valid placement or None after 50 attempts."""
    max_x = bg.width - fg.width
    max_y = bg.height - fg.height
    if max_x < 0 or max_y < 0:
        return None

    fg_mask = np.array(fg.split()[-1]) > 0
    mh, mw = fg_mask.shape

    for _ in range(50):
        x, y = random.randint(0, max_x), random.randint(0, max_y)
        if y + mh > occupied.shape[0] or x + mw > occupied.shape[1]:
            continue

        crop = occupied[y:y + mh, x:x + mw]
        overlap = np.sum(np.logical_and(fg_mask, crop))
        fg_area = np.sum(fg_mask)
        occ_area = np.sum(crop)

        if overlap <= max_overlap_pct / 100 * fg_area and \
           overlap <= max_overlap_pct / 100 * occ_area if occ_area else True:
            occupied[y:y + mh, x:x + mw] = np.logical_or(crop, fg_mask)
            return x, y
    return None


# ── Dataset generation ────────────────────────────────────────────────────────

def _load_bg_images(bg_dir: Path) -> list[Path]:
    files = [f for f in bg_dir.iterdir() if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    if not files:
        raise FileNotFoundError(f"No background images found in {bg_dir}")
    return files


def _scale_fg(fg: Image.Image, bg: Image.Image, min_ratio=0.05, max_ratio=0.35) -> Image.Image:
    ow, oh = fg.size
    scale_min = max(bg.width * min_ratio / ow, bg.height * min_ratio / oh, 0.01)
    scale_max = min(bg.width * max_ratio / ow, bg.height * max_ratio / oh, 1.0)
    scale_max = max(scale_max, scale_min)
    scale = random.uniform(scale_min, scale_max)
    return fg.resize((max(1, int(ow * scale)), max(1, int(oh * scale))), Image.BICUBIC)


def run(
    workdir: str,
    output_folder: str,
    bg_dir: str | None = None,
    images_to_generate: int = 15000,
    validation_ratio: float = 0.1,
    test_ratio: float = 0.1,
    max_objects_per_image: int = 8,
    original_classes: list[str] | None = None,
) -> str:
    """Returns path to the generated data.yaml."""
    workdir = Path(workdir)
    out = Path(output_folder)
    ds_res_dir = workdir / "cropped"
    bg_dir = Path(bg_dir) if bg_dir else workdir / "backgrounds"

    if not ds_res_dir.exists():
        raise FileNotFoundError(f"cropped/ not found at {ds_res_dir}. Run segmentation first.")

    # ── Class setup ──────────────────────────────────────────────────────────
    class_dirs = sorted([d for d in ds_res_dir.iterdir() if d.is_dir()])
    if not class_dirs:
        raise ValueError(f"No class directories in {ds_res_dir}")

    detected = [d.name for d in class_dirs]
    final_classes = list(original_classes or [])
    for cls in detected:
        if cls not in final_classes:
            final_classes.append(cls)

    class_id = {cls: i for i, cls in enumerate(final_classes)}
    fg_files = {d.name: [f for f in d.iterdir() if f.suffix == ".png"] for d in class_dirs}
    logger.info("Classes: %s", class_id)

    # ── Directory structure ───────────────────────────────────────────────────
    for split in ("train", "valid", "test"):
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        (out / split / "labels").mkdir(parents=True, exist_ok=True)

    bg_files = _load_bg_images(bg_dir)
    logger.info("Backgrounds: %d | FG classes: %d | Target images: %d",
                len(bg_files), len(class_dirs), images_to_generate)

    # ── Generation loop ───────────────────────────────────────────────────────
    objects_list = list(fg_files.keys())
    train_dir = out / "train"

    for img_id in range(images_to_generate):
        if img_id % 500 == 0:
            logger.info("  %d / %d", img_id, images_to_generate)

        label_path = train_dir / "labels" / f"{img_id}.txt"
        label_path.write_text("")

        num_objects = 0 if random.random() < 0.10 else random.randint(1, max_objects_per_image)

        bg_img = Image.open(random.choice(bg_files)).convert("RGBA")
        occupied = np.zeros((bg_img.height, bg_img.width), dtype=np.uint8)

        for class_name in random.choices(objects_list, k=num_objects):
            files = fg_files[class_name]
            if not files:
                continue
            fg = Image.open(random.choice(files)).convert("RGBA")
            fg = fg.rotate(random.randint(-5, 5), resample=Image.BICUBIC, expand=True)
            fg = _scale_fg(fg, bg_img)
            fg = ImageEnhance.Brightness(fg).enhance(random.uniform(0.8, 1.2))
            fg = ImageEnhance.Contrast(fg).enhance(random.uniform(0.8, 1.2))
            fg = ImageEnhance.Color(fg).enhance(random.uniform(0.8, 1.2))
            fg = fg.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.0, 0.8)))

            pos = _try_place(bg_img, fg, occupied)
            if pos is None:
                continue
            x, y = pos

            # Segmentation label (polygon from alpha mask)
            fg_arr = np.array(fg)
            mask = fg_arr[:, :, 3] != 0
            coords = []
            for row in range(mask.shape[0]):
                for col in range(mask.shape[1]):
                    if mask[row, col]:
                        coords.extend([(col + x) / bg_img.width, (row + y) / bg_img.height])

            if coords:
                with label_path.open("a") as f:
                    f.write(f"{class_id[class_name]} {' '.join(map(str, coords))}\n")

            bg_img.paste(fg, (x, y), fg)

        bg_img = _augment(bg_img)
        bg_img.convert("RGB").save(train_dir / "images" / f"{img_id}.jpg", quality=100)

    logger.info("Generation complete. Splitting train/valid/test…")

    # ── Train / valid / test split ────────────────────────────────────────────
    all_imgs = list((train_dir / "images").iterdir())
    random.shuffle(all_imgs)
    n_valid = int(len(all_imgs) * validation_ratio)
    n_test  = int(len(all_imgs) * test_ratio)

    def _move(files: list[Path], split: str) -> None:
        for f in files:
            shutil.move(str(f), str(out / split / "images" / f.name))
            label = train_dir / "labels" / f.with_suffix(".txt").name
            if label.exists():
                shutil.move(str(label), str(out / split / "labels" / label.name))

    _move(all_imgs[:n_valid], "valid")
    _move(all_imgs[n_valid: n_valid + n_test], "test")

    remaining = len(list((train_dir / "images").iterdir()))
    logger.info("Split → train: %d | valid: %d | test: %d", remaining, n_valid, n_test)

    # ── data.yaml ─────────────────────────────────────────────────────────────
    yaml_path = out / "data.yaml"
    yaml_path.write_text(yaml.dump({
        "train": str(out / "train" / "images"),
        "val":   str(out / "valid" / "images"),
        "test":  str(out / "test"  / "images"),
        "nc":    len(final_classes),
        "names": final_classes,
    }, default_flow_style=False))

    logger.info("data.yaml written: %s", yaml_path)
    logger.info("Stage complete.")
    return str(yaml_path)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        stream=sys.stdout,
        force=True,
    )
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) < 2:
        print("Usage: python -m stages.generate <workdir> <output_folder> [--n 15000]")
        sys.exit(1)
    n = int(next((sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == "--n"), 15000))
    run(workdir=args[0], output_folder=args[1], images_to_generate=n)
