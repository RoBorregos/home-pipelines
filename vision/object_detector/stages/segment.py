"""
Segmentation stage: GroundingDINO bounding boxes + SAM3 masks.

Input:  {workdir}/images/{ClassName}/
Output: {workdir}/cropped/{ClassName}/  (BGRA PNGs cropped to object bbox)

CLI usage:
    python -m stages.segment /path/to/workdir [--debug]
"""

import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

_BASE_DIR   = Path(__file__).parent.parent   # object_detector/
_MODELS_DIR = _BASE_DIR / "models"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic"}


def _ensure_importable():
    # GroundingDINO is installed via groundingdino-py (compiled _C extension).
    # Only sam3 needs a manual path since it has no PyPI package.
    p = str(_MODELS_DIR / "sam3")
    if p not in sys.path and Path(p).exists():
        sys.path.insert(0, p)


# ── GroundingDINO helpers ────────────────────────────────────────────────────

def _load_grounding_model(config_file: str, checkpoint: str, device: str):
    from groundingdino.util.slconfig import SLConfig
    from groundingdino.util.utils import clean_state_dict
    from groundingdino.models import build_model

    args = SLConfig.fromfile(config_file)
    args.device = device
    model = build_model(args)
    ckpt = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(clean_state_dict(ckpt["model"]), strict=False)
    return model.eval().to(device)


def _image_to_tensor(image_pil: Image.Image):
    import groundingdino.datasets.transforms as T
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    tensor, _ = transform(image_pil, None)
    return tensor


def _detect_boxes(model, image_tensor, caption: str, box_thr: float, text_thr: float, device: str):
    from groundingdino.util.utils import get_phrases_from_posmap

    caption = caption.lower().strip().rstrip(".") + "."
    tensor = image_tensor.to(device)

    with torch.no_grad():
        outputs = model(tensor[None], captions=[caption])

    logits = outputs["pred_logits"].sigmoid()[0]
    boxes = outputs["pred_boxes"][0]

    keep = logits.max(dim=1)[0] > box_thr
    logits, boxes = logits[keep].cpu(), boxes[keep].cpu()

    tokenizer = model.tokenizer
    tokenized = tokenizer(caption)
    phrases = [
        get_phrases_from_posmap(l > text_thr, tokenized, tokenizer)
        for l in logits
    ]
    return boxes, phrases


# ── Box / mask helpers ───────────────────────────────────────────────────────

def _box_to_xyxy(box, W: int, H: int, padding: int = 10) -> tuple[int, int, int, int]:
    x0 = int(box[0] * W - box[2] * W / 2) - padding
    y0 = int(box[1] * H - box[3] * H / 2) - padding
    x1 = int(box[0] * W + box[2] * W / 2) + padding
    y1 = int(box[1] * H + box[3] * H / 2) + padding
    return max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)


def _xyxy_to_cxcywh_norm(x0, y0, x1, y1, W, H) -> list[float]:
    W, H = max(float(W), 1e-6), max(float(H), 1e-6)
    clamp = lambda v: max(0.0, min(1.0, float(v)))
    return [
        clamp((x0 + x1) / 2 / W),
        clamp((y0 + y1) / 2 / H),
        clamp(max((x1 - x0) / W, 1e-6)),
        clamp(max((y1 - y0) / H, 1e-6)),
    ]


def _apply_morphology(mask: np.ndarray) -> np.ndarray:
    kernel = np.ones((9, 9), np.uint8)
    _, binary = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)


# ── Per-image processing ─────────────────────────────────────────────────────

def _segment_image(
    img_path: Path,
    class_name: str,
    cropped_dir: Path,
    grounding_model,
    sam3_processor,
    device: str,
    box_thr: float,
    text_thr: float,
) -> int:
    image_pil = Image.open(img_path).convert("RGB")
    W, H = image_pil.size
    img_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

    boxes, phrases = _detect_boxes(
        grounding_model, _image_to_tensor(image_pil), class_name, box_thr, text_thr, device
    )
    if len(boxes) == 0:
        logger.warning("  No detections: %s", img_path.name)
        return 0

    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        sam3_state = sam3_processor.set_image(image_pil.copy())
        sam3_state = sam3_processor.set_text_prompt(prompt=class_name, state=sam3_state)

    saved = 0
    for i, (box, _) in enumerate(zip(boxes, phrases)):
        x0, y0, x1, y1 = _box_to_xyxy(box, W, H)
        norm_box = _xyxy_to_cxcywh_norm(x0, y0, x1, y1, W, H)

        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            sam3_state["geometric_prompt"] = sam3_processor.model._get_dummy_prompt()
            sam3_state = sam3_processor.add_geometric_prompt(box=norm_box, label=True, state=sam3_state)

        masks = sam3_state.get("masks")
        scores = sam3_state.get("scores")
        if masks is None or masks.nelement() == 0:
            continue

        best = int(torch.argmax(scores).item())
        raw_mask = masks[best, 0].detach().cpu().numpy().astype(np.uint8)
        clean_mask = _apply_morphology(raw_mask)

        bgra = cv2.cvtColor(img_cv.copy(), cv2.COLOR_BGR2BGRA)
        bgra[:, :, 3] = clean_mask

        # crop to tight bbox inline — no intermediate segmented/ directory
        rgba_pil = Image.fromarray(cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGBA), "RGBA")
        bbox = Image.composite(rgba_pil, Image.new("RGBA", rgba_pil.size), rgba_pil).getbbox()
        if not bbox:
            continue

        out_path = cropped_dir / f"{img_path.stem}_{i}.png"
        rgba_pil.crop(bbox).save(str(out_path))
        saved += 1

    return saved


# ── Public entry point ────────────────────────────────────────────────────────

def run(
    workdir: str,
    box_threshold: float = 0.3,
    text_threshold: float = 0.25,
    sam3_score_threshold: float = 0.4,
) -> None:
    workdir = Path(workdir)
    images_dir  = workdir / "images"
    cropped_dir = workdir / "cropped"

    class_dirs = [d for d in sorted(images_dir.iterdir()) if d.is_dir()]
    if not class_dirs:
        raise ValueError(f"No class subdirectories found in {images_dir}")
    logger.info("Classes: %s", [d.name for d in class_dirs])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    config_file = _MODELS_DIR / "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    gdino_ckpt  = _MODELS_DIR / "groundingdino_swint_ogc.pth"
    sam3_ckpt   = _MODELS_DIR / "sam3.pt"

    for p in (config_file, gdino_ckpt, sam3_ckpt):
        if not p.exists():
            raise FileNotFoundError(f"Required model file not found: {p}")

    logger.info("Loading GroundingDINO…")
    _ensure_importable()
    grounding_model = _load_grounding_model(str(config_file), str(gdino_ckpt), device)

    logger.info("Loading SAM3…")
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    sam3_model = build_sam3_image_model(
        device=device, eval_mode=True,
        checkpoint_path=str(sam3_ckpt), load_from_HF=False,
    )
    sam3_processor = Sam3Processor(sam3_model, device=device, confidence_threshold=sam3_score_threshold)

    total = 0
    for class_dir in class_dirs:
        cropped_class = cropped_dir / class_dir.name
        if cropped_class.exists() and any(cropped_class.iterdir()):
            logger.info("Skipping %s — already in cropped/", class_dir.name)
            continue

        cropped_class.mkdir(parents=True, exist_ok=True)
        images = [f for f in class_dir.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS]
        n_images = len(images)
        logger.info("--- %s (%d images) ---", class_dir.name, n_images)

        class_saved = 0
        for idx, img_path in enumerate(images):
            try:
                n = _segment_image(
                    img_path, class_dir.name, cropped_class,
                    grounding_model, sam3_processor,
                    device, box_threshold, text_threshold,
                )
                class_saved += n
                total += n
            except Exception as exc:
                logger.error("  Failed %s: %s", img_path.name, exc)
            if (idx + 1) % 10 == 0 or (idx + 1) == n_images:
                pct = (idx + 1) * 100 // n_images
                logger.info("  [%s] %d%% (%d/%d images, %d saved)",
                            class_dir.name, pct, idx + 1, n_images, class_saved)

    logger.info("Stage complete — %d total cropped segments", total)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG if "--debug" in sys.argv else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        stream=sys.stdout,
        force=True,
    )
    _workdir = next((a for a in sys.argv[1:] if not a.startswith("--")), ".")
    run(_workdir)
