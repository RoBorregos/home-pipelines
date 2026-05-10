import asyncio
import io
import logging
import os
import re
import subprocess
import tempfile
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
from pydantic import BaseModel

import pipeline_runner
import state as ps
from state import SEGMENT, GENERATE, TRAIN, BASE_DIR, RUNS_DIR

logging.basicConfig(level=logging.INFO)

API_KEY = os.environ.get("PIPELINE_API_KEY", "TESTING*/*1234567890")

app = FastAPI(title="Object Detector Pipeline")

templates = Jinja2Templates(directory=str(Path(__file__).parent / "review_app" / "templates"))
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "review_app" / "static")), name="static")


def _auth(key: str) -> None:
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


def _safe_child(raw: str, base: Path) -> Path:
    resolved = Path(raw).resolve()
    if not str(resolved).startswith(str(base.resolve())):
        raise HTTPException(status_code=400, detail=f"Invalid path: {raw}")
    return resolved


# ── Pages ─────────────────────────────────────────────────────────────────────

@app.get("/")
def dashboard(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.get("/review")
def review_page(request: Request):
    return templates.TemplateResponse(request, "review.html", {"class_name": ""})


@app.get("/infer")
def infer_page(request: Request):
    return templates.TemplateResponse(request, "infer.html")


# ── Status & runs ─────────────────────────────────────────────────────────────

@app.get("/status")
def get_status():
    return asdict(ps.load())


@app.get("/runs")
def get_runs():
    return {"runs": ps.list_runs()}


class CreateRunBody(BaseModel):
    name: str


@app.post("/runs")
def create_run(body: CreateRunBody, x_api_key: str = Header(None)):
    _auth(x_api_key)
    name = body.name.strip().replace(" ", "_")
    if not name:
        raise HTTPException(status_code=400, detail="Run name cannot be empty")
    run_dir = RUNS_DIR / name
    if run_dir.exists():
        raise HTTPException(status_code=409, detail=f"Run '{name}' already exists")
    for sub in ("images", "cropped", "dataset", "training", "logs"):
        (run_dir / sub).mkdir(parents=True)
    # Activate the new run
    s = ps.load()
    s.run_name = name
    s.segment_done = s.review_done = s.generate_done = s.train_done = False
    s.error = s.running = s.data_yaml = s.best_weights = ""
    s.segmented_classes = {}
    ps.save(s)
    return {"run_name": name}


@app.post("/runs/{name}/activate")
def activate_run(name: str, x_api_key: str = Header(None)):
    _auth(x_api_key)
    if not (RUNS_DIR / name).exists():
        raise HTTPException(status_code=404, detail=f"Run '{name}' not found")
    s = ps.load()
    if s.running:
        raise HTTPException(status_code=409, detail=f"Stage '{s.running}' is running — wait for it to finish")
    s.run_name = name
    # Restore completion state from filesystem
    run_dir = RUNS_DIR / name
    cropped = run_dir / "cropped"
    s.segmented_classes = {
        d.name: True for d in cropped.iterdir() if d.is_dir() and any(d.iterdir())
    } if cropped.exists() else {}
    s.segment_done  = bool(s.segmented_classes)
    s.generate_done = (run_dir / "dataset" / "data.yaml").exists()
    s.train_done    = (run_dir / "training").exists()
    s.review_done   = s.segment_done  # assume reviewed if segmented (user can override)
    data_yaml = run_dir / "dataset" / "data.yaml"
    s.data_yaml = str(data_yaml) if data_yaml.exists() else ""
    ps.save(s)
    return asdict(s)


@app.post("/pipeline/reset")
def pipeline_reset(x_api_key: str = Header(None)):
    _auth(x_api_key)
    s = ps.load()
    s.running = s.error = ""
    ps.save(s)
    return {"running": ""}


# ── Log streaming ─────────────────────────────────────────────────────────────

@app.get("/logs/stream")
async def logs_stream():
    async def generate():
        s = ps.load()
        log_file = Path(s.log_file) if s.log_file else None
        if not log_file or not log_file.exists():
            yield "data: [no active log]\n\n"
            return
        pos = 0
        while True:
            with log_file.open(encoding="utf-8") as f:
                f.seek(pos)
                chunk = f.read()
                pos = f.tell()
            if chunk:
                for line in chunk.splitlines():
                    yield f"data: {line}\n\n"
            current = ps.load()
            if not current.running:
                yield f"data: {'[ERROR] ' + current.error if current.error else '[DONE]'}\n\n"
                break
            await asyncio.sleep(0.3)

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/logs/file")
def logs_file():
    s = ps.load()
    if not s.log_file or not Path(s.log_file).exists():
        return {"lines": []}
    return {"lines": Path(s.log_file).read_text(encoding="utf-8").splitlines()}


# ── Stage control ─────────────────────────────────────────────────────────────

@app.post("/stage/segment/run")
def stage_segment(x_api_key: str = Header(None)):
    _auth(x_api_key)
    try:
        pipeline_runner.start_segment()
        return {"running": SEGMENT}
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


class GenerateBody(BaseModel):
    images_to_generate: int = 15000


@app.post("/stage/generate/run")
def stage_generate(body: GenerateBody, x_api_key: str = Header(None)):
    _auth(x_api_key)
    s = ps.load()
    if not s.review_done:
        raise HTTPException(status_code=409, detail="Review must be done first")
    try:
        pipeline_runner.start_generate(body.images_to_generate)
        return {"running": GENERATE}
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


class TrainBody(BaseModel):
    device: str = "0"
    epochs: int = 100
    batch: int = 64


@app.post("/stage/train/run")
def stage_train(body: TrainBody, x_api_key: str = Header(None)):
    _auth(x_api_key)
    s = ps.load()
    if not s.generate_done:
        raise HTTPException(status_code=409, detail="Generate stage must be done first")
    try:
        pipeline_runner.start_train(body.device, body.epochs, body.batch)
        return {"running": TRAIN}
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


# ── Video upload via gdown ────────────────────────────────────────────────────
# Video naming convention: <ClassName><Index>.<ext>
# Examples: Soap1.mp4, Soap2.mp4, Mug1.mp4, dish_soap1.mp4
# All videos go in a single flat Drive folder.

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
_CLASS_RE = re.compile(r"^([A-Za-z][A-Za-z_]*)(\d+)$")


def _parse_class(stem: str) -> str:
    """'Soap1' → 'Soap',  'dish_soap3' → 'dish_soap',  'unknown' → 'unknown'"""
    m = _CLASS_RE.match(stem)
    return m.group(1) if m else stem


def _frames_exist(out_dir: Path, stem: str) -> bool:
    """True if frames for this video were already extracted."""
    return out_dir.exists() and any(out_dir.glob(f"{stem}_*.png"))


class GdownBody(BaseModel):
    drive_url: str   # shared Google Drive folder (flat, all videos together)


@app.post("/upload/gdrive")
def upload_gdrive(body: GdownBody, x_api_key: str = Header(None)):
    _auth(x_api_key)
    s = ps.load()
    if not s.run_name:
        raise HTTPException(status_code=400, detail="No active run")

    run_images = RUNS_DIR / s.run_name / "images"
    log_dir = BASE_DIR / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"{s.run_name}_upload.log"

    snap = ps.load()
    snap.log_file = str(log_file)
    ps.save(snap)

    def _pull():
        log = logging.getLogger("stages")
        log.handlers.clear()
        fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(message)s"))
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter("%(levelname)-8s %(message)s"))
        log.addHandler(fh)
        log.addHandler(sh)
        log.setLevel(logging.INFO)

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            log.info("Downloading Drive folder: %s", body.drive_url)

            result = subprocess.run(
                ["gdown", "--folder", body.drive_url, "-O", str(tmp), "--quiet"],
                capture_output=True, text=True,
            )
            if result.returncode != 0:
                log.error("gdown failed:\n%s", result.stderr)
                _fail_upload(result.stderr)
                return

            # gdown places content inside a subfolder named after the Drive folder
            roots = [d for d in tmp.iterdir() if d.is_dir()]
            download_root = roots[0] if len(roots) == 1 else tmp

            all_videos = sorted([
                f for f in download_root.iterdir()
                if f.is_file() and f.suffix.lower() in VIDEO_EXTS
            ])

            if not all_videos:
                log.error("No video files found in Drive folder")
                return

            log.info("Found %d video(s) in Drive folder", len(all_videos))

            new_count = skipped_count = total_frames = 0

            for video in all_videos:
                class_name = _parse_class(video.stem)
                out_dir = run_images / class_name

                if _frames_exist(out_dir, video.stem):
                    log.info("  SKIP  %s  (already extracted)", video.name)
                    skipped_count += 1
                    continue

                out_dir.mkdir(parents=True, exist_ok=True)
                pattern = str(out_dir / f"{video.stem}_%05d.png")
                subprocess.run(
                    ["ffmpeg", "-i", str(video), "-vf", "fps=10", pattern, "-y"],
                    capture_output=True, text=True,
                )
                n = len(list(out_dir.glob(f"{video.stem}_*.png")))
                total_frames += n
                new_count += 1
                log.info("  NEW   %s  → class '%s' → %d frames", video.name, class_name, n)

            log.info(
                "Done: %d new (%d frames total), %d skipped (already on server)",
                new_count, total_frames, skipped_count,
            )

    import threading
    threading.Thread(target=_pull, daemon=True).start()
    return {"status": "downloading", "run": s.run_name}


def _fail_upload(error: str) -> None:
    s = ps.load()
    s.error = f"Upload failed: {error}"
    ps.save(s)


# ── Review ────────────────────────────────────────────────────────────────────

@app.get("/review/images")
def review_images(class_name: str, page: int = 0, page_size: int = 24):
    s = ps.load()
    if not s.run_name:
        raise HTTPException(status_code=400, detail="No active run")
    review_dir = RUNS_DIR / s.run_name / "cropped" / class_name
    if not review_dir.exists():
        return {"images": [], "total": 0, "class_name": class_name}

    all_images = sorted([
        f"/review/imgs/{s.run_name}/cropped/{class_name}/{f.name}"
        for f in review_dir.iterdir()
        if f.suffix.lower() in {".png", ".jpg", ".jpeg"}
    ])
    start = page * page_size
    return {
        "images": all_images[start: start + page_size],
        "total": len(all_images),
        "class_name": class_name,
        "page": page,
        "page_size": page_size,
    }


class DeleteBody(BaseModel):
    paths: list[str]


@app.post("/review/delete")
def review_delete(body: DeleteBody):
    s = ps.load()
    base = RUNS_DIR / s.run_name / "cropped"
    deleted = 0
    for raw in body.paths:
        parts = Path(raw).parts  # /review/imgs/<run>/<class>/<file>
        if len(parts) < 2:
            continue
        filename = parts[-1]
        class_name = parts[-2]
        target = _safe_child(str(base / class_name / filename), base)
        if target.exists():
            target.unlink()
            deleted += 1
    return {"deleted": deleted}


@app.post("/review/approve")
def review_approve(x_api_key: str = Header(None)):
    _auth(x_api_key)
    s = ps.load()
    s.review_done = True
    ps.save(s)
    return {"review_done": True}


# ── Inference ─────────────────────────────────────────────────────────────────

_infer_model = None
_infer_model_path: str = ""


@app.post("/infer")
async def infer(
    file: UploadFile = File(...),
    conf: float = 0.25,
    x_api_key: str = Header(None),
):
    _auth(x_api_key)
    s = ps.load()
    if not s.best_weights or not Path(s.best_weights).exists():
        raise HTTPException(status_code=404, detail="No trained model found for active run")

    global _infer_model, _infer_model_path
    if _infer_model_path != s.best_weights:
        from ultralytics import YOLO
        _infer_model = YOLO(s.best_weights)
        _infer_model_path = s.best_weights

    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    results = _infer_model.predict(img, conf=conf, verbose=False)
    annotated = results[0].plot()  # BGR numpy array

    _, buf = cv2.imencode(".jpg", annotated)
    return Response(content=buf.tobytes(), media_type="image/jpeg")


# ── Image serving — mount runs/ so all run images are accessible ──────────────

RUNS_DIR.mkdir(exist_ok=True)
app.mount("/review/imgs", StaticFiles(directory=str(RUNS_DIR)), name="imgs")
