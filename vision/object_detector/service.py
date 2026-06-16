"""FastAPI service — HTTP endpoints, log streaming, inference, file serving."""
import asyncio
import io
import json
import logging
import os
import re
import shutil
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
from PIL import Image, ImageOps
from pydantic import BaseModel

from pipeline_runner import PipelineRunner, register_thread_log, unregister_thread_log
import state as ps
from state import SEGMENT, GENERATE, TRAIN, BASE_DIR, RUNS_DIR
from repo import RepoManager

logging.basicConfig(level=logging.INFO)

API_KEY = os.environ.get("PIPELINE_API_KEY", "TESTING*/*1234567890")

app = FastAPI(title="Object Detector Pipeline")
runner = PipelineRunner()

templates = Jinja2Templates(directory=str(Path(__file__).parent / "review_app" / "templates"))
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "review_app" / "static")), name="static")


def _auth(key: str) -> None:
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


def _run(run_name: str | None) -> str:
    """Resolve and validate the target run for a request (X-Run header or ?run=)."""
    name = (run_name or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="No run specified")
    if not (RUNS_DIR / name).exists():
        raise HTTPException(status_code=404, detail=f"Run '{name}' not found")
    return name


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


@app.get("/help")
def help_page(request: Request):
    return templates.TemplateResponse(request, "help.html")


# ── Status & runs ─────────────────────────────────────────────────────────────

@app.get("/status")
def get_status(run: str):
    return asdict(ps.load(_run(run)))


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
    # Initialize the new run's state file
    ps.save(ps.PipelineState(run_name=name))
    return {"run_name": name}


@app.post("/runs/{name}/activate")
def activate_run(name: str, x_api_key: str = Header(None)):
    _auth(x_api_key)
    if not (RUNS_DIR / name).exists():
        raise HTTPException(status_code=404, detail=f"Run '{name}' not found")
    # The run's own state.json is the source of truth once it exists.
    if ps._state_file(name).exists():
        return asdict(ps.load(name))
    # No persisted state yet (legacy / externally-created run): seed it from the filesystem.
    s = ps.load(name)
    run_dir = RUNS_DIR / name
    cropped = run_dir / "cropped"
    s.segmented_classes = {
        d.name: True for d in cropped.iterdir() if d.is_dir() and any(d.iterdir())
    } if cropped.exists() else {}
    # Restore imported_classes from per-run sidecar, then merge into segmented_classes
    # so pointer-based (repo) classes are visible even without a local cropped/ dir
    sidecar = run_dir / "imported_classes.json"
    s.imported_classes = json.loads(sidecar.read_text()) if sidecar.exists() else {}
    for cls, info in s.imported_classes.items():
        if info.get("from_repo"):
            s.segmented_classes.setdefault(cls, True)
    s.segment_done  = bool(s.segmented_classes)
    s.generate_done = (run_dir / "dataset" / "data.yaml").exists()
    best_pt         = run_dir / "training" / "yolo" / "weights" / "best.pt"
    s.train_done    = best_pt.exists()
    s.best_weights  = str(best_pt) if best_pt.exists() else ""
    s.review_done   = s.segment_done  # assume reviewed if segmented (user can override)
    data_yaml = run_dir / "dataset" / "data.yaml"
    s.data_yaml = str(data_yaml) if data_yaml.exists() else ""
    ps.save(s)
    return asdict(s)


@app.post("/pipeline/reset")
def pipeline_reset(x_api_key: str = Header(None), x_run: str = Header(None)):
    _auth(x_api_key)
    s = ps.load(_run(x_run))
    s.running = s.error = ""
    ps.save(s)
    return {"running": ""}


# ── Global object repository ──────────────────────────────────────────────────

@app.get("/repo")
def repo_list():
    return {"entries": RepoManager().list_entries()}


@app.get("/repo/{label}")
def repo_label(label: str):
    rm = RepoManager()
    entries = rm.list_entries()
    label_key = rm._find_label_key(entries, label)
    if label_key is None:
        raise HTTPException(status_code=404, detail=f"Label '{label}' not found")
    return {"label": label_key, "identifiers": entries[label_key].get("identifiers", {})}


class PublishBody(BaseModel):
    label: str
    notes: str = ""


@app.post("/repo/publish")
def repo_publish(body: PublishBody, x_api_key: str = Header(None), x_run: str = Header(None)):
    _auth(x_api_key)
    s = ps.load(_run(x_run))
    identifier = f"{body.label}_{s.run_name}"
    source_dir = RUNS_DIR / s.run_name / "cropped" / body.label
    if not source_dir.exists():
        raise HTTPException(status_code=400, detail=f"Class '{body.label}' not found in active run's cropped directory")
    rm = RepoManager()
    try:
        count = rm.publish(body.label, identifier, source_dir,
                           source_run=s.run_name, notes=body.notes)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))

    # Remove local copy — canonical source is now the global repo
    shutil.rmtree(source_dir)

    # Update state: class is now a repo pointer, not a local copy
    repo_path = rm.entry_dir(body.label, identifier)
    s.imported_classes[body.label] = {
        "identifier": identifier,
        "from_repo": True,
        "repo_path": str(repo_path),
    }
    sidecar = RUNS_DIR / s.run_name / "imported_classes.json"
    tmp = sidecar.with_suffix(".tmp")
    tmp.write_text(json.dumps(s.imported_classes, indent=2))
    os.replace(tmp, sidecar)
    ps.save(s)

    return {"label": body.label, "identifier": identifier, "image_count": count}


class RepoImportBody(BaseModel):
    imports: list  # [{"label": "Soap", "identifier": "BlueBottle"}, ...]


@app.post("/repo/import")
def repo_import(body: RepoImportBody, x_api_key: str = Header(None), x_run: str = Header(None)):
    _auth(x_api_key)
    s = ps.load(_run(x_run))
    rm = RepoManager()
    entries = rm.list_entries()
    imported = []
    for entry in body.imports:
        label = entry.get("label", "").strip()
        identifier = entry.get("identifier", "").strip()
        if not label or not identifier:
            raise HTTPException(status_code=400, detail="Each import entry needs 'label' and 'identifier'")
        label_key = rm._find_label_key(entries, label)
        if label_key is None:
            raise HTTPException(status_code=404, detail=f"Label '{label}' not found in repository")
        if identifier not in entries[label_key].get("identifiers", {}):
            raise HTTPException(status_code=404, detail=f"Identifier '{identifier}' not found under '{label_key}'")
        repo_path = rm.entry_dir(label_key, identifier)
        if not repo_path.exists():
            raise HTTPException(status_code=404, detail=f"Repository directory missing: {repo_path}")
        count = entries[label_key]["identifiers"][identifier].get("image_count", 0)
        imported.append({"label": label_key, "identifier": identifier, "count": count})
        s.segmented_classes[label_key] = True
        s.imported_classes[label_key] = {
            "identifier": identifier,
            "from_repo": True,
            "repo_path": str(repo_path),
        }

    if s.segmented_classes and not s.segment_done:
        s.segment_done = True
    ps.save(s)

    sidecar = RUNS_DIR / s.run_name / "imported_classes.json"
    tmp = sidecar.with_suffix(".tmp")
    tmp.write_text(json.dumps(s.imported_classes, indent=2))
    os.replace(tmp, sidecar)

    return {"imported": imported}


# ── Log streaming ─────────────────────────────────────────────────────────────

@app.get("/logs/stream")
async def logs_stream(run: str):
    run_name = _run(run)
    async def generate():
        s = ps.load(run_name)
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
            current = ps.load(run_name)
            if not current.running:
                yield f"data: {'[ERROR] ' + current.error if current.error else '[DONE]'}\n\n"
                break
            await asyncio.sleep(0.3)

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/logs/file")
def logs_file(run: str):
    s = ps.load(_run(run))
    if not s.log_file or not Path(s.log_file).exists():
        return {"lines": []}
    return {"lines": Path(s.log_file).read_text(encoding="utf-8").splitlines()}


# ── Stage control ─────────────────────────────────────────────────────────────

@app.post("/stage/segment/run")
def stage_segment(x_api_key: str = Header(None), x_run: str = Header(None)):
    _auth(x_api_key)
    run = _run(x_run)
    try:
        runner.start_segment(run)
        return {"running": SEGMENT}
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


class GenerateBody(BaseModel):
    images_to_generate: int = 15000


@app.post("/stage/generate/run")
def stage_generate(body: GenerateBody, x_api_key: str = Header(None), x_run: str = Header(None)):
    _auth(x_api_key)
    run = _run(x_run)
    s = ps.load(run)
    if not s.review_done:
        raise HTTPException(status_code=409, detail="Review must be done first")
    try:
        runner.start_generate(run, body.images_to_generate)
        return {"running": GENERATE}
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


class TrainBody(BaseModel):
    device: str = "0"
    epochs: int = 100
    batch: int = 64


@app.post("/stage/train/run")
def stage_train(body: TrainBody, x_api_key: str = Header(None), x_run: str = Header(None)):
    _auth(x_api_key)
    run = _run(x_run)
    s = ps.load(run)
    if not s.generate_done:
        raise HTTPException(status_code=409, detail="Generate stage must be done first")
    try:
        runner.start_train(run, body.device, body.epochs, body.batch)
        return {"running": TRAIN}
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


# ── Video upload via gdown ────────────────────────────────────────────────────
# Video naming convention: <ClassName><Index>.<ext>
# Examples: Soap1.mp4, Soap2.mp4, Mug1.mp4, dish_soap1.mp4
# All videos go in a single flat Drive folder.

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
_CLASS_RE = re.compile(r"^([A-Za-z][A-Za-z_]*)(\d+)$")
# Videos with these (lowercased, number-stripped) names populate the shared
# backgrounds folder instead of being segmented as an object class.
BACKGROUND_NAMES = {"background", "backgrounds"}


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
def upload_gdrive(body: GdownBody, x_api_key: str = Header(None), x_run: str = Header(None)):
    _auth(x_api_key)
    run = _run(x_run)
    s = ps.load(run)

    run_images = RUNS_DIR / run / "images"
    log_dir = BASE_DIR / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"{run}_upload.log"
    log_file.write_text("", encoding="utf-8")

    s.log_file = str(log_file)
    ps.save(s)

    def _pull():
        register_thread_log(log_file)
        log = logging.getLogger("stages")
        try:
            _do_pull(log)
        finally:
            unregister_thread_log()

    def _do_pull(log):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            log.info("Downloading Drive folder: %s", body.drive_url)

            result = subprocess.run(
                ["gdown", "--folder", body.drive_url, "-O", str(tmp), "--quiet"],
                capture_output=True, text=True,
            )
            if result.returncode != 0:
                log.error("gdown failed:\n%s", result.stderr)
                _fail_upload(run, result.stderr)
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

            new_count = skipped_count = total_frames = bg_count = 0

            for video in all_videos:
                class_name = _parse_class(video.stem)
                is_background = class_name.lower() in BACKGROUND_NAMES
                out_dir = (BASE_DIR / "backgrounds") if is_background \
                          else (run_images / class_name)

                # Object classes skip if already extracted; background videos
                # always re-extract, replacing their existing frames.
                if not is_background and _frames_exist(out_dir, video.stem):
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
                if is_background:
                    bg_count += 1
                    log.info("  BG    %s  → backgrounds (replaced) → %d frames", video.name, n)
                else:
                    new_count += 1
                    log.info("  NEW   %s  → class '%s' → %d frames", video.name, class_name, n)

            log.info(
                "Done: %d new, %d background (%d frames total), %d skipped (already on server)",
                new_count, bg_count, total_frames, skipped_count,
            )

    import threading
    threading.Thread(target=_pull, daemon=True).start()
    return {"status": "downloading", "run": run}


def _fail_upload(run: str, error: str) -> None:
    s = ps.load(run)
    s.error = f"Upload failed: {error}"
    ps.save(s)


# ── Review ────────────────────────────────────────────────────────────────────

@app.get("/review/images")
def review_images(run: str, class_name: str, page: int = 0, page_size: int = 24):
    run_name = _run(run)
    review_dir = RUNS_DIR / run_name / "cropped" / class_name
    if not review_dir.exists():
        return {"images": [], "total": 0, "class_name": class_name}

    all_images = sorted([
        f"/review/imgs/{run_name}/cropped/{class_name}/{f.name}"
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
def review_delete(body: DeleteBody, x_run: str = Header(None)):
    base = RUNS_DIR / _run(x_run) / "cropped"
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


class RejectClassBody(BaseModel):
    class_name: str


@app.post("/review/class/reject")
def review_class_reject(body: RejectClassBody, x_run: str = Header(None)):
    run_name = _run(x_run)
    class_dir = _safe_child(str(RUNS_DIR / run_name / "cropped" / body.class_name),
                            RUNS_DIR / run_name / "cropped")
    if not class_dir.exists():
        return {"deleted": 0}
    deleted = sum(1 for f in class_dir.iterdir()
                  if f.is_file() and f.suffix.lower() in {".png", ".jpg", ".jpeg"}
                  and not f.unlink())
    return {"deleted": deleted}


@app.post("/review/approve")
def review_approve(x_api_key: str = Header(None), x_run: str = Header(None)):
    _auth(x_api_key)
    s = ps.load(_run(x_run))
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
    x_run: str = Header(None),
):
    _auth(x_api_key)
    s = ps.load(_run(x_run))
    if not s.best_weights or not Path(s.best_weights).exists():
        raise HTTPException(status_code=404, detail="No trained model found for active run")

    global _infer_model, _infer_model_path
    if _infer_model_path != s.best_weights:
        from ultralytics import YOLO
        _infer_model = YOLO(s.best_weights)
        _infer_model_path = s.best_weights

    img_bytes = await file.read()
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(img_bytes))).convert("RGB")

    results = _infer_model.predict(img, conf=conf, verbose=False)
    annotated = results[0].plot()  # BGR numpy array

    _, buf = cv2.imencode(".jpg", annotated)
    return Response(content=buf.tobytes(), media_type="image/jpeg")


# ── Image serving — mount runs/ so all run images are accessible ──────────────

RUNS_DIR.mkdir(exist_ok=True)
app.mount("/review/imgs", StaticFiles(directory=str(RUNS_DIR)), name="imgs")
