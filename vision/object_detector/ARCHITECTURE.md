# Object Detector Pipeline — Architecture & Developer Reference

## Overview

A self-hosted web application that builds a custom YOLO object detector from raw video files.
It replaces a manual Jupyter notebook workflow with a persistent HTTP service that runs each pipeline stage as a background thread, streams logs in real time, and tracks multi-run state on disk.

---

## Architecture

### Components

```
object_detector/
├── service.py              # FastAPI app — all HTTP endpoints
├── pipeline_runner.py      # Thread management + logging setup for stages
├── state.py                # PipelineState dataclass + disk persistence
├── stages/
│   ├── segment.py          # GroundingDINO + SAM3 segmentation
│   ├── generate.py         # Synthetic dataset compositor
│   └── train.py            # YOLO training
├── review_app/
│   ├── templates/
│   │   ├── index.html      # Pipeline dashboard
│   │   ├── review.html     # Segmented image review UI
│   │   └── infer.html      # Live inference tester
│   └── static/
│       ├── auth.js         # Shared API key modal (localStorage)
│       ├── app.js          # Dashboard logic + SSE log consumer
│       ├── review.js       # Review grid, class navigation, delete/reject
│       └── infer.js        # Image upload + inference display
├── backgrounds/            # Shared background images for dataset generation
├── pipeline_runs/          # One subdirectory per named run (created at runtime)
│   └── {run_name}/
│       ├── images/         # Extracted video frames, one subdir per class
│       ├── cropped/        # Segmented BGRA PNGs cropped to object bbox
│       ├── dataset/        # Generated YOLO dataset + data.yaml
│       └── training/       # YOLO training output (best.pt inside)
├── pipeline_state.json     # Active run + stage completion flags (single file)
└── logs/                   # Per-stage log files (streamed live to UI)
```

### State management

`state.py` serializes a single `PipelineState` dataclass to `pipeline_state.json` on every write, using an atomic `os.replace` to prevent corruption. The state tracks the active run name, which stage (if any) is currently running, per-stage completion flags (`segment_done`, `review_done`, `generate_done`, `train_done`), and the path to the active log file.

When a run is activated, `activate_run` reconstructs completion flags from the filesystem (e.g., `cropped/` directory presence for `segment_done`, `data.yaml` existence for `generate_done`) so state survives service restarts.

### Stage execution

Each stage runs in a `daemon=True` background thread launched by `pipeline_runner`. The runner:
1. Opens a new log file at `logs/{run}_{stage}_{time}.log`
2. Sets the `stages` logger to write to it
3. Calls `state.transition(stage_name, ...)` to mark the stage as running
4. Starts the thread; thread calls `_finish()` or `_fail()` on completion

Only one stage can run at a time — all mutating endpoints check `s.running` and return HTTP 409 if busy.

### Log streaming

`GET /logs/stream` is a Server-Sent Events endpoint. It tails the active log file in a `while True` loop with 0.3 s sleep, stops when `state.running` becomes empty, and sends a final `[DONE]` or `[ERROR]` event. The frontend opens an `EventSource` as soon as a stage is triggered (and auto-reconnects on page load if a stage is already running).

---

## Pipeline Stages

### 1. Upload (`POST /upload/gdrive`)

- Downloads a public flat Google Drive folder using `gdown --folder`
- Videos must be named `<ClassName><N>.<ext>` (e.g. `Soap1.mp4`, `Mug2.mp4`)
- Runs `ffmpeg` at 10 fps to extract frames into `images/{ClassName}/`
- Already-extracted videos (detected by glob match) are skipped

### 2. Segment (`stages/segment.py`)

- Loads GroundingDINO (SwinT backbone) and SAM3 once, then iterates all class directories
- For each image: GroundingDINO finds bounding boxes for the class name as text prompt → SAM3 generates masks → best mask selected by score
- Mask is cleaned with morphological close+open, then the BGRA image is cropped to the tight bounding box and saved as a transparent PNG in `cropped/{ClassName}/`
- Classes with existing content in `cropped/` are skipped on re-runs
- Progress logged every 10 images as `[ClassName] NN% (n/total images, k saved)`

### 3. Review (`review_app/review.js` + endpoints)

Browser-based image gallery:
- `GET /review/images?class_name=&page=&page_size=` — paginated image list (48 per page)
- `POST /review/delete` — delete selected images by path (path traversal protected via `_safe_child`)
- `POST /review/class/reject` — delete all images for a class
- `POST /review/approve` — set `review_done = True` (required to unlock Generate)

UI provides per-class navigation with Accept (advance to next class, keep images) and Reject (delete all, advance) shortcuts.

### 4. Generate dataset (`stages/generate.py`)

Composites segmented objects on random backgrounds to create a synthetic YOLO segmentation dataset:
- Random number of objects per image (0–8); 10% chance of empty background
- Objects are randomly scaled (5–35% of background dimensions), slightly rotated (±5°), and augmented (brightness, contrast, saturation, hue, blur, noise, JPEG artifacts, polygon blobs)
- Overlap detection prevents objects from stacking excessively
- Labels written as YOLO polygon format (normalized pixel coordinates)
- After generation, images are split 80/10/10 train/valid/test
- Progress logged every 100 images as `N / total (NN%)`
- Output: `dataset/{train,valid,test}/{images,labels}/` + `dataset/data.yaml`

### 5. Train (`stages/train.py`)

Trains `yolo11m` (or a local checkpoint) via Ultralytics:
- Augmentation: mosaic, mixup, copy-paste, flip, rotation, perspective, scale
- Ultralytics logger is redirected to the stages log file so all output appears in the UI
- An `on_fit_epoch_end` callback logs per-epoch metrics: box loss, cls loss, mAP50
- Output: `training/yolo/weights/best.pt`

---

## Frontend & API

Three single-page views (`index.html`, `review.html`, `infer.html`), each with a matching `.js` file. All write endpoints require `x-api-key: <PIPELINE_API_KEY>` (set via env var).

**`auth.js`** — loaded first on every page. Stores the key in `localStorage` (`od_api_key`), shows a modal on first visit. Exposes `getApiKey()` and `resetApiKey()`.

---

## Relationship to `dataset_pipeline.ipynb`

The notebook was the original manual workflow; the service is its automated replacement. The directory layout (`images/`, `cropped/`, `dataset/`, `training/`) is shared, so both can operate on the same run directory.

Differences: the service enforces stage ordering, supports multiple named runs, and streams logs live instead of printing to cells.

---

## Cloudflare Tunnel

The service listens on `localhost:8000` inside Docker. To reach it from a phone or another network without opening firewall ports or configuring a VPN, use `cloudflared`.

### Quick tunnel (no account, temporary URL)

```bash
# Install once
curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 \
  -o cloudflared && chmod +x cloudflared && sudo mv cloudflared /usr/local/bin/

# Start tunnel — prints a *.trycloudflare.com URL
cloudflared tunnel --url http://localhost:8000
```

The URL is valid for the lifetime of the process. Restart `cloudflared` to get a new one.

### Persistent tunnel (named, survives restarts)

Requires a free Cloudflare account and a domain managed by Cloudflare.

```bash
# Authenticate (opens browser once)
cloudflared tunnel login

# Create a named tunnel
cloudflared tunnel create object-detector

# Route a subdomain to it — replace with your domain
cloudflared tunnel route dns object-detector detector.yourdomain.com

# Create config file at ~/.cloudflared/config.yml
cat > ~/.cloudflared/config.yml <<EOF
tunnel: object-detector
credentials-file: /home/$USER/.cloudflared/<tunnel-id>.json

ingress:
  - hostname: detector.yourdomain.com
    service: http://localhost:8000
  - service: http_status:404
EOF

# Run
cloudflared tunnel run object-detector
```

### Security

Cloudflare Tunnel is the recommended way to expose this service because:
- **No open ports** — `cloudflared` opens an outbound connection to Cloudflare's edge; the router/firewall never needs to be touched.
- **End-to-end TLS** — traffic between the client and Cloudflare's edge is encrypted via HTTPS automatically, even without a certificate on the host.
- **DDoS and bot protection** — requests pass through Cloudflare's network before reaching the machine.

The tunnel still exposes the service publicly, so the API key (`x-api-key` header, entered via the "API key" button on any page) is required for all write operations. Read-only endpoints (status, image browsing) are unauthenticated.

---

## Setup

### 1. Place model weights and sources

```
models/groundingdino_swint_ogc.pth
models/sam3.pt
models/GroundingDINO/   ← git clone https://github.com/IDEA-Research/GroundingDINO
models/sam3/            ← git clone <sam3 repo>
```

### 2. Add background images

Drop `.jpg`/`.png` files into `backgrounds/`. Kaggle indoor/kitchen scene datasets work well.

### 3. Set API key

```bash
echo "PIPELINE_API_KEY=your-secret-key" > .env
```

### 4. Build and start

```bash
# Web service only
docker compose -f pc.yaml build pipeline-service
docker compose -f pc.yaml up pipeline-service -d

# Both jupyter + web service
docker compose -f pc.yaml build
docker compose -f pc.yaml up -d

# Verify
docker compose -f pc.yaml logs -f pipeline-service
```

---