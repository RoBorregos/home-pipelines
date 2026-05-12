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

### How the layers fit together

The backend has three layers with clear responsibilities:

**`service.py` — API layer.** Receives HTTP requests, validates the API key, and delegates work. It never runs a stage directly — it just calls `pipeline_runner` and returns immediately. Also owns the SSE log stream endpoint and static file serving.

**`pipeline_runner.py` — Orchestrator.** The only place that launches threads. When a stage is triggered it opens a dedicated log file, configures the `stages` logger to write to it, marks the stage as running in state, then starts the thread. When the thread finishes it calls `_finish()` or `_fail()` to update state. Only one stage can run at a time — if `state.running` is not empty, the runner rejects the call with an error.

**`stages/` — Workers.** Pure functions that do the actual ML work. Each has a single `run()` entry point, logs progress via the `stages` logger (already wired by the runner), and knows nothing about HTTP or threads.

```
Browser → service.py → pipeline_runner.py → Thread → stages/segment.py
                ↕                   ↕
           HTTP response       state.py (pipeline_state.json)
```

**`state.py` — Shared memory.** A single JSON file (`pipeline_state.json`) that all layers read and write. Writes are atomic (`os.replace`) to prevent corruption. Stores the active run, which stage is running, per-stage completion flags, and the current log file path. When the service restarts, `activate_run` rebuilds the flags by inspecting the filesystem directly, so no state is lost.

**Log streaming.** `GET /logs/stream` is a Server-Sent Events endpoint that tails the active log file every 0.3 s and pushes each new line to the browser. It stops when `state.running` clears and sends a final `[DONE]` or `[ERROR]`. The browser opens this connection the moment a stage starts and auto-reconnects if the page reloads mid-run.

---

## Pipeline Stages

### 1. Upload (`POST /upload/gdrive`)

Downloads a public flat Google Drive folder with `gdown`, then extracts frames with `ffmpeg` at 10 fps into `images/{ClassName}/`. Videos must be named `<ClassName><N>.<ext>` (e.g. `Soap1.mp4`, `Mug2.mp4`). Already-extracted videos are skipped.

### 2. Segment (`stages/segment.py`)

Loads GroundingDINO + SAM3 once, then for each image: GroundingDINO detects bounding boxes using the class name as a text prompt → SAM3 generates a mask → best mask is selected by score → mask is morphologically cleaned and the object is cropped to its tight bounding box and saved as a transparent PNG in `cropped/{ClassName}/`. Classes already present in `cropped/` are skipped. Progress logged every 10 images.

### 3. Review (`review_app/review.js` + endpoints)

Browser gallery for inspecting and deleting bad segmented images per class. Supports per-image delete, bulk delete, Accept class (keep all, move to next), and Reject class (delete all, move to next). Marking as reviewed unlocks the Generate stage.

### 4. Generate dataset (`stages/generate.py`)

Composites segmented objects onto random backgrounds to produce a synthetic YOLO dataset. Each image gets 1–8 randomly placed objects (10% chance of empty background), augmented with brightness, contrast, blur, noise, and JPEG artifacts. Labels are written in YOLO polygon format. Output is split 80/10/10 into `dataset/{train,valid,test}/` with a `data.yaml`. Progress logged every 100 images.

### 5. Train (`stages/train.py`)

Trains `yolo11m` via Ultralytics on the generated dataset. Ultralytics log output is redirected to the stages log file so it appears live in the UI. An `on_fit_epoch_end` callback logs box loss, cls loss, and mAP50 per epoch. Output: `training/yolo/weights/best.pt`.

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