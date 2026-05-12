"""PipelineState dataclass — atomic JSON persistence and run directory helpers."""
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import json
import os


BASE_DIR   = Path(__file__).parent
RUNS_DIR   = BASE_DIR / "pipeline_runs"
STATE_FILE = BASE_DIR / "pipeline_state.json"

# Stage name constants
SEGMENT  = "segment"
GENERATE = "generate"
TRAIN    = "train"
REVIEW   = "review"


@dataclass
class PipelineState:
    run_name: str = ""
    running: str = ""               # name of the currently running stage, or ""
    error: str = ""
    log_file: str = ""
    started_at: str = ""
    stage_started_at: str = ""
    images_uploaded: int = 0
    data_yaml: str = ""
    best_weights: str = ""
    # Per-stage completion flags
    segment_done: bool = False
    review_done: bool = False
    generate_done: bool = False
    train_done: bool = False
    # Per-class segmentation tracking {class_name: bool}
    segmented_classes: dict = None

    def __post_init__(self):
        if self.segmented_classes is None:
            self.segmented_classes = {}

    def run_workdir(self) -> Path:
        if not self.run_name:
            raise ValueError("No active run. Create or select a run first.")
        return RUNS_DIR / self.run_name

    def bg_dir(self) -> Path:
        return BASE_DIR / "backgrounds"


def load() -> PipelineState:
    if not STATE_FILE.exists():
        return PipelineState()
    data = json.loads(STATE_FILE.read_text())
    fields = PipelineState.__dataclass_fields__
    return PipelineState(**{k: v for k, v in data.items() if k in fields})


def save(state: PipelineState) -> None:
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(asdict(state), indent=2))
    os.replace(tmp, STATE_FILE)


def transition(stage_running: str, **kwargs) -> PipelineState:
    s = load()
    s.running = stage_running
    s.stage_started_at = datetime.now().isoformat()
    s.error = ""
    for k, v in kwargs.items():
        setattr(s, k, v)
    save(s)
    return s


def list_runs() -> list[dict]:
    if not RUNS_DIR.exists():
        return []
    runs = []
    for d in sorted(RUNS_DIR.iterdir()):
        if not d.is_dir():
            continue
        classes = [c.name for c in (d / "cropped").iterdir() if c.is_dir()] \
                  if (d / "cropped").exists() else []
        runs.append({
            "name": d.name,
            "classes": classes,
            "has_model": (d / "training").exists(),
        })
    return runs
