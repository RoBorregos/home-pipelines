"""PipelineState dataclass — atomic JSON persistence and run directory helpers."""
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import json
import os


BASE_DIR   = Path(__file__).parent
RUNS_DIR   = BASE_DIR / "pipeline_runs"

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
    # Classes imported from the global repo {class_name: {"identifier": str, "from_repo": True}}
    imported_classes: dict = None

    def __post_init__(self):
        if self.segmented_classes is None:
            self.segmented_classes = {}
        if self.imported_classes is None:
            self.imported_classes = {}

    def run_workdir(self) -> Path:
        if not self.run_name:
            raise ValueError("No active run. Create or select a run first.")
        return RUNS_DIR / self.run_name

    def bg_dir(self) -> Path:
        return BASE_DIR / "backgrounds"


def _state_file(run_name: str) -> Path:
    return RUNS_DIR / run_name / "state.json"


def load(run_name: str) -> PipelineState:
    """Load a run's state from its own state.json, or a fresh default if absent."""
    state_file = _state_file(run_name)
    if not state_file.exists():
        return PipelineState(run_name=run_name)
    data = json.loads(state_file.read_text())
    fields = PipelineState.__dataclass_fields__
    s = PipelineState(**{k: v for k, v in data.items() if k in fields})
    s.run_name = run_name  # path is the source of truth for identity
    return s


def save(state: PipelineState) -> None:
    state_file = _state_file(state.run_name)
    state_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = state_file.with_suffix(".tmp")
    tmp.write_text(json.dumps(asdict(state), indent=2))
    os.replace(tmp, state_file)


def transition(run_name: str, stage_running: str, **kwargs) -> PipelineState:
    s = load(run_name)
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
        if not d.is_dir() or d.name.startswith("_"):
            continue
        classes = [c.name for c in (d / "cropped").iterdir() if c.is_dir()] \
                  if (d / "cropped").exists() else []
        sidecar = d / "imported_classes.json"
        if sidecar.exists():
            try:
                imported = json.loads(sidecar.read_text())
                for cls in imported:
                    if cls not in classes:
                        classes.append(cls)
            except Exception:
                pass
        runs.append({
            "name": d.name,
            "classes": classes,
            "has_model": (d / "training").exists(),
        })
    return runs
