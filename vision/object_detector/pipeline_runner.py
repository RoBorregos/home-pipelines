"""Launches pipeline stages as background threads and manages their log files."""
import logging
import sys
import threading
from datetime import datetime
from pathlib import Path

import state as ps
from state import SEGMENT, GENERATE, TRAIN, BASE_DIR


class PipelineRunner:
    """Orchestrates stage execution. Inject state_store/thread_factory for testing."""

    def __init__(self, state_store=None, thread_factory=None):
        self._ps = state_store or ps
        self._thread_factory = thread_factory or threading.Thread

    # ── Logging setup ─────────────────────────────────────────────────────────

    def _open_log(self, run_name: str, stage: str) -> Path:
        log_dir = BASE_DIR / "logs"
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / f"{run_name}_{stage}_{datetime.now().strftime('%H%M%S')}.log"

        root = logging.getLogger("stages")
        root.handlers.clear()
        root.setLevel(logging.INFO)

        fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(message)s"))
        root.addHandler(fh)

        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(logging.Formatter("%(levelname)-8s %(message)s"))
        root.addHandler(sh)

        return log_file

    # ── State transitions ─────────────────────────────────────────────────────

    def _finish(self, stage_done_flag: str, **extra) -> None:
        s = self._ps.load()
        s.running = ""
        s.error = ""
        setattr(s, f"{stage_done_flag}_done", True)
        for k, v in extra.items():
            setattr(s, k, v)
        self._ps.save(s)

    def _fail(self, error: str) -> None:
        s = self._ps.load()
        s.running = ""
        s.error = error
        self._ps.save(s)

    # ── Guards ────────────────────────────────────────────────────────────────

    def _assert_idle(self) -> ps.PipelineState:
        s = self._ps.load()
        if s.running:
            raise RuntimeError(f"Stage '{s.running}' is already running")
        return s

    def _assert_run(self, s: ps.PipelineState) -> None:
        if not s.run_name:
            raise RuntimeError("No active run selected")

    # ── Stage runners ─────────────────────────────────────────────────────────

    def _run_segment(self, run_workdir: str) -> None:
        from stages import segment
        try:
            segment.run(run_workdir)
            s = self._ps.load()
            cropped = Path(run_workdir) / "cropped"
            if cropped.exists():
                for d in cropped.iterdir():
                    if d.is_dir() and any(d.iterdir()):
                        s.segmented_classes[d.name] = True
            self._finish(SEGMENT, segmented_classes=s.segmented_classes)
        except Exception as exc:
            logging.getLogger("stages").exception("Segmentation failed")
            self._fail(str(exc))

    def _run_generate(self, run_workdir: str, bg_dir: str, images_to_generate: int) -> None:
        from stages import generate
        try:
            yaml_path = generate.run(
                workdir=run_workdir,
                bg_dir=bg_dir,
                output_folder=str(Path(run_workdir) / "dataset"),
                images_to_generate=images_to_generate,
            )
            self._finish(GENERATE, data_yaml=yaml_path)
        except Exception as exc:
            logging.getLogger("stages").exception("Generation failed")
            self._fail(str(exc))

    def _run_train(self, data_yaml: str, device: str, epochs: int, batch: int) -> None:
        from stages import train
        try:
            best = train.run(data_yaml=data_yaml, device=device, epochs=epochs, batch=batch)
            self._finish(TRAIN, best_weights=best)
        except Exception as exc:
            logging.getLogger("stages").exception("Training failed")
            self._fail(str(exc))

    # ── Public API ────────────────────────────────────────────────────────────

    def start_segment(self, class_name: str = "", image_count: int = 0) -> None:
        s = self._assert_idle()
        self._assert_run(s)
        run_workdir = str(s.run_workdir())
        log_file = self._open_log(s.run_name, SEGMENT)
        self._ps.transition(SEGMENT, log_file=str(log_file), images_uploaded=image_count,
                            started_at=datetime.now().isoformat(), segment_done=False,
                            review_done=False, generate_done=False, train_done=False)
        self._thread_factory(target=self._run_segment, args=(run_workdir,), daemon=True).start()

    def start_generate(self, images_to_generate: int = 15000) -> None:
        s = self._assert_idle()
        self._assert_run(s)
        log_file = self._open_log(s.run_name, GENERATE)
        self._ps.transition(GENERATE, log_file=str(log_file))
        self._thread_factory(
            target=self._run_generate,
            args=(str(s.run_workdir()), str(s.bg_dir()), images_to_generate),
            daemon=True,
        ).start()

    def start_train(self, device: str = "0", epochs: int = 100, batch: int = 64) -> None:
        s = self._assert_idle()
        self._assert_run(s)
        if not s.data_yaml:
            raise RuntimeError("No data.yaml — run generate first")
        log_file = self._open_log(s.run_name, TRAIN)
        self._ps.transition(TRAIN, log_file=str(log_file))
        self._thread_factory(
            target=self._run_train,
            args=(s.data_yaml, device, epochs, batch),
            daemon=True,
        ).start()


