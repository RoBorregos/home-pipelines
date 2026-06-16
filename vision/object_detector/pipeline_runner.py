"""Launches pipeline stages as background threads and manages their log files."""
import logging
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path

import state as ps
from state import SEGMENT, GENERATE, TRAIN, BASE_DIR


# ── Per-run, thread-routed logging ──────────────────────────────────────────────
# Each stage runs in its own thread. A single handler on the "stages" logger routes
# every record to the log file registered for the thread that emitted it, so
# concurrent runs never clobber each other's files (no handlers.clear() needed).
_RUN_LOG: dict[int, Path] = {}


def register_thread_log(log_file: Path) -> None:
    _RUN_LOG[threading.get_ident()] = Path(log_file)


def unregister_thread_log() -> None:
    _RUN_LOG.pop(threading.get_ident(), None)


class _ThreadRoutedHandler(logging.Handler):
    """Append each record to the log file registered for the current thread."""

    def emit(self, record: logging.LogRecord) -> None:
        log_file = _RUN_LOG.get(threading.get_ident())
        if log_file is None:
            return
        try:
            with open(log_file, "a", encoding="utf-8") as fh:
                fh.write(self.format(record) + "\n")
        except Exception:
            self.handleError(record)


def _install_log_handlers() -> None:
    root = logging.getLogger("stages")
    if getattr(root, "_routed", False):
        return
    root.setLevel(logging.INFO)
    root.propagate = False
    fh = _ThreadRoutedHandler()
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(message)s"))
    root.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(levelname)-8s %(message)s"))
    root.addHandler(sh)
    root._routed = True  # type: ignore[attr-defined]


_install_log_handlers()


class PipelineRunner:
    """Orchestrates stage execution. Inject state_store/thread_factory for testing."""

    def __init__(self, state_store=None, thread_factory=None):
        self._ps = state_store or ps
        self._thread_factory = thread_factory or threading.Thread

    # ── Logging setup ─────────────────────────────────────────────────────────

    def _open_log(self, run_name: str, stage: str) -> Path:
        """Create a fresh log file for this run+stage (handlers are global/thread-routed)."""
        log_dir = BASE_DIR / "logs"
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / f"{run_name}_{stage}_{datetime.now().strftime('%H%M%S')}.log"
        log_file.write_text("", encoding="utf-8")
        return log_file

    # ── State transitions ─────────────────────────────────────────────────────

    def _finish(self, run_name: str, stage_done_flag: str, **extra) -> None:
        s = self._ps.load(run_name)
        s.running = ""
        s.error = ""
        setattr(s, f"{stage_done_flag}_done", True)
        for k, v in extra.items():
            setattr(s, k, v)
        self._ps.save(s)

    def _fail(self, run_name: str, error: str) -> None:
        s = self._ps.load(run_name)
        s.running = ""
        s.error = error
        self._ps.save(s)

    # ── Guards ────────────────────────────────────────────────────────────────

    def _assert_idle(self, run_name: str) -> ps.PipelineState:
        if not run_name:
            raise RuntimeError("No run selected")
        s = self._ps.load(run_name)
        if s.running:
            raise RuntimeError(f"Stage '{s.running}' is already running for this run")
        return s

    # ── Stage runners ─────────────────────────────────────────────────────────

    def _run_segment(self, run_name: str, run_workdir: str, log_file: Path) -> None:
        register_thread_log(log_file)
        from stages import segment
        try:
            segment.run(run_workdir)
            s = self._ps.load(run_name)
            cropped = Path(run_workdir) / "cropped"
            if cropped.exists():
                for d in cropped.iterdir():
                    if d.is_dir() and any(d.iterdir()):
                        s.segmented_classes[d.name] = True
            self._finish(run_name, SEGMENT, segmented_classes=s.segmented_classes)
        except Exception as exc:
            logging.getLogger("stages").exception("Segmentation failed")
            self._fail(run_name, str(exc))
        finally:
            unregister_thread_log()

    def _run_generate(self, run_name: str, run_workdir: str, bg_dir: str,
                      images_to_generate: int, log_file: Path) -> None:
        register_thread_log(log_file)
        from stages import generate
        try:
            yaml_path = generate.run(
                workdir=run_workdir,
                bg_dir=bg_dir,
                output_folder=str(Path(run_workdir) / "dataset"),
                images_to_generate=images_to_generate,
            )
            self._finish(run_name, GENERATE, data_yaml=yaml_path)
        except Exception as exc:
            logging.getLogger("stages").exception("Generation failed")
            self._fail(run_name, str(exc))
        finally:
            unregister_thread_log()

    def _run_train(self, run_name: str, data_yaml: str, device: str,
                   epochs: int, batch: int, log_file: Path) -> None:
        # Run in a fresh subprocess
        register_thread_log(log_file)
        log = logging.getLogger("stages")
        try:
            proc = subprocess.Popen(
                [sys.executable, "-m", "stages.train", data_yaml,
                 "--device", device, "--epochs", str(epochs), "--batch", str(batch)],
                cwd=str(Path(__file__).parent),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
            for line in proc.stdout:
                log.info("%s", line.rstrip())
            if proc.wait() != 0:
                raise RuntimeError(f"training subprocess exited with code {proc.returncode}")
            best = Path(data_yaml).parent.parent / "training" / "yolo" / "weights" / "best.pt"
            self._finish(run_name, TRAIN, best_weights=str(best))
        except Exception as exc:
            log.exception("Training failed")
            self._fail(run_name, str(exc))
        finally:
            unregister_thread_log()

    # ── Public API ────────────────────────────────────────────────────────────

    def start_segment(self, run_name: str, class_name: str = "", image_count: int = 0) -> None:
        s = self._assert_idle(run_name)
        run_workdir = str(s.run_workdir())
        log_file = self._open_log(run_name, SEGMENT)
        self._ps.transition(run_name, SEGMENT, log_file=str(log_file), images_uploaded=image_count,
                            started_at=datetime.now().isoformat(), segment_done=False,
                            review_done=False, generate_done=False, train_done=False)
        self._thread_factory(target=self._run_segment,
                             args=(run_name, run_workdir, log_file), daemon=True).start()

    def start_generate(self, run_name: str, images_to_generate: int = 15000) -> None:
        s = self._assert_idle(run_name)
        log_file = self._open_log(run_name, GENERATE)
        self._ps.transition(run_name, GENERATE, log_file=str(log_file))
        self._thread_factory(
            target=self._run_generate,
            args=(run_name, str(s.run_workdir()), str(s.bg_dir()), images_to_generate, log_file),
            daemon=True,
        ).start()

    def start_train(self, run_name: str, device: str = "0", epochs: int = 100, batch: int = 64) -> None:
        s = self._assert_idle(run_name)
        if not s.data_yaml:
            raise RuntimeError("No data.yaml — run generate first")
        log_file = self._open_log(run_name, TRAIN)
        self._ps.transition(run_name, TRAIN, log_file=str(log_file))
        self._thread_factory(
            target=self._run_train,
            args=(run_name, s.data_yaml, device, epochs, batch, log_file),
            daemon=True,
        ).start()


