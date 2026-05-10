"""
Training stage: YOLO model training.

Input:  {dataset_dir}/data.yaml
Output: {dataset_dir}/training_results/

CLI usage:
    python -m stages.train /path/to/data.yaml [--device 0] [--epochs 100] [--batch 64]
"""

import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


_MODELS_DIR = Path(__file__).parent.parent / "models"


def run(
    data_yaml: str,
    device: str = "0",
    epochs: int = 100,
    batch: int = 64,
    model: str = "yolo11m.pt",
) -> str:
    """Returns path to best weights file."""
    from ultralytics import YOLO

    data_yaml = Path(data_yaml)
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

    # data.yaml is at pipeline_runs/{name}/dataset/data.yaml
    # training output goes to pipeline_runs/{name}/training/yolo/ (mirrors notebook)
    run_dir = data_yaml.parent.parent
    project  = str(run_dir / "training")
    logger.info("Training YOLO on %s", data_yaml)
    logger.info("Device: %s | Epochs: %d | Batch: %d", device, epochs, batch)

    local_weights = _MODELS_DIR / model
    yolo = YOLO(str(local_weights) if local_weights.exists() else model)
    yolo.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=640,
        batch=batch,
        degrees=15,
        translate=0.1,
        shear=5,
        scale=0.75,
        perspective=0.001,
        flipud=0.3,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.2,
        copy_paste=0.1,
        patience=25,
        save_period=10,
        cache=True,
        amp=True,
        device=device,
        project=project,
        name="yolo",
    )

    best = Path(project) / "yolo" / "weights" / "best.pt"
    logger.info("Training complete. Best weights: %s", best)
    logger.info("Stage complete.")
    return str(best)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        stream=sys.stdout,
        force=True,
    )
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        print("Usage: python -m stages.train <data.yaml> [--device 0] [--epochs 100] [--batch 64]")
        sys.exit(1)

    def _arg(flag: str, default: str) -> str:
        try:
            return sys.argv[sys.argv.index(flag) + 1]
        except (ValueError, IndexError):
            return default

    run(
        data_yaml=args[0],
        device=_arg("--device", "0"),
        epochs=int(_arg("--epochs", "100")),
        batch=int(_arg("--batch", "64")),
    )
