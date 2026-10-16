"""Fine-tune the three small YOLO detectors. Run: python train_yolo.py"""

import argparse
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO


ROOT = Path(__file__).parent
DATA = ROOT / "data/yolo"
WEIGHTS = ROOT / "weights"
CLASS_NAMES = {
    "maternal": ["weight_kg", "medications_taken"],
    "newborn": ["weight_g", "treatment"],
    "handwriting": ["handwriting"],
}


def absolute_dataset_file(name):
    """Ultralytics needs an absolute dataset path if the project is moved."""
    dataset = DATA / name
    description = {
        "path": str(dataset.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": dict(enumerate(CLASS_NAMES[name])),
    }
    destination = WEIGHTS / f"{name}_dataset.yaml"
    destination.write_text(yaml.safe_dump(description, sort_keys=False))
    return destination


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["all", *CLASS_NAMES], default="all")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for training this demo.")
    WEIGHTS.mkdir(exist_ok=True)

    selected = CLASS_NAMES if args.model == "all" else [args.model]
    for name in selected:
        print(f"\nTraining {name} detector on CUDA")
        # Ultralytics downloads the small public pretrained checkpoint on first use.
        model = YOLO("yolo11n.pt")
        results = model.train(
            data=str(absolute_dataset_file(name)),
            epochs=args.epochs,
            imgsz=960,       # High enough to preserve the small printed labels.
            batch=args.batch_size,
            device=0,
            workers=2,
            project=str(WEIGHTS / "training_runs"),
            name=name,
            exist_ok=True,
            verbose=False,
        )
        best = Path(results.save_dir) / "weights/best.pt"
        destination = WEIGHTS / f"{name}_yolo.pt"
        destination.write_bytes(best.read_bytes())
        print(f"saved {destination.name}")


if __name__ == "__main__":
    main()
