from pathlib import Path
import argparse

import torch
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/segmentation_yolo/dataset.yaml"),
        help="Path to YOLO segmentation dataset YAML.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n-seg.pt",
        help="YOLO segmentation model checkpoint to start from.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs.",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Training image size.",
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=8,
        help="Training batch size. For an RTX 3080, start with 8.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="GPU device. Use '0' for first CUDA GPU or 'cpu' for CPU.",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="DataLoader workers. 0 is safer on Windows.",
    )

    parser.add_argument(
        "--project",
        type=Path,
        default=Path("outputs/segmentation_training"),
        help="Output folder for training runs.",
    )

    parser.add_argument(
        "--name",
        type=str,
        default="yolo_segmentation",
        help="Training run name.",
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=25,
        help="Early stopping patience.",
    )

    args = parser.parse_args()

    print("Segmentation training configuration:")
    print(f"Dataset YAML: {args.data}")
    print(f"Model: {args.model}")
    print(f"Epochs: {args.epochs}")
    print(f"Image size: {args.imgsz}")
    print(f"Batch size: {args.batch}")
    print(f"Device: {args.device}")
    print(f"Workers: {args.workers}")
    print(f"Project: {args.project}")
    print(f"Run name: {args.name}")

    print("\nCUDA check:")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = YOLO(args.model)

    print("\nStarting YOLO segmentation training...")

    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(args.project),
        name=args.name,
        task="segment",
        patience=args.patience,
        plots=True,
        save=True,
        verbose=True,
        exist_ok=True,
    )

    print("\nRunning validation on best/latest trained model...")

    model.val(
        data=str(args.data),
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        task="segment",
        plots=True,
    )

    print("\nSegmentation training complete.")
    print("Check this folder:")
    print(args.project / args.name)
    print("\nImportant files to inspect:")
    print(args.project / args.name / "weights" / "best.pt")
    print(args.project / args.name / "results.png")
    print(args.project / args.name / "confusion_matrix.png")
    print(args.project / args.name / "val_batch0_labels.jpg")
    print(args.project / args.name / "val_batch0_pred.jpg")


if __name__ == "__main__":
    main()