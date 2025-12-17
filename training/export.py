from ultralytics import YOLO  # type: ignore
from pathlib import Path
import shutil


def main():
    # 1. Setup Paths
    # Script location: .../demo-edge-detect/training
    script_dir = Path(__file__).parent.resolve()

    # Project root: .../demo-edge-detect
    project_root = script_dir.parent

    # Run location: .../demo-edge-detect/runs (как ты и подтвердил)
    source_weights = (
        project_root / "runs" / "detect" / "raptor_run" / "weights" / "best.pt"
    )

    if not source_weights.exists():
        print(f"Error: Model file not found at {source_weights}")
        return

    print(f"Loading model from: {source_weights}")
    model = YOLO(source_weights)

    # 2. Export to ONNX
    print("Starting ONNX export...")
    # Export happens next to the .pt file
    model.export(format="onnx", imgsz=640)

    # Expect: .../runs/detect/raptor_run/weights/best.onnx
    source_onnx = source_weights.with_suffix(".onnx")

    # 3. Move to Models Directory
    dest_dir = project_root / "models"
    dest_dir.mkdir(parents=True, exist_ok=True)

    dest_file = dest_dir / "model.onnx"

    if source_onnx.exists():
        print(f"Moving {source_onnx} to {dest_file}...")
        shutil.move(source_onnx, dest_file)
        print("Success! Model is ready for Go inference.")
    else:
        print(f"Error: Export finished but {source_onnx} was not found.")


if __name__ == "__main__":
    main()
