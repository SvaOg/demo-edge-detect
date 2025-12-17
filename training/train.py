import os
import shutil
from pathlib import Path
from dotenv import load_dotenv
from ultralytics import YOLO  # type: ignore
from roboflow import Roboflow
import yaml


def main():
    print("Setting up environment...")

    # 1. Resolve paths
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent
    env_path = project_root / ".env"

    # Target directory: .../demo-edge-detect/data/dataset
    target_data_dir = project_root / "data" / "dataset"

    # 2. Load secrets
    if not env_path.exists():
        raise FileNotFoundError(f".env file not found at {env_path}")

    load_dotenv(env_path)

    api_key = os.getenv("ROBOFLOW_API_KEY")
    workspace = os.getenv("ROBOFLOW_WORKSPACE")
    project_name = os.getenv("ROBOFLOW_PROJECT")
    version_num = int(os.getenv("ROBOFLOW_VERSION", 1))

    if not all([api_key, workspace, project_name]):
        raise ValueError("Missing Roboflow configuration in .env file")

    print(f"Target data directory: {target_data_dir}")

    # 3. Download Dataset (The Robust Way)
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project_name)
    version = project.version(version_num)

    # Clean target directory if it exists to avoid conflicts
    if target_data_dir.exists():
        print("Cleaning existing data directory...")
        shutil.rmtree(target_data_dir)

    # Download to default location (usually inside 'training/') because it is reliable
    print("Downloading dataset...")
    dataset = version.download("yolov8")

    # The SDK returns an object where dataset.location is the path to the downloaded folder
    downloaded_path = Path(dataset.location)

    print(f"Moving data from {downloaded_path} to {target_data_dir}...")

    # Move the folder to our desired structure
    # We use shutil.move which handles cross-drive moves if necessary
    shutil.move(str(downloaded_path), str(target_data_dir))

    # Update our path variable to point to the new location
    final_dataset_path = target_data_dir
    print(f"Dataset successfully moved to: {final_dataset_path}")

    # 4. Fix Data Configuration (Windows Paths)
    # Now we work with the file in the NEW location
    yaml_path = final_dataset_path / "data.yaml"

    if not yaml_path.exists():
        raise FileNotFoundError(f"data.yaml not found at {yaml_path}")

    with open(yaml_path, "r") as f:
        data_config = yaml.safe_load(f)

    # Inject absolute paths
    data_config["path"] = str(final_dataset_path)

    val_path = final_dataset_path / "valid" / "images"
    if not val_path.exists():
        print("Warning: 'valid' folder not found. Using 'train' for validation.")
        data_config["val"] = "train/images"
    else:
        data_config["val"] = "valid/images"

    data_config["train"] = "train/images"

    if "test" in data_config:
        del data_config["test"]

    with open(yaml_path, "w") as f:
        yaml.dump(data_config, f)
    print("Config fixed: Absolute paths injected.")

    # 5. Training
    print("Initializing YOLOv8 Nano model...")
    model = YOLO("yolov8n.pt")

    print("Starting Training on RTX 5080...")
    results = model.train(
        data=str(yaml_path),
        epochs=100,
        imgsz=640,
        device=0,
        batch=16,
        name="raptor_run",
        exist_ok=True,
    )

    print("Training Complete!")

    best_weight_path = (
        script_dir / "runs" / "detect" / "raptor_run" / "weights" / "best.pt"
    )
    print(f"Best model saved at: {best_weight_path}")

    print("\nValidating model metrics...")
    # Validate using the model we just trained
    # Note: model.val() automatically uses the best weights from the run
    metrics = model.val()
    print(f"Map50-95: {metrics.box.map}")


if __name__ == "__main__":
    main()
