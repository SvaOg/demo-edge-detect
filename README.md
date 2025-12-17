# Demo Edge Detect

This project demonstrates a complete MLOps workflow for training a custom object detection model and deploying it for inference on an "edge" device, emulated using Go.

## 🎯 Objective

The goal is to create a robust pipeline for:
1.  Training a custom YOLOv8 object detection model using PyTorch.
2.  Exporting the trained model to the ONNX format for interoperability.
3.  Running inference with the ONNX model in a Go application.

## 🛠️ Tech Stack

*   **Training:** Python, PyTorch, `ultralytics` (YOLOv8)
*   **Model Exchange:** ONNX
*   **Inference:** Go
*   **Dataset Management:** Roboflow (recommended)

## 📂 Project Structure

```
.
├── data/              # Stores raw images and YOLOv8 datasets
├── training/          # Python scripts for training and exporting the model
├── models/            # Exported .pt and .onnx model files
├── inference/         # Go application for running inference
└── runs/              # Output from YOLOv8 training (weights, results)
```

## 🚀 Workflow

The project is divided into four main phases:

### 1. Data Preparation
- Collect and label images for the custom object you want to detect.
- Use a tool like Roboflow to manage annotations and export the dataset in the required "YOLOv8" format.
- Place the final dataset into the `data/dataset` directory.

### 2. Model Training
- Use the `train.py` script in the `training` directory to fine-tune a pre-trained YOLOv8 model on your custom dataset.
- Training progress, weights, and results will be saved in the `runs/detect/` directory.

### 3. Edge Optimization
- Export the final trained PyTorch model (`.pt`) to the ONNX format using the `export.py` script.
- The resulting `.onnx` file will be saved in the `models` directory. This format is optimized for cross-platform inference.

### 4. Production Inference
- The Go application in the `inference` directory loads the `.onnx` model.
- It includes code to preprocess input images, run them through the model, and process the bounding box coordinates returned by the model.

## 🏁 Getting Started

### Prerequisites
- Python 3.x
- Go 1.x
- `uv` (for Python environment management)

### Training Setup

1.  **Navigate to the training directory:**
    ```bash
    cd training
    ```

2.  **Create a virtual environment:**
    ```bash
    uv venv
    ```

3.  **Activate the environment:**
    *   **Windows:**
        ```powershell
        .venv\Scripts\activate
        ```
    *   **Linux/macOS:**
        ```bash
        source .venv/bin/activate
        ```

4.  **Install dependencies:**
    ```bash
    uv pip install ultralytics
    ```

Now you are ready to run the training and export scripts.
