# Project: Demo Edge Detect

**Objective:** Create an ML model training pipeline for custom object detection and run inference in Go (Edge emulation).
**Stack:** Python (Training), YOLOv8 (Model), ONNX (Exchange Format), Go (Inference).

## 🚀 Roadmap

### Phase 1: Data Preparation
- [ ] Select an object for detection (e.g., helmet, mug, specific tool).
- [ ] Take 30-50 photos of the object (different angles, backgrounds).
- [ ] Upload photos to Roboflow and label them.
- [ ] Export the dataset in "YOLOv8" format to the `data/dataset` folder.

### Phase 2: Training (Python)
- [ ] Set up venv and install `ultralytics`.
- [ ] Write `train.py`.
- [ ] Start training (Fine-tuning) for 50-100 epochs.
- [ ] Check training graphs (Loss should decrease).
- [ ] Test the model on new photos/videos.

### Phase 3: Edge Optimization
- [ ] Export the model to ONNX format (`export.py`).
- [ ] Ensure the ONNX file works (via Netron.app or a test script).

### Phase 4: Production Inference (Go)
- [ ] Initialize Go module.
- [ ] Connect the ONNX library (e.g., `github.com/yalue/onnxruntime_go` or OpenCV wrapper).
- [ ] Write model loading code.
- [ ] Write incoming image processing (resize, normalization).
- [ ] Get Bounding Box coordinates from the model in Go.

## 📚 Knowledge Base

### Key Concepts
*   **Epoch:** One full pass of the neural network through the entire dataset.
*   **Batch Size:** How many images the network "sees" at once before updating weights.
*   **Overfitting:** When the network simply memorizes your 50 photos but doesn't recognize the object in new ones. Solution: more augmentation (rotations, noise) or fewer epochs.
*   **ONNX (Open Neural Network Exchange):** "PDF for neural networks". Allows training in PyTorch and running in C++, Go, JS.

### Useful Commands

**Python Setup:**
```bash
cd training
uv run training.py
