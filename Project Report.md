# **Project Report: End-to-End Edge AI Pipeline**

## **1\. Executive Summary**

The goal of the "demo-edge-detect" project was to build a complete ML pipeline simulating a TurbineOne use case: rapid model retraining on limited data (Few-Shot Learning) and deployment to an edge environment using Go.

The project successfully demonstrated:

1. **Data Ops:** Custom dataset collection and automated preprocessing.  
2. **Transfer Learning:** Fine-tuning YOLOv8 on NVIDIA RTX 5080\.  
3. **Model portability:** Exporting weights to ONNX format.  
4. **System Integration:** Running hardware-accelerated inference in Go via C bindings.

## **2\. Technology Stack**

* **Training:** Python 3.13, PyTorch (Nightly), YOLOv8 (Ultralytics), Roboflow SDK.  
* **Inference:** Go 1.25, cgo, Microsoft ONNX Runtime (C-API).  
* **Hardware:** NVIDIA RTX 5080 (Blackwell Architecture).  
* **Tooling:** uv (Python dependency management), TDM-GCC (C Compiler).

## **3\. Architecture**

The pipeline is designed to be decoupled:

1. **Training Environment (Python):** Handles data download, path correction, GPU training, and ONNX export. It is isolated in a virtual environment.  
2. **Shared Storage:** The models/ directory acts as the interface between the training and inference stages.  
3. **Inference Environment (Go):** A standalone binary that loads the ONNX runtime library dynamically and executes the model logic.

## **4\. Key Technical Challenges & Solutions**

### **Challenge 1: Bleeding Edge Hardware Compatibility**

Problem: The workstation uses an NVIDIA RTX 5080 (Compute Capability sm\_120). The current stable release of PyTorch (2.6.0) and CUDA (12.4) does not support this architecture, leading to fallback on CPU or JIT compilation errors.  
Solution:

* Identified the architecture mismatch via system diagnostics.  
* Migrated the dependency stack to **PyTorch Nightly** builds.  
* Explicitly configured uv (pyproject.toml) to pull wheels from the specific PyTorch Nightly index with CUDA 12.8 support.

### **Challenge 2: Data Pipeline Robustness on Windows**

Problem: The YOLOv8 training engine relies on a YAML configuration file with relative paths. When integrating with the Roboflow SDK on Windows, the automatic download resulted in incorrect relative paths, causing dataset not found errors.  
Solution:

* Implemented a "Download & Move" pattern: data is downloaded to a temporary location and programmatically moved to a strict project structure (data/dataset).  
* Developed a Python script that parses the data.yaml post-download and injects **absolute paths** based on the current environment. This ensures the training script is portable across different drive letters and folder depths.

### **Challenge 3: CGO and Compiler Versioning**

Problem: To bridge Go and the ONNX Runtime DLL, cgo is required. The initial attempt using the latest MSYS2 GCC (v15.2.0) failed because the Go runtime build process treats compiler warnings as errors (-Werror), and the new GCC is stricter than previous versions.  
Solution:

* Diagnosed the build failure by analyzing verbose Go build logs (go run \-x).  
* Deployed **TDM-GCC (v10.3.0)** as a stable, proven compiler for the build environment.  
* Configured the CC environment variable to point Go specifically to this compiler instance.

### **Challenge 4: "DLL Hell" and Windows Library Loading**

Problem: The Go application crashed or failed to load the model even when the code seemed correct. Debugging revealed that SetSharedLibraryPath("onnxruntime.dll") was loading an incompatible, older version of the library found in the system PATH (System32 or other application folders), ignoring the local version.  
Solution:

* Analyzed Windows DLL search order behavior.  
* Refactored the Go initialization logic to resolve the **absolute path** of the local onnxruntime.dll before passing it to the runtime. This forces Windows to load the specific library version bundled with the project.

## **5\. Inference Logic Implementation**

The Go inference engine was built from scratch without high-level wrappers for image processing:

1. **Preprocessing:** Implemented image resizing (Catmull-Rom), normalization (0-255 to 0.0-1.0), and HWC to CHW memory layout conversion to match PyTorch tensor requirements.  
2. **Tensor Management:** Utilized onnxruntime\_go to manually manage memory allocation for input/output tensors.  
3. **Post-processing:** Implemented a parser for the raw YOLO output tensor \[1, 5, 8400\], filtering anchors by confidence score to identify the target object.

## **6\. Conclusion**

The project proves the viability of using Go for high-performance edge inference using models trained in Python. The resulting application is compiled, type-safe, and independent of the heavy Python runtime, making it suitable for deployment on constrained hardware.