import torch
import sys

def check_cuda():
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    if cuda_available:
        print(f"CUDA version: {torch.version.cuda}")
        device_count = torch.cuda.device_count()
        print(f"GPU Count: {device_count}")
        current_device = torch.cuda.current_device()
        print(f"Current GPU: {torch.cuda.get_device_name(current_device)}")
    else:
        print("WARNING: CUDA is not available. Training will be slow on CPU.")

if __name__ == "__main__":
    check_cuda()