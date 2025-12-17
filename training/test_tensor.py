import torch
import time

def smoke_test():
    print(f"CUDA Available: {torch.cuda.is_available()}")
    try:
        print("Attempting to move tensor to GPU...")
        # Создаем тензор
        x = torch.rand(1000, 1000)
        
        # Засекаем время (первый запуск может быть долгим из-за JIT-компиляции)
        start = time.time()
        x_gpu = x.to("cuda")
        torch.cuda.synchronize() # Ждем окончания операции
        end = time.time()
        
        print(f"Success! Tensor moved to GPU in {end - start:.4f} seconds.")
        print(f"Device of tensor: {x_gpu.device}")
        
        # Проверим вычисления (матричное умножение)
        print("Running matmul on GPU...")
        y_gpu = x_gpu @ x_gpu
        print("Calculation complete.")
        
    except Exception as e:
        print("\nCRITICAL FAILURE:")
        print(e)
        print("\nDiagnosis: The stable PyTorch build lacks PTX for forward compatibility.")

if __name__ == "__main__":
    smoke_test()
