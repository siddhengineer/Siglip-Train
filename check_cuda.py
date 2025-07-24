import torch

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"Number of CUDA Devices: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    print(f"CUDA Device Name (0): {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is NOT available.")