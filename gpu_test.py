import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.current_device()}")
print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
