import torch

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    print("ERROR: CUDA not available. Exiting.")
    exit(1)

device = torch.device("cuda")
print(f"Device: {torch.cuda.get_device_name(0)}")

# Matrix multiply on GPU
size = 2048
a = torch.randn(size, size, device=device)
b = torch.randn(size, size, device=device)

# Warmup
torch.matmul(a, b)
torch.cuda.synchronize()

import time
start = time.perf_counter()
c = torch.matmul(a, b)
torch.cuda.synchronize()
elapsed = time.perf_counter() - start

print(f"\nMatrix multiply {size}x{size}: {elapsed*1000:.1f} ms")
print(f"Result tensor device: {c.device}")
print(f"Result shape: {c.shape}")
print(f"Result sample value: {c[0, 0].item():.4f}")
print("\nGPU is working correctly.")
