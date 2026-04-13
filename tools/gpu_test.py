import torch

for i in range(torch.cuda.device_count()):
    try:
        print(f"\nTesting GPU {i}")
        torch.cuda.set_device(i)
        x = torch.randn(5000, 5000, device=i)
        y = torch.matmul(x, x)
        torch.cuda.synchronize()
        print(f"GPU {i} OK")
    except Exception as e:
        print(f"GPU {i} FAILED: {e}")