import torch

is_gpu = torch.cuda.is_available()
with open("gpu_available.txt", "w") as f:
    f.write(f"Torch sees GPU: {is_gpu}\n")
print(f"Torch sees GPU: {is_gpu}")
