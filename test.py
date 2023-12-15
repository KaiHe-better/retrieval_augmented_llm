import torch

# 假设 tensor 是一个 PyTorch 张量
tensor = torch.tensor([[0, float('-inf'), 2], [float('-inf'), 5, float('-inf')]])

# 将 -inf 替换为 1e-9
tensor = torch.where(tensor == float('-inf'), torch.tensor(1e-9), tensor)

print(tensor)