import torch

# 创建一个torch.bool类型的Tensor
bool_tensor = torch.tensor([True], dtype=torch.bool)

# 转换为Python原生的bool类型
bool_value = bool_tensor.item()

print(type(bool_value))  # <class 'bool'>
print(bool_value)  # True 或 False
