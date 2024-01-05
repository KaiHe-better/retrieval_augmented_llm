import torch
from torch.nn import MultiheadAttention

# 随机生成输入数据
batch_size = 3
embed_size = 512
num_heads = 8

L = 5  # 目标序列长度
S = 6  # 源序列长度

# 生成随机输入
query = torch.rand(batch_size, L, embed_size)
key = torch.rand(batch_size, S, embed_size)
value = torch.rand(batch_size, S, embed_size)

# 随机生成attention masks
query_mask = torch.randint(0, 2, (batch_size, L)).bool()  # mask size: (batch_size, L)
key_mask = torch.randint(0, 2, (batch_size, S)).bool()   # mask size: (batch_size, S)

# 创建最终的attn_mask
# mask的维度需要是 (batch_size * num_heads, L, S)
# 使用batch-wise broadcasting
attn_mask = torch.einsum('bl,bs->bls', query_mask, key_mask)  # (batch_size, L, S)
attn_mask = attn_mask.repeat_interleave(num_heads, dim=0)  # (batch_size * num_heads, L, S)

# 创建MultiheadAttention实例
mha = MultiheadAttention(embed_dim=embed_size, num_heads=num_heads, batch_first=True)
print(attn_mask.shape)

# 应用MultiheadAttention
output, _ = mha(query, key, value, attn_mask=attn_mask)

print(output.shape)  # 查看输出形状
