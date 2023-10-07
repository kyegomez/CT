import torch
from ct.attention import ComplexAttention

# # Usage example
dim = 512
heads = 8
seq_len = 512
batch_size = 32

q = torch.randn(batch_size, seq_len, dim) + 1j * torch.randn(
    batch_size, seq_len, dim
)
k = torch.randn(batch_size, seq_len, dim) + 1j * torch.randn(
    batch_size, seq_len, dim
)
v = torch.randn(batch_size, seq_len, dim) + 1j * torch.randn(
    batch_size, seq_len, dim
)

attention_layer = ComplexAttention(
    dim,
    heads,
    qk_norm=True,
    dropout=0.1,
)
attn_output = attention_layer(q, k, v)
print("Attention Output Shape:", attn_output.shape)
