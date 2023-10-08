[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

# Complex Transformer (WIP)
The open source implementation of the attention and transformer from "Building Blocks for a Complex-Valued Transformer Architecture" where they propose an an attention mechanism for complex valued signals or images such as MRI and remote sensing.

They present:
- complex valued scaled dot product attention
- complex valued layer normalization
- results show improved robustness to overfitting while maintaing performance wbhen compared to real valued transformer

## Install
`pip install complex-attn`

## Usage
```python
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

```

# Architecture
- I use regular norm instead of complex norm for simplicity

# License
MIT

# Citations
```
@article{2306.09827,
Author = {Florian Eilers and Xiaoyi Jiang},
Title = {Building Blocks for a Complex-Valued Transformer Architecture},
Year = {2023},
Eprint = {arXiv:2306.09827},
Doi = {10.1109/ICASSP49357.2023.10095349},
}
```
