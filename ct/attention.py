import torch
from torch import nn
import torch.nn.functional as F

#helpers
class ComplexLayerNorm(nn.Module):
    """ "
    Complex Layer Normalization where the normalization is computed as:
    Complex Layer Norm(x) = a_2 * (x - mean(x)) / (std(x) + eps) + b_2

    Architecture:
        - a_2: nn.Parameter
        - b_2: nn.Parameter
        - eps: float

    Args:
        dim (int): dimension of the input
        eps (float): epsilon value to avoid division by zero

    Usage
    >>> x = torch.randn(32, 512, 512)
    >>> complex_layer_norm = ComplexLayerNorm(512)
    >>> output = complex_layer_norm(x)
    >>> output.shape
    torch.Size([32, 512, 512])

    """

    def __init__(self, dim: int = None, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.a_2 = nn.Parameter(torch.ones(dim))
        self.b_2 = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        """
        Forward pass of the Complex Layer Norm
        """
        mean = x.mean(-1, keepdim=True)
        variance = ((x - mean) ** 2).mean(-1, keepdim=True)
        std = torch.sqrt(variance + self.eps)
        norm_x = (x - mean) / std
        return self.a_2 * norm_x + self.b_2

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization where the normalization is computed as:
    RMSNorm(x) = x / sqrt(mean(x^2))

    Architecture:
        - scale: float
        - g: nn.Parameter
    
    Args:
        dim (int): dimension of the input
    
    Usage
    >>> x = torch.randn(32, 512, 512)
    >>> rms_norm = RMSNorm(512)
    >>> output = rms_norm(x)
        
    """
    def __init__(
        self,
        dim: int = None,
    ):
        super().__init__()
        self.scale = dim ** 0.5
        self.g = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        """
        Forward pass of the RMSNorm
        where x is the input tensor
        """
        return F.normalize(
            x,
            dim=-1
        ) * self.scale * self.g
    

#main
class ComplexAttention(nn.Module):
    """
    Complex Attention where the attention is computed as:
    """

    def __init__(
        self,
        dim: int = None,
        heads: int = None,
        mask: bool = False,
        dropout: float = 0.0,
        qk_norm: bool = False,
    ):
        """
        __init__ method for the ComplexAttention class

        Args:
            dim (int): dimension of the input
            heads (int): number of heads for the attention
            qk_norm (bool): whether to normalize the query and key vectors

        """
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.mask = mask
        self.dropout = dropout
        self.qk_norm = qk_norm
        self.dim_head = dim // heads

        # seperate linear layers for real and imaginary parts
        self.query_real = nn.Linear(dim, dim)
        self.key_real = nn.Linear(dim, dim)
        self.value_real = nn.Linear(dim, dim)

        # seperate linear layers for imaginary parts
        self.query_imag = nn.Linear(dim, dim)
        self.key_imag = nn.Linear(dim, dim)
        self.value_imag = nn.Linear(dim, dim)

        # dropout
        self.dropout = nn.Dropout(dropout)

        # masking
        self.mask = mask

        # qk norm
        self.q_norm = nn.LayerNorm(dim)
        self.k_norm = nn.LayerNorm(dim)

        # out linear proj
        self.out = nn.Linear(dim, dim)

        # complex layer norm
        self.complex_norm = ComplexLayerNorm(dim)

    def forward(self, q, k, v):
        """
        Run forward pass on the complex attention module
        projections of the query, key, and value are computed -> dot product -> softmax
        -> mat mul with value -> concat -> output

        Args:
            q (torch.Tensor): query tensor
            k (torch.Tensor): key tensor
            v (torch.Tensor): value tensor

        Returns:
            output (torch.Tensor): output tensor

        """
        # seperate real and img parts
        q_real, q_imag = q.real, q.imag
        k_real, k_imag = k.real, k.imag
        v_real, v_imag = v.real, v.imag

        # apply linear layers
        q_real, q_imag = self.query_real(q_real), self.query_imag(q_imag)
        k_real, k_imag = self.key_real(k_real), self.key_imag(k_imag)
        v_real, v_imag = self.value_real(v_real), self.value_imag(v_imag)

        # apply qk norm
        if self.qk_norm:
            q_real, q_imag = self.q_norm(q_real), self.q_norm(q_imag)
            k_real, k_imag = self.k_norm(k_real), self.k_norm(k_imag)

        # apply dropout
        if self.dropout != 0.0:
            q_real, q_imag = self.dropout(q_real), self.dropout(q_imag)
            k_real, k_imag = self.dropout(k_real), self.dropout(k_imag)
            v_real, v_imag = self.dropout(v_real), self.dropout(v_imag)


        # compute dot product
        attn_weights = (
            # q_real @ k_real + q_imag @ k_imag
            q_real @ k_real.transpose(-2, 1)
            + q_imag @ k_imag.transpose(-2, -1)
        ) / self.dim_head**0.5

        # apply softmax
        attn_weights = F.softmax(attn_weights, dim=-1)

        # output, mat mul => attn_weights @ v_real + attn_weights @ v_imag
        output_real = attn_weights @ v_real
        output_imag = attn_weights @ v_imag

        # combine real and output parts
        output = output_real + 1j * output_imag

        # Apply Complex Layer Norm
        output = self.complex_norm(output)

        # # apply out linear layer
        # output = self.out(output)

        return output

# # Usage example
d_model = 512
nhead = 8
seq_len = 512
batch_size = 32

q = torch.randn(batch_size, seq_len, d_model) + 1j * torch.randn(batch_size, seq_len, d_model)
k = torch.randn(batch_size, seq_len, d_model) + 1j * torch.randn(batch_size, seq_len, d_model)
v = torch.randn(batch_size, seq_len, d_model) + 1j * torch.randn(batch_size, seq_len, d_model)

attention_layer = ComplexAttention(d_model, nhead, qk_norm=True)
attn_output = attention_layer(q, k, v)
print("Attention Output Shape:", attn_output.shape)

# ln_layer = ComplexLayerNorm(d_model)
# ln_output = ln_layer(attn_output)
# print("LayerNorm Output Shape:", ln_output.shape)
