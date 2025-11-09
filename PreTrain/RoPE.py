import torch
import torch.nn as nn


class RoPE(nn.Module):
    
    def __init__(self, dim, base=10000):
        super().__init__()
        assert dim%2 == 0 , "维度需要被2整除"
  
        i = torch.arange(0, dim, 2)
        theta = base ** (-i / dim)
        
        self.register_buffer('theta', theta)

    def forward(self, X):
        positions = torch.arange(0, X.size(1))
        angles = positions.unsqueeze(-1) * self.theta.unsqueeze(0)
        print(angles.shape)

        cos_val = torch.cos(angles).unsqueeze(0)
        print(cos_val.shape)
        sin_val = torch.sin(angles).unsqueeze(0)
        print(sin_val.shape)

        X_real = X[..., 0::2]
        X_imag = X[..., 1::2]

        X_rotate_real = X_real * cos_val - X_imag * sin_val
        X_rotate_imag = X_real * sin_val + X_imag * cos_val

        X_rotate = torch.stack([X_rotate_real, X_rotate_imag], dim=-1)
        X_rotate = X_rotate.flatten(-2)
        
        return X_rotate