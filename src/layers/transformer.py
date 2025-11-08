import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import CredalAttention

class CredalTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.self_attn = CredalAttention(d_model, nhead, dropout=dropout)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, src: torch.Tensor, src_mask: torch.Tensor = None, 
                src_key_padding_mask: torch.Tensor = None):
        
        attn_output, _, uncertainty = self.self_attn(
            src, src, src, 
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask
        )
        
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)
        
        ff_output = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(ff_output)
        src = self.norm2(src)
        
        # 返回該層的輸出和計算出的不確定性
        return src, uncertainty
