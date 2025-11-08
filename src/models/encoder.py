import torch
import torch.nn as nn
from typing import List
from ..layers.transformer import CredalTransformerEncoderLayer

class CredalTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer: CredalTransformerEncoderLayer, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([
            CredalTransformerEncoderLayer(
                encoder_layer.self_attn.embed_dim, 
                encoder_layer.self_attn.num_heads,
                encoder_layer.linear1.out_features,
                encoder_layer.dropout.p
            ) for _ in range(num_layers)
        ])
        self.num_layers = num_layers

    def forward(self, src: torch.Tensor, mask: torch.Tensor = None, 
                src_key_padding_mask: torch.Tensor = None):
        output = src
        # 存儲每一層的不確定性
        uncertainties: List[torch.Tensor] = []

        for mod in self.layers:
            output, uncertainty = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            uncertainties.append(uncertainty)

        # 返回最後一層的輸出和所有層的不確定性列表
        return output, uncertainties
