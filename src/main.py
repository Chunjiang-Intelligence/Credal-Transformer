import torch
from layers.transformer import CredalTransformerEncoderLayer
from models.encoder import CredalTransformerEncoder

def demonstrate_credal_transformer():
    """演示 Credal Transformer Encoder 的用法和輸出。"""

    d_model = 512
    nhead = 8
    num_layers = 3
    dim_feedforward = 2048
    dropout = 0.1
    
    seq_len = 10
    batch_size = 4
    
    encoder_layer = CredalTransformerEncoderLayer(
        d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout
    )
    credal_encoder = CredalTransformerEncoder(encoder_layer, num_layers=num_layers)
    
    src = torch.randn(seq_len, batch_size, d_model)
    
    padding_mask = torch.tensor([
        [False, False, False, False, False, False, True,  True,  True,  True], # 6
        [False, False, False, False, False, False, False, False, False, False],# 10
        [False, False, False, False, True,  True,  True,  True,  True,  True], # 4
        [False, False, False, False, False, False, False, False, True,  True]  # 8
    ], dtype=torch.bool)
    
    print("--- 模型輸入 ---")
    print(f"Input Shape: {src.shape}")
    print(f"Padding Mask Shape: {padding_mask.shape}\n")
    
    output, uncertainties = credal_encoder(src, src_key_padding_mask=padding_mask)
    
    print("--- 模型輸出 ---")
    print(f"Final Output Shape: {output.shape}")
    print(f"Number of uncertainty tensors collected: {len(uncertainties)}")
    
    for i, u in enumerate(uncertainties):
        print(f"  - Uncertainty from Layer {i+1} Shape: {u.shape}")
        
    print("\n--- 不確定性分析 (第一層) ---")
    first_layer_uncertainty = uncertainties[0]
    
    print("\n樣本 1 (有效長度 6):")
    # 只打印未被 padding 的 token 的不確定性
    print(first_layer_uncertainty[0, :6].detach().numpy())
    
    print("\n樣本 3 (有效長度 4):")
    print(first_layer_uncertainty[2, :4].detach().numpy())


if __name__ == '__main__':
    demonstrate_credal_transformer()
