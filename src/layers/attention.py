import torch
import torch.nn as nn
import math

class CredalAttention(nn.Module):
    """
    實現論文 "Credal Transformer" 中描述的信任注意力機制 (Credal Attention Mechanism - CAM)。
    此版本加入了數值穩定性技巧以防止 exp() 溢出。
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.in_proj = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                key_padding_mask: torch.Tensor = None,
                attn_mask: torch.Tensor = None):
        """
        前向傳播。
        Args:
            query (Tensor): (L, N, E)
            key (Tensor): (S, N, E)
            value (Tensor): (S, N, E)
            key_padding_mask (ByteTensor, optional): (N, S)
            attn_mask (ByteTensor, optional): (L, S)
        Returns:
            Tuple[Tensor, Tensor, Tensor]: (attn_output, attn_weights, uncertainty)
        """
        tgt_len, bsz, embed_dim = query.shape
        src_len = key.shape[0]

        q, k, v = self.in_proj(query), self.in_proj(key), self.in_proj(value)
        
        q = q.reshape(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.reshape(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.reshape(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        attn_scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.head_dim)

        # 應用 attention mask
        if attn_mask is not None:
            attn_scores += attn_mask.unsqueeze(0)

        # 應用 padding mask, 將 padding 位置設為 -inf
        if key_padding_mask is not None:
            attn_scores = attn_scores.view(bsz, self.num_heads, tgt_len, src_len)
            attn_scores = attn_scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool), float("-inf"))
            attn_scores = attn_scores.view(bsz * self.num_heads, tgt_len, src_len)

        # 沿著 key 的維度減去最大值
        # 這會改變不確定性的絕對值，但保留了相對關係並防止溢出
        attn_scores_max, _ = torch.max(attn_scores, dim=-1, keepdim=True)
        # 處理完全被 mask 的情況，此時 max 為 -inf
        attn_scores_max = attn_scores_max.masked_fill(torch.isneginf(attn_scores_max), 0)
        stable_attn_scores = attn_scores - attn_scores_max

        evidence = torch.exp(stable_attn_scores)
        alpha = evidence + 1

        # 如果有 padding mask, 確保被 mask 掉的位置 alpha 值為 0
        if key_padding_mask is not None:
            alpha = alpha.view(bsz, self.num_heads, tgt_len, src_len)
            alpha = alpha.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool), 0)
            alpha = alpha.view(bsz * self.num_heads, tgt_len, src_len)
            
        total_evidence = torch.sum(alpha, dim=-1, keepdim=True)

        # 計算不確定性
        if key_padding_mask is not None:
            num_unmasked = (~key_padding_mask).sum(dim=1, dtype=torch.float32)
            num_unmasked = num_unmasked.view(bsz, 1, 1).expand(bsz, self.num_heads, tgt_len)
            num_unmasked = num_unmasked.reshape(bsz * self.num_heads, tgt_len, 1)
        else:
            num_unmasked = float(src_len)
            
        uncertainty = num_unmasked / (total_evidence + 1e-10)

        # 計算注意力權重 (Dirichlet 分佈的期望值)
        attn_weights = alpha / (total_evidence + 1e-10)
        attn_weights_dropped = self.dropout(attn_weights)

        # 計算注意力輸出
        attn_output = torch.bmm(attn_weights_dropped, v)
        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2).reshape(tgt_len, bsz, embed_dim)
        attn_output = self.out_proj(attn_output)

        # 整理輸出格式
        attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        uncertainty_reshaped = uncertainty.squeeze(-1).view(bsz, self.num_heads, tgt_len)
        avg_uncertainty = uncertainty_reshaped.mean(dim=1)

        return attn_output, attn_weights_reshaped, avg_uncertainty
