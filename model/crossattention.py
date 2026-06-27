import torch
import torch.nn as nn


class CrossAttention(nn.Module):
    def __init__(self, embed_size, text_size):
        super(CrossAttention, self).__init__()
        self.scale = embed_size ** 0.5

        # Linear projections for cross-modal alignment
        self.q_net = nn.Linear(embed_size, embed_size)
        self.k_net = nn.Linear(text_size, embed_size)
        self.v_net = nn.Linear(text_size, embed_size)
        self.norm = nn.LayerNorm(embed_size)

    def forward(self, code_output, text_output, text_mask=None):
        # Generate Query from code, Key and Value from text context
        q = self.q_net(code_output)
        k = self.k_net(text_output)
        v = self.v_net(text_output)

        # Compute cross-modal attention scores
        scores = torch.matmul(q, k.transpose(-1, -2)) / self.scale

        # Apply mask to prevent attending to padding tokens
        if text_mask is not None:
            mask = text_mask.unsqueeze(1).to(scores.dtype)
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = torch.softmax(scores, dim=-1)

        # Weighted sum using text semantics
        context_vec = torch.matmul(attn_weights, v)

        # Residual connection over Query (code) and layer normalization
        return self.norm(context_vec + code_output)