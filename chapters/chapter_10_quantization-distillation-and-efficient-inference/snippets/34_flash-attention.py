import torch
import torch.nn as nn
import torch.nn.functional as F
# Flash Attention is typically used via libraries like xformers or flash-attn
# Here is how to integrate it into an attention block with a safe fallback
try:
    from flash_attn import flash_attn_func
    FLASH_ATTENTION_AVAILABLE = True
except ImportError:
    FLASH_ATTENTION_AVAILABLE = False
class FlashAttentionLayer(nn.Module):
    """Attention layer using Flash Attention for efficiency"""
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.hidden_size, self.num_heads = hidden_size, num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
    def forward(self, hidden_states):
        batch_size, seq_len, _ = hidden_states.shape
        # Project to Q, K, V
        qkv = self.qkv_proj(hidden_states).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        if FLASH_ATTENTION_AVAILABLE:
            # Use Flash Attention
            attn_output = flash_attn_func(q, k, v, causal=True)
        else:
            # Fallback to standard attention
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)  # [batch, heads, seq, dim]
            scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
            probs = F.softmax(scores, dim=-1)
            attn_output = torch.matmul(probs, v).transpose(1, 2)
        # Reshape and project
        attn_output = attn_output.reshape(batch_size, seq_len, self.hidden_size)
        return self.out_proj(attn_output)
