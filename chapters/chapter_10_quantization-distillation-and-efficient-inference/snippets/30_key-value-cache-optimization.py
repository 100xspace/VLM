Q = self.q_proj(hidden_states)
        K = self.k_proj(hidden_states)
        V = self.v_proj(hidden_states)
        Q = Q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        if use_cache:
            if self.kv_cache is not None:
                past_K, past_V = self.kv_cache
                K = torch.cat([past_K, K], dim=2)
                V = torch.cat([past_V, V], dim=2)
            self.kv_cache = (K, V)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_probs, V)

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, self.hidden_size)
        return self.out_proj(attn_output)
    def reset_cache(self):
        """Clear KV cache between sequences"""
        self.kv_cache = None
# Benchmark: with vs. without KV cache
def benchmark_kv_cache(model, input_ids, max_length=100):
    """Compare generation speed with and without KV caching"""
    import time
    # Without cache
    start = time.time()
    model.reset_cache()
    for i in range(max_length):
        _ = model(input_ids[:, :i+1], use_cache=False)
    t_no_cache = time.time() - start
    # With cache
    start = time.time()
    model.reset_cache()
    for i in range(max_length):
        _ = model(input_ids[:, i:i+1], use_cache=True)
    t_with_cache = time.time() - start
    print(f"Time without KV cache: {t_no_cache:.2f}s")
    print(f"Time with KV cache: {t_with_cache:.2f}s")
    print(f"Speedup: {t_no_cache / t_with_cache:.2f}x")
