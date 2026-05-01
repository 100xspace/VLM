abs_activations = torch.abs(activation_samples)
 max_val = torch.max(abs_activations)
    hist, bin_edges = torch.histogram(abs_activations, bins=num_bins, range=(0, max_val))
    best_threshold = max_val
 min_kl_div = float('inf')
    # Search for optimal threshold
 for i in range(num_quantized_bins, num_bins):
     threshold = bin_edges[i]
        # Quantize and measure KL divergence
     sliced_hist = hist[:i].clone()
     sliced_hist[-1] += hist[i:].sum()  # Add clipped values to last bin
      # Compute KL divergence (simplified)
     p = hist / hist.sum()
     q = sliced_hist / sliced_hist.sum()
     # Pad q to match p's length
     q_padded = torch.zeros_like(p)
     q_padded[:len(q)] = q
     kl_div = torch.sum(p * torch.log((p + 1e-10) / (q_padded + 1e-10)))
     if kl_div < min_kl_div:
         min_kl_div = kl_div
         best_threshold = threshold
  return -best_threshold, best_threshold

# Example: Compare calibration methods
sample_activations = torch.randn(10000) * 2.0
sample_activations[::1000] = 10.0  # Add outliers
minmax_range = calibrate_minmax(sample_activations)
percentile_range = calibrate_percentile(sample_activations, percentile=99.9)
entropy_range = calibrate_entropy(sample_activations)
print(f"MinMax range: [{minmax_range[0]:.2f}, {minmax_range[1]:.2f}]")
print(f"Percentile range: [{percentile_range[0]:.2f}, {percentile_range[1]:.2f}]")
print(f"Entropy range: [{entropy_range[0]:.2f}, {entropy_range[1]:.2f}]")
