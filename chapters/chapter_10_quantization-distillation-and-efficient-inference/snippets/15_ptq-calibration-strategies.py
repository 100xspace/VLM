def calibrate_minmax(activation_samples):
 """MinMax calibration: use observed min/max"""
 min_val = torch.min(activation_samples)
 max_val = torch.max(activation_samples)
 return min_val, max_val
def calibrate_percentile(activation_samples, percentile=99.99):
 """Percentile calibration: clip extreme outliers"""
 abs_max = torch.quantile(torch.abs(activation_samples), percentile / 100.0)
 return -abs_max, abs_max
def calibrate_entropy(activation_samples, num_bins=2048, num_quantized_bins=128):
 """
 Entropy calibration: minimize KL divergence.
