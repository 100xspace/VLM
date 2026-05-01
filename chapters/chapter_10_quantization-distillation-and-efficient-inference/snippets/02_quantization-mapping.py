if symmetric:
        # Symmetric quantization: range is [-max, max]
        max_val = torch.max(torch.abs(tensor))
