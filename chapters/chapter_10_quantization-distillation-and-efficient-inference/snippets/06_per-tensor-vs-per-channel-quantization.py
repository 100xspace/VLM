max_vals = torch.max(torch.abs(weight_tensor), dim=1 - dim, keepdim=True)[0]
