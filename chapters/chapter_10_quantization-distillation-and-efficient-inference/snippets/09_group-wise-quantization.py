num_groups = math.ceil(in_features / group_size)

    # Reshape into groups
    grouped = weight_tensor[:, :num_groups * group_size]\
              .reshape(out_features, num_groups, group_size)
    # Compute per-group scales
    max_vals = torch.max(torch.abs(grouped), dim=2, keepdim=True)[0]
