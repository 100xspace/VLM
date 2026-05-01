def calculate_pos_weight(dataframe):
    num_pos = dataframe['label'].sum()
    num_neg = len(dataframe) - num_pos
    pos_weight = num_neg / num_pos if num_pos > 0 else 1.0  # guard
    return torch.tensor(pos_weight, dtype=torch.float32)
# Usage
# weight = calculate_pos_weight(train_df)
# criterion = nn.BCEWithLogitsLoss(pos_weight=weight)
In practice, alternatives such as focal loss or weighted sampling can also be used to handle class imbalance, especially when minority cases are extremely rare.
