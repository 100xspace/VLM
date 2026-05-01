return -torch.mean(torch.log(torch.sigmoid(logits)))
