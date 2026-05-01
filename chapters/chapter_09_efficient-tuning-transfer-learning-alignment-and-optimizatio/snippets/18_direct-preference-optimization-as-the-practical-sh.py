return F.cross_entropy(
    logits.view(-1, logits.size(-1)),
    labels.view(-1),
    ignore_index=-100
)

Expected behavior: on a well-trained base, sft_loss should sit in the 0.5 to 2.0 range on domain data and fall steadily during the first epoch. A loss that plateaus near log(vocab_size) (roughly 10 to 11 for a 32k vocabulary) means the labels are not reaching the model, usually a tokenization or ignore_index mismatch rather than a modelling issue.
