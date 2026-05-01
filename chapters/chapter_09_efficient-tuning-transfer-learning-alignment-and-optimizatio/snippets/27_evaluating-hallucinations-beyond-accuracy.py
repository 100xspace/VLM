def pope_eval(model, dataset):
hallucinations = 0
    for sample in dataset:
    output = model.generate(**sample)
    text = tokenizer.decode(output[0], skip_special_tokens=True)
        if "yes" in text.lower():
        hallucinations += 1
    return 1 - hallucinations / len(dataset)
