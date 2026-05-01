def mmvet_eval(model, dataset):
correct = 0
    for sample in dataset:
    output = model.generate(**sample["inputs"])
        if sample["expected_answer"] in tokenizer.decode(output[0]):
        correct += 1
    return correct / len(dataset)

POPE: Explicitly probes object hallucination by asking about objects that are not present in the image.
def pope_eval(model, dataset):
hallucinations = 0
    for sample in dataset:
    output = model.generate(**sample["inputs"])
    text = tokenizer.decode(output[0], skip_special_tokens=True)
        if sample["false_object"] in text.lower():
        hallucinations += 1
    return 1 - hallucinations / len(dataset)

MathVista: Serves as the gold standard for chart reading and visual mathematical reasoning.
def mathvista_eval(pred, gt, tolerance=1e-2):
    try:
        return abs(float(pred) - float(gt)) < tolerance
    except:
        return False
