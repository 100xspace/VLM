def judge_eval(model, dataset, judge_model):
scores = []
    for sample in dataset:
    pred = model.generate(**sample["inputs"])
    score = llm_judge(judge_model, {
            "image_desc": sample["image_desc"],
            "question": sample["question"],
            "ground_truth": sample["ground_truth"],
            "prediction": pred
    })
    scores.append(score)
    return sum(scores) / len(scores)
