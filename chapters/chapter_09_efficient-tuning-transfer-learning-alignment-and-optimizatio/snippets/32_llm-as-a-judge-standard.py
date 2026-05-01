def llm_judge(judge_model, sample):
prompt = JUDGE_PROMPT.format(
    image_desc=sample["image_desc"],
    question=sample["question"],
    reference=sample["ground_truth"],
    prediction=sample["prediction"]
)
response = judge_model.generate(prompt)
    return float(response.strip())
