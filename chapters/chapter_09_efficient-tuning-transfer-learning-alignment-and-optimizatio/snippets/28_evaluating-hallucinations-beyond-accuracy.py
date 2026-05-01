def mmhal_eval(model, prompts):
score = 0
    for p in prompts:
    out = model.generate(**p)
        if "cannot determine" in tokenizer.decode(out[0]):
        score += 1
    return score / len(prompts)
