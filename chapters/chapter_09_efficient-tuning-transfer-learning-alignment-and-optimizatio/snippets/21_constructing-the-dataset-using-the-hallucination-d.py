@torch.no_grad()
def generate_candidates(model, tokenizer, image, prompt, n=5):
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(
    **inputs,
    max_new_tokens=64,
    num_return_sequences=n,
    do_sample=True,
    temperature=1.0
)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)
