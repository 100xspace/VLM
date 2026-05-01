inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=80)

response = processor.decode(output[0], skip_special_tokens=True)
print("LLaVA Response:", response)
