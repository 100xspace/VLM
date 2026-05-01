def build_dpo_triplets(image, prompt, ground_truth, hallucinations):
triplets = []
    for h in hallucinations:
    triplets.append({
            "image": image,
            "prompt": prompt,
            "chosen": ground_truth,
            "rejected": h
    })
    return triplets
