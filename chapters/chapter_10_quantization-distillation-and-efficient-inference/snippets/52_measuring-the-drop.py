from nltk.translate.bleu_score import corpus_bleu
    original_preds, quantized_preds, references = [], [], []
    for image, ground_truth in dataset:
        # Generate predictions
        orig_text = predict(original_model, image)
        quant_text = predict(quantized_model, image)
        original_preds.append(orig_text.split())
        quantized_preds.append(quant_text.split())
        references.append([ground_truth.split()])
    orig_score = corpus_bleu(references, original_preds)
    quant_score = corpus_bleu(references, quantized_preds)
    drop = (orig_score - quant_score) / orig_score * 100
    print(f"Original BLEU: {orig_score:.4f}")
    print(f"Quantized BLEU: {quant_score:.4f}")
    print(f"Accuracy Drop: {drop:.2f}%")
