18. with torch.no_grad():
19.     features = model(input_tensor)
20. print(features.shape)  # Expected output: [1, 2048, 1, 1]
