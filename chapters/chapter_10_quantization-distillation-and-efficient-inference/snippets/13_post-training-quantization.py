model.eval()
    model.qconfig = torch.quantization.get_default_qconfig("fbgemm")
    # Fuse layers (Conv+BN+ReLU) using the demo class module names
    torch.quantization.fuse_modules(model, [["conv1", "bn1", "relu"]], inplace=True)
    # Prepare for calibration
    torch.quantization.prepare(model, inplace=True)
    # Calibration pass: collect activation statistics
    print("Running calibration...")
    with torch.no_grad():
        for batch_idx, (images, texts) in enumerate(calibration_dataloader):
            if batch_idx >= 100:  # Calibrate on ~100 batches
