return quantize_dynamic(model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=dtype)
def apply_ptq_static(model, calibration_dataloader, device="cuda"):
    """
