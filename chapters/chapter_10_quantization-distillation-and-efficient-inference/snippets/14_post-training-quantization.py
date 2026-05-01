images = images.to(device)
            _ = model(images)
    # Convert to quantized model
    torch.quantization.convert(model, inplace=True)
    return model
class SimpleVisionEncoder(torch.nn.Module):
    """Simplified vision encoder for demonstration"""
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(64, 128, 3, padding=1)
        self.pool = torch.nn.AdaptiveAvgPool2d((7, 7))
        self.fc = torch.nn.Linear(128 * 7 * 7, 512)
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x
# Initialize model
fp32_model = SimpleVisionEncoder()
fp32_model.eval()
# Dynamic PTQ (no calibration needed)
quantized_dynamic = apply_ptq_dynamic(fp32_model)
print(
    f"Original model size: "
    f"{sum(p.numel() * p.element_size() for p in fp32_model.parameters()) / 1e6:.2f} MB"
)
print(
    f"Quantized model size: "
    f"{sum(p.numel() * p.element_size() for p in quantized_dynamic.parameters()) / 1e6:.2f} MB"
)
