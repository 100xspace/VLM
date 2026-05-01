def forward(self, x):
        if self.training:
            if self.symmetric:
                self.scale, self.zero_point = torch.max(torch.abs(x)) / self.qmax, torch.tensor(0.0)
            else:
                min_val, max_val = torch.min(x), torch.max(x)
                self.scale = (max_val - min_val) / (self.qmax - self.qmin)
                self.zero_point = self.qmin - min_val / self.scale
            x_quant = torch.clamp(torch.round(x / self.scale) + self.zero_point, self.qmin, self.qmax)
# During training: quantize → simulate integer representation → dequantize back to float for gradient flow
            return (x_quant - self.zero_point) * self.scale
        x_quant = torch.clamp(torch.round(x / self.scale) + self.zero_point, self.qmin, self.qmax).to(torch.int8)
        return x_quant

class QATLinear(torch.nn.Module):
    """Linear layer with quantization-aware training."""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features)
        self.weight_quantizer = FakeQuantize(num_bits=8, symmetric=True)
        self.activation_quantizer = FakeQuantize(num_bits=8, symmetric=False)

    def forward(self, x):
        x_quant = self.activation_quantizer(x); w_quant = self.weight_quantizer(self.linear.weight)
        return F.linear(x_quant, w_quant, self.linear.bias)

def train_qat_model(model, train_loader, num_epochs=3, device="cuda"):
    """Train a model with quantization-aware training."""
    model = model.to(device); model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images); loss = criterion(outputs, labels)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch} completed. Average loss: {avg_loss:.4f}")
    return model

class QATVisionModel(torch.nn.Module):
    """Vision model with QAT layers."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, 3, padding=1); self.relu1 = torch.nn.ReLU(); self.quant1 = FakeQuantize(num_bits=8)
        self.conv2 = torch.nn.Conv2d(64, 128, 3, padding=1); self.relu2 = torch.nn.ReLU(); self.quant2 = FakeQuantize(num_bits=8)
        self.pool = torch.nn.AdaptiveAvgPool2d((4, 4)); self.fc = QATLinear(128 * 4 * 4, num_classes)

    def forward(self, x):
        x = self.relu1(self.conv1(x)); x = self.quant1(x)
        x = self.relu2(self.conv2(x)); x = self.quant2(x)
        x = self.pool(x); x = x.flatten(1)
        return self.fc(x)

qat_model = QATVisionModel(num_classes=10)
print("QAT model initialized with fake quantization nodes")
