import torch.optim as optim

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()  # Set model to training mode (enables Dropout)
    running_loss = 0.0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # 1. Zero Gradients
        optimizer.zero_grad()

        # 2. Forward Pass
        outputs = model(inputs)  # Shape: [Batch_Size, 1]

        # Flatten outputs to match label shape
        outputs = outputs.view(-1)

        # 3. Calculate Loss
        loss = criterion(outputs, labels)

        # 4. Backward Pass (Backpropagation)
        loss.backward()

        # 5. Optimizer Step (Update Weights)
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    return running_loss / len(loader.dataset)

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Use the pos_weight calculated earlier
# pos_weight = torch.tensor([3.0]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# Adam is the standard optimizer for vision models
optimizer = optim.Adam(model.parameters(), lr=1e-4)
