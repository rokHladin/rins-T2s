import os
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split

# === Settings ===
data_dir = "filtered_data"
batch_size = 64
num_epochs = 20
lr = 1e-4
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

print(f"📦 Using device: {device}")

# === Data transforms ===
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# === Load dataset ===
dataset = datasets.ImageFolder(data_dir, transform=train_transform)
num_classes = len(dataset.classes)
print(f"✅ Loaded {len(dataset)} images across {num_classes} classes")

# === Split train/val ===
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, val_size])
train_set.dataset.transform = train_transform
val_set.dataset.transform = val_transform

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size)

# === Model setup ===
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# === Training loop ===
for epoch in range(num_epochs):
    model.train()
    total_loss, correct = 0, 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs)
        loss = criterion(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (preds.argmax(1) == labels).sum().item()

    train_acc = correct / len(train_set)
    print(f"[Epoch {epoch+1}] 🔹 Train Loss: {total_loss:.2f}, Acc: {train_acc:.3f}")

    # === Validation ===
    model.eval()
    val_correct = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs)
            val_correct += (preds.argmax(1) == labels).sum().item()
    val_acc = val_correct / len(val_set)
    print(f"            🔸 Val Acc: {val_acc:.3f}")

# === Save model ===
torch.save(model.state_dict(), "bird_species_resnet18.pt")
print("✅ Model saved to bird_species_resnet18.pt")
