import torch
import os
from torch.utils.data import Dataset, DataLoader
torch.backends.cudnn.benchmark = True

class ADNI2p5DSingleSubjectDataset(Dataset):
    def __init__(self, split_dir):
        self.files = sorted([os.path.join(split_dir, f) for f in os.listdir(split_dir) if f.endswith(".pt")])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        slices_tensor, label_tensor = torch.load(self.files[idx])
        return slices_tensor, label_tensor

def collate_subject_slices(batch):
    all_slices = []
    all_labels = []
    for slices, label in batch:
        all_slices.append(slices)  # [num_slices, 3, H, W]
        all_labels.append(label.repeat(slices.size(0)))  # répéter label par slice
    all_slices = torch.cat(all_slices, dim=0)  # [total_slices, 3, H, W]
    all_labels = torch.cat(all_labels, dim=0)  # [total_slices]
    return all_slices, all_labels

train_loader = DataLoader(ADNI2p5DSingleSubjectDataset("slices_cache_filtered/train"),
                          batch_size=2, shuffle=True,
                          num_workers=4, pin_memory=True,
                          collate_fn=collate_subject_slices)

valid_loader = DataLoader(ADNI2p5DSingleSubjectDataset("slices_cache_filtered/valid"),
                          batch_size=2, shuffle=False,
                          num_workers=4, pin_memory=True,
                          collate_fn=collate_subject_slices)

test_loader = DataLoader(ADNI2p5DSingleSubjectDataset("slices_cache_filtered/test"),
                         batch_size=2, shuffle=False,
                         num_workers=4, pin_memory=True,
                         collate_fn=collate_subject_slices)


print(f"Train slices: {len(train_loader.dataset)}, Valid slices: {len(valid_loader.dataset)}, Test slices: {len(test_loader.dataset)}")


import matplotlib.pyplot as plt

# Charger un batch
batch_slices, batch_labels = next(iter(train_loader))

print(f"Batch slices shape: {batch_slices.shape}")  # [total_slices_in_batch, 3, H, W]
print(f"Batch labels shape: {batch_labels.shape}")

# Afficher quelques slices
num_to_show = 5
fig, axes = plt.subplots(num_to_show, 3, figsize=(10, num_to_show*3))

for i in range(num_to_show):
    slice_2p5d = batch_slices[i]  # [3, H, W]
    label = batch_labels[i].item()
    # Chaque "canal" correspond à une slice adjacente
    for j in range(3):
        axes[i, j].imshow(slice_2p5d[j].cpu(), cmap="gray")
        axes[i, j].axis("off")
        if j == 1:
            axes[i, j].set_title(f"Label: {label}")
plt.tight_layout()
plt.show()


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Modèle ---
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# --- Boucle d'entraînement ---
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    correct_train = 0
    total_train = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        preds = logits.argmax(dim=1)
        correct_train += (preds == y).sum().item()
        total_train += y.size(0)

    train_acc = correct_train / total_train
    avg_loss = total_loss / total_train

    # --- Validation ---
    model.eval()
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct_val += (pred == y).sum().item()
            total_val += y.size(0)

    val_acc = correct_val / total_val
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f} - Train Acc: {train_acc:.4f} - Val Acc: {val_acc:.4f}")

# --- Évaluation finale sur le test set ---
model.eval()
correct_test = 0
total_test = 0
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct_test += (pred == y).sum().item()
        total_test += y.size(0)

test_acc = correct_test / total_test
print(f"\n✅ Test Accuracy = {test_acc:.4f}")
