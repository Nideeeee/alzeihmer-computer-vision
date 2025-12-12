import torch
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import balanced_accuracy_score, accuracy_score

torch.backends.cudnn.benchmark = True
plt.style.use('ggplot') 

# ==========================================
# 1. DATASET & DATALOADERS
# ==========================================
class ADNI2p5DSingleSubjectDataset(Dataset):
    def __init__(self, split_dir):
        self.files = sorted([os.path.join(split_dir, f) for f in os.listdir(split_dir) if f.endswith(".pt")])
    def __len__(self): return len(self.files)
    def __getitem__(self, idx): return torch.load(self.files[idx])

def collate_subject_slices(batch):
    all_slices = []
    all_labels = []
    for slices, label in batch:
        all_slices.append(slices) 
        all_labels.append(label.repeat(slices.size(0)))
    return torch.cat(all_slices, dim=0), torch.cat(all_labels, dim=0)

train_dir = "slices_cache_filtered/mci/train" 
valid_dir = "slices_cache_filtered/mci/valid"
test_dir  = "slices_cache_filtered/mci/test"

train_loader = DataLoader(ADNI2p5DSingleSubjectDataset(train_dir), batch_size=4, shuffle=True, 
                          num_workers=4, pin_memory=True, collate_fn=collate_subject_slices)
valid_loader = DataLoader(ADNI2p5DSingleSubjectDataset(valid_dir), batch_size=1, shuffle=False, num_workers=4)
test_loader  = DataLoader(ADNI2p5DSingleSubjectDataset(test_dir), batch_size=1, shuffle=False, num_workers=4)

print(f"Train: {len(train_loader.dataset)}, Valid: {len(valid_loader.dataset)}, Test: {len(test_loader.dataset)}")

# ==========================================
# 2. MODÈLE (ResNet Partial Fine-Tuning)
# ==========================================
class ResNetWithEmbeddings(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNetWithEmbeddings, self).__init__()
        base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(base_model.children())[:-1])
        self.fc = nn.Linear(512, num_classes)

        # Layers 1-2 Gelés / Layers 3-4 + FC Entraînables
        for param in self.features.parameters(): param.requires_grad = False
        for name, param in self.features.named_parameters():
            if "layer3" in name or "layer4" in name: param.requires_grad = True

    def forward(self, x):
        if x.max() > 10.0: x = x / 255.0
        x = self.features(x)
        x = x.view(x.size(0), -1) 
        return self.fc(x), x

device = "cuda" if torch.cuda.is_available() else "cpu"
model = ResNetWithEmbeddings(num_classes=2).to(device)

# ==========================================
# 3. SETUP ENTRAÎNEMENT
# ==========================================
def compute_class_weights(loader):
    labels = [loader.dataset[i][1].item() for i in range(len(loader.dataset))]
    c0, c1 = labels.count(0), labels.count(1)
    if c0 == 0 or c1 == 0: return torch.tensor([1.0, 1.0]).to(device)
    w = torch.tensor([len(labels)/(2*c0), len(labels)/(2*c1)], dtype=torch.float32).to(device)
    print(f"Poids Classes: CN/Stable={w[0]:.2f}, AD/Conv={w[1]:.2f}")
    return w

class_weights = compute_class_weights(train_loader)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

# ==========================================
# 4. FONCTION EVALUATION
# ==========================================
def evaluate(model, loader, device, criterion):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    
    criterion_val = nn.CrossEntropyLoss() 

    with torch.no_grad():
        for slices, label in loader:
            slices, label = slices[0].to(device), label.to(device)
            logits, _ = model(slices) 
            
            loss = criterion_val(logits, label.repeat(slices.size(0)))
            total_loss += loss.item()

            preds = logits.argmax(dim=1)
            most_common = Counter(preds.cpu().numpy()).most_common(1)[0][0]
            all_preds.append(most_common)
            all_labels.append(label.item())
    
    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    return avg_loss, acc, bal_acc

# ==========================================
# 5. BOUCLE PRINCIPALE
# ==========================================
num_epochs = 30
train_losses, val_losses = [], []
train_accs, val_accs = [], []
best_bal_acc = 0.0
save_path = "best_model.pth"

print(f"\n--- Start Training ({num_epochs} Epochs) ---")

for epoch in range(num_epochs):
    model.train()
    ep_loss, correct, total = 0.0, 0, 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits, _ = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        ep_loss += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)

    # Metrics Train
    train_loss = ep_loss / total
    train_acc = correct / total
    
    # Metrics Valid
    val_loss, val_acc, val_bal = evaluate(model, valid_loader, device, criterion)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    # Affichage avec Balanced Acc dans le terminal (pour info)
    print(f"Epoch {epoch+1:02d} | Loss: {train_loss:.4f} / {val_loss:.4f} | Acc: {train_acc:.4f} / {val_acc:.4f} | Bal: {val_bal:.4f}")

    # Sauvegarde basée sur Balanced Acc (cachée des plots mais utilisée pour save)
    if val_bal > best_bal_acc:
        best_bal_acc = val_bal
        torch.save(model.state_dict(), save_path)
        print("  ★ New Best Model Saved (Best Bal Acc)!")

# ==========================================
# 6. COURBES (ZOOM AUTOMATIQUE)
# ==========================================
def plot_curves():
    plt.figure(figsize=(12, 6))
    
    # 1. Loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train', color='blue')
    plt.plot(val_losses, label='Valid', color='red')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 2. Accuracy Standard (Zoom Auto)
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train (Slice)', color='green')
    plt.plot(val_accs, label='Valid (Patient)', color='orange')
    plt.axhline(y=0.5, color='gray', linestyle='--', label='Hasard (0.5)')
    # plt.ylim(0, 1.0) <--- LIGNE SUPPRIMÉE POUR LAISSER LE ZOOM AUTO
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_curves_autozoom.png', dpi=300)
    print("\nGraphiques sauvegardés : training_curves_autozoom.png")

plot_curves()

# === Test Final ===
print("\n--- Testing ---")
model.load_state_dict(torch.load(save_path))
test_loss, test_acc, test_bal = evaluate(model, test_loader, device, criterion)
print(f"Test Acc: {test_acc:.4f} | Test Bal Acc: {test_bal:.4f}")