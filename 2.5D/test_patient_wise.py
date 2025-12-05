import torch
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

torch.backends.cudnn.benchmark = True

# ==========================================
# 1. DATASET (PATIENT-WISE)
# ==========================================
class ADNI2p5DSingleSubjectDataset(Dataset):
    def __init__(self, split_dir):
        self.files = sorted([os.path.join(split_dir, f) for f in os.listdir(split_dir) if f.endswith(".pt")])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Renvoie (slices_tensor, label_tensor)
        return torch.load(self.files[idx])

# Configuration des chemins
train_dir = "slices_cache_filtered/mci/train" 
valid_dir = "slices_cache_filtered/mci/valid"
test_dir  = "slices_cache_filtered/mci/test"

# DataLoaders : BATCH_SIZE = 1 (On traite Patient par Patient)
# On n'utilise pas collate_fn complexe ici car on charge 1 patient à la fois
train_loader = DataLoader(ADNI2p5DSingleSubjectDataset(train_dir), batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
valid_loader = DataLoader(ADNI2p5DSingleSubjectDataset(valid_dir), batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
test_loader  = DataLoader(ADNI2p5DSingleSubjectDataset(test_dir),  batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

print(f"Dataset -> Train: {len(train_loader.dataset)}, Valid: {len(valid_loader.dataset)}, Test: {len(test_loader.dataset)}")

# ==========================================
# 2. MODÈLE (PARTIAL FINE-TUNING)
# ==========================================
class ResNetPartial(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNetPartial, self).__init__()
        # Chargement poids ImageNet
        base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        self.features = nn.Sequential(*list(base_model.children())[:-1])
        self.fc = nn.Linear(512, num_classes)

        # 1. Geler tout le backbone
        for param in self.features.parameters():
            param.requires_grad = False
        
        # 2. Dégeler uniquement Layer 4
        for name, param in self.features.named_parameters():
            if "layer4" in name:
                param.requires_grad = True

        print("Modèle Patient-wise : Layers 1-3 GELÉS | Layer 4 + FC ENTRAÎNABLES")

    def forward(self, x):
        # Normalisation si nécessaire
        if x.max() > 10.0: x = x / 255.0
        x = self.features(x)
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        return logits

device = "cuda" if torch.cuda.is_available() else "cpu"
model = ResNetPartial(num_classes=2).to(device)

# ==========================================
# 3. SETUP (POIDS, OPTIM, AMP)
# ==========================================
def compute_class_weights(loader):
    print("Calcul des poids...")
    labels = []
    for _, label in loader:
        labels.append(label.item())
    
    c0 = labels.count(0)
    c1 = labels.count(1)
    if c0 == 0 or c1 == 0: return torch.tensor([1.0, 1.0]).to(device)
    
    w0 = len(labels) / (2 * c0)
    w1 = len(labels) / (2 * c1)
    print(f"   -> CN/Stable: {c0}, AD/Conv: {c1} | Poids: {w0:.2f}, {w1:.2f}")
    return torch.tensor([w0, w1], dtype=torch.float32).to(device)

class_weights = compute_class_weights(train_loader)
criterion = nn.CrossEntropyLoss(weight=class_weights)

# LR faible car Fine-Tuning
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5, weight_decay=1e-4)
scaler = GradScaler()

# ==========================================
# 4. FONCTION D'ÉVALUATION (PATIENT-WISE)
# ==========================================
def evaluate_patient_metrics(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    criterion_val = nn.CrossEntropyLoss() # Loss brute pour validation

    with torch.no_grad():
        for slices, label in loader:
            slices = slices[0].to(device) # [N_slices, 3, H, W]
            label = label.to(device)      # [1]
            
            logits = model(slices)
            
            # Loss sur toutes les slices (Label répété)
            loss = criterion_val(logits, label.repeat(slices.size(0)))
            total_loss += loss.item()

            # --- Agrégation Patient ---
            # Moyenne des logits -> Argmax
            patient_logit = logits.mean(dim=0)
            patient_pred = patient_logit.argmax().item()
            
            all_preds.append(patient_pred)
            all_labels.append(label.item())
            
    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    
    return avg_loss, acc, bal_acc, all_labels, all_preds

# ==========================================
# 5. ENTRAÎNEMENT
# ==========================================
num_epochs = 15
best_val_bal_acc = 0.0
save_path = "best_model_patientwise.pth"

train_losses, val_losses = [], []
train_accs, val_accs = [], []
val_bal_accs = []

print(f"\n--- Démarrage ({num_epochs} époques) ---")

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0.0
    correct_train_patients = 0
    total_train_patients = 0
    
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
    
    for slices, label in loop:
        slices = slices[0].to(device)
        label = label.to(device)
        
        optimizer.zero_grad()
        
        # --- Mixed Precision ---
        with autocast():
            logits = model(slices)
            # La loss est calculée sur chaque slice indépendamment
            loss = criterion(logits, label.repeat(slices.size(0)))
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Metrics Train (Patient-wise approximation)
        total_train_loss += loss.item() # Moyenne des slices du patient
        
        # Prediction patient pour le suivi Train
        patient_pred = logits.mean(dim=0).argmax().item()
        if patient_pred == label.item():
            correct_train_patients += 1
        total_train_patients += 1
        
        loop.set_postfix(loss=loss.item())

    # Moyennes Train
    avg_train_loss = total_train_loss / total_train_patients
    avg_train_acc = correct_train_patients / total_train_patients
    
    # Validation
    val_loss, val_acc, val_bal_acc, _, _ = evaluate_patient_metrics(model, valid_loader, device)
    
    # Stockage
    train_losses.append(avg_train_loss)
    val_losses.append(val_loss)
    train_accs.append(avg_train_acc)
    val_accs.append(val_acc)
    val_bal_accs.append(val_bal_acc)
    
    # Affichage Complet
    print(f"Epoch {epoch+1} Results:")
    print(f"  • Train Loss: {avg_train_loss:.4f} | Train Acc (Patient): {avg_train_acc:.4f}")
    print(f"  • Val Loss  : {val_loss:.4f}     | Val Acc: {val_acc:.4f} | Bal Acc: {val_bal_acc:.4f}")
    
    # Sauvegarde sur Balanced Accuracy
    if val_bal_acc > best_val_bal_acc:
        best_val_bal_acc = val_bal_acc
        torch.save(model.state_dict(), save_path)
        print(f"  ★ Nouveau record (Bal Acc) ! Modèle sauvegardé.")
    print("-" * 50)

# ==========================================
# 6. TEST FINAL
# ==========================================
print(f"\n--- Test Final (Meilleur Modèle) ---")
model.load_state_dict(torch.load(save_path))

test_loss, test_acc, test_bal_acc, y_true, y_pred = evaluate_patient_metrics(model, test_loader, device)

print(f"Accuracy Globale : {test_acc:.2%}")
print(f"Balanced Accuracy: {test_bal_acc:.2%}")

target_names = ["Stable (0)", "Converter (1)"]
print("\n--- Rapport Détaillé ---")
print(classification_report(y_true, y_pred, target_names=target_names))

print("\n--- Matrice de Confusion ---")
cm = confusion_matrix(y_true, y_pred)
print(f"TN={cm[0][0]} | FP={cm[0][1]}")
print(f"FN={cm[1][0]} | TP={cm[1][1]}")

# Courbes
plt.figure(figsize=(15,5))
plt.subplot(1,3,1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.title("Loss")
plt.legend()

plt.subplot(1,3,2)
plt.plot(train_accs, label='Train Acc')
plt.plot(val_accs, label='Val Acc')
plt.title("Accuracy (Patient-wise)")
plt.legend()

plt.subplot(1,3,3)
plt.plot(val_bal_accs, label='Val Bal Acc', color='green')
plt.title("Balanced Accuracy")
plt.legend()

plt.tight_layout()
plt.savefig("metrics_patientwise_finetuned.png")
print("\nCourbes sauvegardées sous 'metrics_patientwise_finetuned.png'")