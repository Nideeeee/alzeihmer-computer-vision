import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torchvision.models as models
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from collections import Counter
from tqdm import tqdm

# ==========================================
# 1. DÉFINITIONS
# ==========================================

class ADNI2p5DSingleSubjectDataset(Dataset):
    def __init__(self, split_dir):
        self.files = sorted([os.path.join(split_dir, f) for f in os.listdir(split_dir) if f.endswith(".pt")])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return torch.load(self.files[idx])

class ResNetForInference(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNetForInference, self).__init__()
        base_model = models.resnet18(weights=None) 
        self.features = nn.Sequential(*list(base_model.children())[:-1])
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        if x.max() > 10.0: x = x / 255.0
        x = self.features(x)
        x = x.view(x.size(0), -1) 
        features = x      
        logits = self.fc(x) 
        return logits, features

def extract_embeddings(loader, model, device):
    embeddings, preds, labels = [], [], []
    with torch.no_grad():
        for slices, label in tqdm(loader, desc="Extraction"):
            slices = slices[0].to(device)
            label = label.item()
            
            logits, features = model(slices)
            
            # Agrégation Patient (Moyenne)
            patient_feat = features.mean(dim=0).cpu().numpy()
            slice_preds = logits.argmax(dim=1).cpu().numpy()
            patient_pred = Counter(slice_preds).most_common(1)[0][0]
            
            embeddings.append(patient_feat)
            preds.append(patient_pred)
            labels.append(label)
    return np.array(embeddings), np.array(preds), np.array(labels)

# ==========================================
# 2. CONFIGURATION
# ==========================================
device = "cuda" if torch.cuda.is_available() else "cpu"

# Chemins
mci_test_dir = "slices_cache_filtered/mci/test" 
cn_ad_test_dir = "slices_cache_filtered/mci/train" # Nouveau dossier pour CN/AD

model_path = "weights/best_model_loss.pth" 

if not os.path.exists(model_path):
    print(f"⚠ '{model_path}' introuvable, essai avec 'best_model.pth'...")
    model_path = "best_model.pth"

# Chargement Modèle
print(f"Chargement du modèle depuis {model_path}...")
model = ResNetForInference(num_classes=2).to(device)
try:
    model.load_state_dict(torch.load(model_path, map_location=device))
except:
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
model.eval()

# ==========================================
# 3. ANALYSE 1 : MCI (STABLE vs CONVERTER)
# ==========================================
print("\n--- ANALYSE MCI (Test Set) ---")
mci_dataset = ADNI2p5DSingleSubjectDataset(mci_test_dir)
mci_loader = DataLoader(mci_dataset, batch_size=1, shuffle=False, num_workers=4)

X_mci, y_pred_mci, y_true_mci = extract_embeddings(mci_loader, model, device)

# t-SNE MCI
tsne = TSNE(n_components=2, perplexity=min(30, len(X_mci)-1), random_state=42)
X_emb_mci = tsne.fit_transform(X_mci)

# PLOT MCI
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Couleurs personnalisées (Hex codes pour être bien visibles)
# 0 = Bleu (Stable), 1 = Rouge (Converter)
colors = ['#1f77b4', '#d62728'] 
labels_map = ['Stable (0)', 'Converter (1)']

# Graphe 1 : Prédictions
for i in [0, 1]:
    idx = y_pred_mci == i
    axes[0].scatter(X_emb_mci[idx, 0], X_emb_mci[idx, 1], c=colors[i], label=labels_map[i], s=80, alpha=0.8, edgecolors='white')
axes[0].set_title("PRÉDICTIONS (MCI)")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Graphe 2 : Vrais Labels
for i in [0, 1]:
    idx = y_true_mci == i
    axes[1].scatter(X_emb_mci[idx, 0], X_emb_mci[idx, 1], c=colors[i], label=labels_map[i], s=80, alpha=0.8, edgecolors='white')
axes[1].set_title("VRAIS LABELS (MCI)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("tsne_mci_vivid.png", dpi=300)
print("✅ tsne_mci_vivid.png sauvegardé.")

# ==========================================
# 4. ANALYSE 2 : CN vs AD (LES EXTRÊMES)
# ==========================================
print("\n--- ANALYSE CN/AD (Test Set) ---")
if os.path.exists(cn_ad_test_dir):
    cn_ad_dataset = ADNI2p5DSingleSubjectDataset(cn_ad_test_dir)
    cn_ad_loader = DataLoader(cn_ad_dataset, batch_size=1, shuffle=False, num_workers=4)

    X_cnad, y_pred_cnad, y_true_cnad = extract_embeddings(cn_ad_loader, model, device)

    # t-SNE CN/AD
    tsne_cnad = TSNE(n_components=2, perplexity=min(30, len(X_cnad)-1), random_state=42)
    X_emb_cnad = tsne_cnad.fit_transform(X_cnad)

    # PLOT CN/AD
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 8))
    
    # Couleurs différentes pour bien distinguer : Vert (CN) vs Violet (AD)
    colors_cnad = ['#2ca02c', '#9467bd'] 
    labels_cnad = ['Cognitively Normal (CN)', 'Alzheimer (AD)']

    for i in [0, 1]: # 0=CN, 1=AD
        idx = y_true_cnad == i
        ax2.scatter(X_emb_cnad[idx, 0], X_emb_cnad[idx, 1], c=colors_cnad[i], label=labels_cnad[i], s=100, alpha=0.8, edgecolors='k')
    
    ax2.set_title("t-SNE : CN vs AD (Vrais Labels)\nPermet de voir la séparabilité 'facile'")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("tsne_cn_ad.png", dpi=300)
    print("✅ tsne_cn_ad.png sauvegardé.")
else:
    print("⚠ Dossier CN/AD introuvable, pas de 2ème graphe.")

plt.show()