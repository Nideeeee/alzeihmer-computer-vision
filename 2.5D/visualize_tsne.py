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
# 1. DÉFINITIONS (Doit matcher l'entraînement)
# ==========================================

class ADNI2p5DSingleSubjectDataset(Dataset):
    def __init__(self, split_dir):
        self.files = sorted([os.path.join(split_dir, f) for f in os.listdir(split_dir) if f.endswith(".pt")])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return torch.load(self.files[idx])

# Modèle identique à l'entraînement, mais qui renvoie (logits, features)
class ResNetForInference(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNetForInference, self).__init__()
        # On initialise l'architecture ResNet18 standard
        base_model = models.resnet18(weights=None) 
        self.features = nn.Sequential(*list(base_model.children())[:-1])
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        # Normalisation si nécessaire (comme à l'entraînement)
        if x.max() > 10.0: x = x / 255.0
        
        x = self.features(x)
        x = x.view(x.size(0), -1) 
        
        features = x      # On capture les embeddings (taille 512)
        logits = self.fc(x) # On calcule la prédiction
        
        # On renvoie les DEUX
        return logits, features

# ==========================================
# 2. CONFIGURATION
# ==========================================
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- CONFIGURATION DES CHEMINS ---
test_dir = "slices_cache_filtered/mci/test" 
model_path = "weights/best_model_loss.pth" # Ton meilleur modèle

if not os.path.exists(model_path):
    print(f"⚠ '{model_path}' introuvable, essai avec 'best_model_temp.pth'...")
    model_path = "weights/best_model_temp.pth"
# Vérifications
if not os.path.exists(test_dir):
    raise FileNotFoundError(f"❌ Dossier de test introuvable : {test_dir}")
if not os.path.exists(model_path):
    print(f"⚠ Attention : '{model_path}' introuvable.")
    model_path = "best_model_slice_training.pth" # Fallback sur l'ancien modèle si besoin
    print(f"-> Essai avec '{model_path}'...")
    if not os.path.exists(model_path):
        raise FileNotFoundError("❌ Aucun modèle trouvé !")

print(f"Chargement des données depuis {test_dir}...")
test_dataset = ADNI2p5DSingleSubjectDataset(test_dir)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

print(f"Chargement du modèle depuis {model_path}...")
model = ResNetForInference(num_classes=2).to(device)

# Chargement des poids (ignore les différences mineures si strict=False, mais ici ça devrait matcher)
try:
    model.load_state_dict(torch.load(model_path, map_location=device))
except RuntimeError as e:
    print("⚠ Erreur de chargement stricte, tentative avec strict=False...")
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)

model.eval()

# ==========================================
# 3. EXTRACTION
# ==========================================
print("Extraction des embeddings et prédictions en cours...")

patient_embeddings = []
patient_preds = []
patient_labels = []

with torch.no_grad():
    for slices, label in tqdm(test_loader):
        slices = slices[0].to(device)
        label = label.item()
        
        # Récupération Logits ET Features
        logits, features = model(slices)
        
        # 1. Moyenne des features (Aggregation Patient pour t-SNE)
        patient_feat = features.mean(dim=0).cpu().numpy()
        
        # 2. Vote majoritaire (Agrégation Patient pour Prédiction)
        slice_preds = logits.argmax(dim=1).cpu().numpy()
        patient_pred = Counter(slice_preds).most_common(1)[0][0]
        
        patient_embeddings.append(patient_feat)
        patient_preds.append(patient_pred)
        patient_labels.append(label)

# Conversion en numpy arrays
X = np.array(patient_embeddings)
y_pred = np.array(patient_preds)
y_true = np.array(patient_labels)

# ==========================================
# 4. STATISTIQUES COMPARATIVES
# ==========================================
print("\n" + "="*40)
print("   BILAN PRÉDICTIONS vs RÉALITÉ")
print("="*40)

# Comptage
count_pred_0 = (y_pred == 0).sum() # Stable prédits
count_pred_1 = (y_pred == 1).sum() # Converter prédits

count_true_0 = (y_true == 0).sum() # Stable réels
count_true_1 = (y_true == 1).sum() # Converter réels

print(f"\n1. VOLUMES GLOBAUX :")
print(f"{'Classe':<15} | {'Prédits (Modèle)':<18} | {'Réels (Labels)':<15}")
print("-" * 55)
print(f"{'Stable (0)':<15} | {count_pred_0:<18} | {count_true_0:<15}")
print(f"{'Converter (1)':<15} | {count_pred_1:<18} | {count_true_1:<15}")

# Matrice de confusion
try:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
except ValueError:
    # Cas rare où une classe serait absente
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]

print(f"\n2. DÉTAILS (Matrice de Confusion) :")
print(f" - Vrais Stables (TN)      : {tn}")
print(f" - Vrais Converters (TP)   : {tp}")
print(f" - Faux Positifs (FP)      : {fp} (Stable prédit Converter)")
print(f" - Faux Négatifs (FN)      : {fn} (Converter prédit Stable)")

total_acc = (tn + tp) / len(y_true)
print(f"\n> Précision Globale : {total_acc:.2%}")
print("="*40 + "\n")

# ==========================================
# 5. CALCUL t-SNE & VISUALISATION
# ==========================================
if len(X) < 5:
    print("⚠ Pas assez de données pour le t-SNE.")
else:
    print(f"Lancement t-SNE sur {len(X)} patients...")
    # Perplexity auto-adaptative
    perp = min(30, len(X) - 1)
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42, init='pca', learning_rate='auto')
    X_embedded = tsne.fit_transform(X)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Graphe 1 : Prédictions
    scatter1 = axes[0].scatter(X_embedded[:, 0], X_embedded[:, 1], c=y_pred, cmap='coolwarm', alpha=0.8, edgecolors='k')
    axes[0].set_title(f"PRÉDICTIONS du Modèle\nStable={count_pred_0}, Converter={count_pred_1}")
    legend1 = axes[0].legend(*scatter1.legend_elements(), title="Pred (0=Stable, 1=Conv)")
    axes[0].add_artist(legend1)
    axes[0].grid(True, alpha=0.3)

    # Graphe 2 : Vrais Labels
    scatter2 = axes[1].scatter(X_embedded[:, 0], X_embedded[:, 1], c=y_true, cmap='viridis', alpha=0.8, edgecolors='k')
    axes[1].set_title(f"VRAIS LABELS (Vérité Terrain)\nStable={count_true_0}, Converter={count_true_1}")
    legend2 = axes[1].legend(*scatter2.legend_elements(), title="True (0=Stable, 1=Conv)")
    axes[1].add_artist(legend2)
    axes[1].grid(True, alpha=0.3)

    out_filename = "tsne_mci_finetuned.png"
    plt.tight_layout()
    plt.savefig(out_filename, dpi=300)
    print(f"✅ Image sauvegardée sous : '{out_filename}'")
    plt.show()