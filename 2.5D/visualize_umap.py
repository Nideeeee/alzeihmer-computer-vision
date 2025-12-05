import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torchvision.models as models
import umap  # <--- Bibliothèque UMAP
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

# Modèle pour l'inférence (doit correspondre à ton architecture entraînée)
class ResNetForInference(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNetForInference, self).__init__()
        base_model = models.resnet18(weights=None) 
        self.features = nn.Sequential(*list(base_model.children())[:-1])
        # Note: Si tu as utilisé un Dropout dans l'entraînement avancé, 
        # il est ignoré ici car on est en mode .eval()
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        if x.max() > 10.0: x = x / 255.0
        x = self.features(x)
        x = x.view(x.size(0), -1) 
        features = x      
        logits = self.fc(x) 
        return logits, features

# ==========================================
# 2. CONFIGURATION
# ==========================================
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- A ADAPTER ---
test_dir = "slices_cache_filtered/mci/test" 
model_path = "weights/best_model_loss.pth" # Ton meilleur modèle

if not os.path.exists(model_path):
    print(f"⚠ '{model_path}' introuvable, essai avec 'best_model_temp.pth'...")
    model_path = "weights/best_model_temp.pth"
# Chargement Données
print(f"Chargement des données depuis {test_dir}...")
test_dataset = ADNI2p5DSingleSubjectDataset(test_dir)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

# Chargement Modèle
print(f"Chargement du modèle depuis {model_path}...")
model = ResNetForInference(num_classes=2).to(device)
try:
    model.load_state_dict(torch.load(model_path, map_location=device))
except:
    print("⚠ Chargement strict échoué, essai avec strict=False...")
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
model.eval()

# ==========================================
# 3. EXTRACTION DES EMBEDDINGS
# ==========================================
print("Extraction des features...")

patient_embeddings = []
patient_preds = []
patient_labels = []

with torch.no_grad():
    for slices, label in tqdm(test_loader):
        slices = slices[0].to(device)
        label = label.item()
        
        logits, features = model(slices)
        
        # Moyenne des embeddings (Aggregation Patient)
        patient_feat = features.mean(dim=0).cpu().numpy()
        
        # Prédiction (Vote majoritaire)
        slice_preds = logits.argmax(dim=1).cpu().numpy()
        patient_pred = Counter(slice_preds).most_common(1)[0][0]
        
        patient_embeddings.append(patient_feat)
        patient_preds.append(patient_pred)
        patient_labels.append(label)

X = np.array(patient_embeddings)
y_pred = np.array(patient_preds)
y_true = np.array(patient_labels)

# ==========================================
# 4. CALCUL UMAP
# ==========================================
print(f"Lancement de UMAP sur {len(X)} patients...")

# Paramètres UMAP optimisés pour la visualisation
# n_neighbors : Taille du voisinage (plus petit = plus local, plus grand = plus global)
# min_dist : Distance min entre les points (plus petit = clusters plus serrés)
reducer = umap.UMAP(
    n_neighbors=15, 
    min_dist=0.1, 
    n_components=2, 
    metric='cosine', # Souvent meilleur pour les embeddings profonds que 'euclidean'
    random_state=42
)
X_embedded = reducer.fit_transform(X)

# ==========================================
# 5. VISUALISATION
# ==========================================
# Stats rapides
tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
acc = (tn + tp) / len(y_true)
print(f"Accuracy du modèle chargé : {acc:.2%}")

fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Graphe 1 : PRÉDICTIONS
scatter1 = axes[0].scatter(X_embedded[:, 0], X_embedded[:, 1], c=y_pred, cmap='coolwarm', alpha=0.8, edgecolors='k')
axes[0].set_title(f"UMAP - PRÉDICTIONS du Modèle\n(Bleu=Stable Prédit, Rouge=Conv Prédit)")
legend1 = axes[0].legend(*scatter1.legend_elements(), title="Pred")
axes[0].add_artist(legend1)
axes[0].grid(True, alpha=0.3)

# Graphe 2 : VRAIS LABELS
scatter2 = axes[1].scatter(X_embedded[:, 0], X_embedded[:, 1], c=y_true, cmap='viridis', alpha=0.8, edgecolors='k')
axes[1].set_title(f"UMAP - VRAIS LABELS (Vérité Terrain)\n(Violet=Stable Réel, Jaune=Conv Réel)")
legend2 = axes[1].legend(*scatter2.legend_elements(), title="True")
axes[1].add_artist(legend2)
axes[1].grid(True, alpha=0.3)

out_filename = "umap_mci_comparison.png"
plt.tight_layout()
plt.savefig(out_filename, dpi=300)
print(f"✅ Image sauvegardée sous : '{out_filename}'")
plt.show()