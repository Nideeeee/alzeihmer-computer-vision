import torch
import os
import matplotlib.pyplot as plt
import random
import numpy as np

# === CONFIGURATION ===
# Dossier contenant tes .pt générés
DATA_DIR = "slices_cache_filtered/mci/train" 

def show_3_channels():
    # 1. Trouver un patient au hasard
    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".pt")]
    if not files:
        print(f"❌ Aucune donnée trouvée dans {DATA_DIR}")
        return

    random_file = random.choice(files)
    path = os.path.join(DATA_DIR, random_file)
    print(f"Visualisation du fichier : {random_file}")

    # 2. Charger les données
    # Shape: [N_slices, 3, H, W]
    slices_tensor, label = torch.load(path)
    
    # 3. Choisir une slice intéressante (au milieu du cerveau)
    # On évite le début et la fin qui sont souvent petits
    idx = len(slices_tensor) // 2 
    
    # On récupère le bloc de 3 channels : [3, H, W]
    image_3d = slices_tensor[idx]

    # Séparation des canaux
    slice_prev = image_3d[0, :, :].numpy() # Canal Rouge (i-1)
    slice_curr = image_3d[1, :, :].numpy() # Canal Vert  (i)
    slice_next = image_3d[2, :, :].numpy() # Canal Bleu  (i+1)

    # Préparation image composite RGB pour affichage
    # Matplotlib veut (H, W, 3), PyTorch est (3, H, W) -> on permute
    rgb_composite = image_3d.permute(1, 2, 0).numpy()
    
    # Normalisation pour affichage propre (0-1) si ce n'est pas déjà le cas
    rgb_composite = (rgb_composite - rgb_composite.min()) / (rgb_composite.max() - rgb_composite.min())

    # === AFFICHAGE ===
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    fig.suptitle(f"Visualisation de l'input 2.5D (Sujet {random_file.split('.')[0]}, Bloc #{idx})", fontsize=16)

    # Canal 1 : Slice Précédente
    axes[0].imshow(slice_prev, cmap='gray')
    axes[0].set_title("Canal 1 : Slice (i-1)", fontsize=14, color='red')
    axes[0].axis('off')

    # Canal 2 : Slice Actuelle
    axes[1].imshow(slice_curr, cmap='gray')
    axes[1].set_title("Canal 2 : Slice Centrale (i)", fontsize=14, color='green')
    axes[1].axis('off')

    # Canal 3 : Slice Suivante
    axes[2].imshow(slice_next, cmap='gray')
    axes[2].set_title("Canal 3 : Slice Suivante (i+1)", fontsize=14, color='blue')
    axes[2].axis('off')

    # Image Composite (Ce que voit le modèle)
    axes[3].imshow(rgb_composite)
    axes[3].set_title("Image Composite (RGB)\nCe que voit le ResNet", fontsize=14, fontweight='bold')
    axes[3].axis('off')

    plt.tight_layout()
    save_name = "viz_2.5D_channels.png"
    plt.savefig(save_name, dpi=150)
    print(f"✅ Image sauvegardée sous : {save_name}")
    plt.show()

if __name__ == "__main__":
    show_3_channels()