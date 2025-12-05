import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

# =================CONFIGURATION=================
IMAGES_DIR = "../../adni"        
MASKS_DIR = "../../adni"         
CSV_FILE = "../../adni/list_standardized_tongtong_2017.csv"

# Tes paramètres actuels
THRESHOLD = 0.001   
STRIDE = 3          
CROP_MARGIN = 0.15 
# ===============================================

def visualize_preprocessing():
    # 1. Charger un sujet au hasard
    if not os.path.exists(CSV_FILE):
        print("❌ CSV introuvable.")
        return

    df = pd.read_csv(CSV_FILE, header=None)
    # On prend la première colonne comme SubjectID
    subject_ids = df[0].values
    
    # On cherche un sujet valide (qui a bien une image sur le disque)
    found = False
    while not found:
        subject_id = random.choice(subject_ids)
        # Format du SubjectID dans le CSV vs Nom de fichier : A ADAPTER SI BESOIN
        # Ton CSV a l'air d'avoir "002_S_0295" etc.
        img_path = os.path.join(IMAGES_DIR, f"n_mmni_fADNI_{subject_id}_1.5T_t1w.nii.gz")
        if os.path.exists(img_path):
            found = True
            print(f"✅ Sujet trouvé pour vérification : {subject_id}")
            print(f"   Chemin : {img_path}")
        else:
            # Sécurité pour ne pas boucler infini si dossier vide
            pass

    # 2. Chargement & Normalisation
    vol = nib.load(img_path).get_fdata()
    
    # Masque (si dispo)
    mask_path = os.path.join(MASKS_DIR, f"mask_n_mmni_fADNI_{subject_id}_1.5T_t1w.nii.gz")
    if os.path.exists(mask_path):
        print("   Application du masque...")
        vol = vol * nib.load(mask_path).get_fdata()
        
    v_min, v_max = vol.min(), vol.max()
    vol_norm = (vol - v_min) / (v_max - v_min) # [0, 1]
    
    # 3. Simulation du Processus (Crop + Stride + Threshold)
    depth = vol_norm.shape[0]
    start = int(depth * CROP_MARGIN)
    end = int(depth * (1 - CROP_MARGIN))
    
    print(f"   Volume original : {depth} slices")
    print(f"   Zone de Crop    : {start} à {end} (Marge {CROP_MARGIN*100}%)")
    
    kept_slices = []
    skipped_indices = []
    
    for i in range(start, end, STRIDE):
        # On extrait la slice centrale du bloc 2.5D pour visualiser
        s = vol_norm[i, :, :]
        
        if s.mean() >= THRESHOLD:
            kept_slices.append(s)
        else:
            skipped_indices.append(i)
            
    print(f"   Slices conservées : {len(kept_slices)} (Stride={STRIDE})")
    print(f"   Slices rejetées par threshold : {len(skipped_indices)}")

    # 4. Affichage (Grid Plot)
    if len(kept_slices) == 0:
        print("❌ Aucune slice n'a survécu au filtrage !")
        return

    # Calcul de la grille (carrée approx)
    n_slices = len(kept_slices)
    cols = 8
    rows = (n_slices // cols) + 1
    
    plt.figure(figsize=(20, rows * 2.5))
    plt.suptitle(f"Slices Finales pour {subject_id}\n(Crop {CROP_MARGIN*100}% | Stride {STRIDE} | Thresh {THRESHOLD})", fontsize=16)
    
    for idx, slice_img in enumerate(kept_slices):
        plt.subplot(rows, cols, idx + 1)
        plt.imshow(slice_img, cmap='gray')
        plt.axis('off')
        plt.title(f"Slice #{idx+1}", fontsize=8)
    
    plt.tight_layout()
    out_file = "verification_slices.png"
    plt.savefig(out_file)
    print(f"\n✅ Image générée : '{out_file}'")
    print("Ouvre cette image pour vérifier que le cerveau est bien visible et centré.")

if __name__ == "__main__":
    visualize_preprocessing()