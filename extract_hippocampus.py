import nibabel as nib
import numpy as np
import os

# --- 1. Dossiers et chemins ---
data_dir = "/home/mfrancois008/ENSEIRB/ENSEIRB_3A/vision_ordinateur/projet/adni1-samples"   # à adapter
atlas_path = "/home/mfrancois008/fsl/data/atlases/HarvardOxford/HarvardOxford-sub-maxprob-thr25-1mm.nii.gz"
output_dir = os.path.join(data_dir, "../hippocampi_cubes")
os.makedirs(output_dir, exist_ok=True)

# --- 2. Charger l’atlas une fois ---
atlas = nib.load(atlas_path)
atlas_data = atlas.get_fdata()

# Créer un masque hippocampes (gauche + droite)
mask_hipp = np.logical_or(atlas_data == 8, atlas_data == 18)

# Bounding box commune (en coordonnées MNI)
coords = np.array(np.where(mask_hipp))
x_min, x_max = coords[0].min(), coords[0].max()
y_min, y_max = coords[1].min(), coords[1].max()
z_min, z_max = coords[2].min(), coords[2].max()

# Ajouter une marge pour bien englober la zone
margin = 15
x_min = max(0, x_min - margin)
x_max = min(mask_hipp.shape[0]-1, x_max + margin)
y_min = max(0, y_min - margin)
y_max = min(mask_hipp.shape[1]-1, y_max + margin)
z_min = max(0, z_min - margin)
z_max = min(mask_hipp.shape[2]-1, z_max + margin)

print(f"Bounding box MNI hippocampes : X[{x_min}:{x_max}], Y[{y_min}:{y_max}], Z[{z_min}:{z_max}]")

# --- 3. Itération sur toutes les IRM ---
for fname in os.listdir(data_dir):
    if not fname.endswith(".nii.gz"):
        continue
    if fname.startswith("mask_"):
        continue  # on ignore les masques ici, on les chargera selon l'image correspondante

    img_path = os.path.join(data_dir, fname)
    mask_path = os.path.join(data_dir, f"mask_{fname}")

    if not os.path.exists(mask_path):
        print(f"⚠️ Pas de masque trouvé pour {fname}, on saute.")
        continue

    print(f"\nTraitement de {fname}...")

    # --- Charger IRM et masque ---
    img = nib.load(img_path)
    mask = nib.load(mask_path)
    img_data = img.get_fdata()
    mask_data = mask.get_fdata()

    # --- Appliquer le masque du cerveau ---
    img_brain = img_data * (mask_data > 0)

    # --- Extraire le cube hippocampes ---
    cube = img_brain[x_min:x_max, y_min:y_max, z_min:z_max]

    # --- Ajuster l’affine ---
    new_affine = img.affine.copy()
    new_affine[:3, 3] += np.dot(img.affine[:3, :3], np.array([x_min, y_min, z_min]))

    # --- Sauvegarder ---
    output_path = os.path.join(output_dir, f"hippocampi_cube_{fname}")
    cube_img = nib.Nifti1Image(cube, new_affine, img.header)
    nib.save(cube_img, output_path)

    print(f"Cube sauvegardé : {output_path}")
    print(f"   Taille : {cube.shape}")
