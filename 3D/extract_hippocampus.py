import nibabel as nib
import numpy as np
import os
import pandas as pd

# --- 1. Dossiers et chemins ---
data_dir = "/home/llatarche002/ComputerVision/alzeihmer-computer-vision/adni"   # à adapter
atlas_path = "/home/llatarche002/ComputerVision/alzeihmer-computer-vision/HarvardOxford-sub-maxprob-thr25-1mm.nii.gz"
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

# --- Paramètres ---
data_dir = "adni"
output_dir = "hippocampi_cubes"
os.makedirs(output_dir, exist_ok=True)

# --- Chargement du CSV ---
csv_path = "adni/list_standardized_tongtong_2017.csv"
df = pd.read_csv(csv_path)

# --- Initialisation des listes ---
x_train = []
y_train = []
x_test = []
y_test = []

# --- Itération sur le CSV ---
for idx, row in df.iterrows():
    nom = row[0]        # première colonne : nom du fichier
    label_raw = row[4]  # cinquième colonne : label
    mci_code = row[5]   # sixième colonne : code pour MCI

    # Construire le chemin du fichier
    fname = f"n_mmni_fADNI_{nom}_1.5T_t1w.nii.gz"
    img_path = os.path.join(data_dir, fname)
    mask_path = os.path.join(data_dir, f"mask_{fname}")

    if not os.path.exists(img_path):
        print(f"⚠️ Fichier IRM non trouvé : {img_path}, on saute.")
        continue
    if not os.path.exists(mask_path):
        print(f"⚠️ Masque non trouvé : {mask_path}, on saute.")
        continue

    #print(f"\nTraitement de {fname}...")

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

    # --- Ajouter aux datasets ---
    if label_raw == "AD":
        x_train.append(cube)
        y_train.append(1)
    elif label_raw == "CN":
        x_train.append(cube)
        y_train.append(0)
    elif label_raw == "MCI":
        if mci_code in [1, 2, 3]:
            x_test.append(cube)
            y_test.append(1)
        elif mci_code == 4:
            x_test.append(cube)
            y_test.append(0)
        elif mci_code == 5:
            print(f"Code 5 pour MCI, cube ignoré : {fname}")
            continue
        else:
            #print(f"⚠️ Code MCI inconnu : {mci_code}, cube ignoré.")
            continue
    else:
        #print(f"⚠️ Label inconnu pour {fname} : {label_raw}, on ignore.")
        continue

# --- Conversion en arrays numpy ---
x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

# --- Sauvegarder array ---
np.save(os.path.join("datas", "x_train_hippocampi.npy"), x_train)
np.save(os.path.join("datas", "y_train_hippocampi.npy"), y_train)
np.save(os.path.join("datas", "x_test_hippocampi.npy"), x_test)
np.save(os.path.join("datas", "y_test_hippocampi.npy"), y_test)

print(f"\n--- Résumé ---")
print(f"x_train : {x_train.shape}, y_train : {y_train.shape}")
print(f"x_test : {x_test.shape}")
