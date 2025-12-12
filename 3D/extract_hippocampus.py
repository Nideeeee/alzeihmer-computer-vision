import nibabel as nib
import numpy as np
import os
import pandas as pd

# --- 1. Dossiers et chemins ---
data_dir = "/home/llatarche002/ComputerVision/alzeihmer-computer-vision/adni"   # à adapter
atlas_path = "/home/llatarche002/ComputerVision/alzeihmer-computer-vision/HarvardOxford-sub-maxprob-thr25-1mm.nii.gz"
output_dir = os.path.join(data_dir, "../hippocampi_cubes")
os.makedirs(output_dir, exist_ok=True)

x_min_1, x_max_1 = 40, 80
x_min_2, x_max_2 = 100, 140
y_min, y_max = 90, 130
z_min, z_max = 40, 80

# --- Paramètres ---
output_dir = "hippocampi_cubes"
os.makedirs(output_dir, exist_ok=True)

# --- Chargement du CSV ---
csv_path = os.path.join(data_dir, "list_standardized_tongtong_2017.csv")
df = pd.read_csv(csv_path)

# --- Initialisation des listes ---
x_train = []
y_train = []
x_test = []
y_test = []

# --- Itération sur le CSV ---
nb_MCI_AD = 30
nb_MCI_CN = 30
for idx, row in df.iterrows():
    nom = row[0]        # première colonne : nom du fichier
    label_raw = row[4]  # cinquième colonne : label
    mci_code = row[5]   # sixième colonne : code pour MCI

    # Construire le chemin du fichier
    fname = f"n_mmni_fADNI_{nom}_1.5T_t1w.nii.gz"
    img_path = os.path.join(data_dir, fname)
    mask_path = os.path.join(data_dir, f"mask_{fname}")

    if not os.path.exists(img_path):
        print(f"Fichier IRM non trouvé : {img_path}.")
        continue
    if not os.path.exists(mask_path):
        print(f"Masque non trouvé : {mask_path}.")
        continue

    # --- Charger IRM et masque ---
    img = nib.load(img_path)
    mask = nib.load(mask_path)
    img_data = img.get_fdata()
    mask_data = mask.get_fdata()

    # --- Appliquer le masque du cerveau ---
    img_brain = img_data * (mask_data > 0)

    # --- Extraire le cube hippocampes ---
    cube1 = img_brain[x_min_1:x_max_1, y_min:y_max, z_min:z_max]
    cube2 = img_brain[x_min_2:x_max_2, y_min:y_max, z_min:z_max]
    cube = np.concatenate((cube1, cube2), axis=0)
    # --- Ajouter aux datasets ---
    if label_raw == "AD":
        x_train.append(cube)
        y_train.append(1)
    elif label_raw == "CN":
        x_train.append(cube)
        y_train.append(0)
    elif label_raw == "MCI":
        if mci_code in [1, 2, 3]:
            if nb_MCI_AD >0:
                x_train.append(cube)
                y_train.append(1)
                nb_MCI_AD -= 1
            else:
                x_test.append(cube)
                y_test.append(1)
        elif mci_code == 4:
            if nb_MCI_CN >0:
                x_train.append(cube)
                y_train.append(0)
                nb_MCI_CN -= 1
            else:
                x_test.append(cube)
                y_test.append(0)
        elif mci_code == 5:
            continue
        else:
            continue
    else:
        continue

# --- Conversion en arrays numpy ---
x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

# --- Sauvegarder array ---
np.save(os.path.join("datas", "x_train_hippocampi_40.npy"), x_train)
np.save(os.path.join("datas", "y_train_hippocampi_40.npy"), y_train)
np.save(os.path.join("datas", "x_test_hippocampi_40.npy"), x_test)
np.save(os.path.join("datas", "y_test_hippocampi_40.npy"), y_test)

print(f"\n--- Résumé ---")
print([c.shape for c in x_train[:20]])
print(f"x_train : {x_train.shape}, y_train : {y_train.shape}")
print(f"x_test : {x_test.shape}")
