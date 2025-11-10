import os
import nibabel as nib
import torch
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split

# === Configuration ===
LABEL_MAP = {"CN": 0, "AD": 1}
images_dir = "adni"
masks_dir = "adni"
csv_file = "adni/list_standardized_tongtong_2017.csv"
cache_dir = "slices_cache_filtered"
os.makedirs(cache_dir, exist_ok=True)
threshold = 0.001  # seuil pour filtrer les slices vides

# === Lecture CSV et split ===
df = pd.read_csv(csv_file, header=None)
df = df[df[4].isin(LABEL_MAP.keys())]
subjects = df[0].tolist()

train_subj, temp_subj = train_test_split(subjects, test_size=0.3, random_state=42)
valid_subj, test_subj = train_test_split(temp_subj, test_size=0.5, random_state=42)
splits = {"train": train_subj, "valid": valid_subj, "test": test_subj}

# === Traitement ===
for split_name, subj_list in splits.items():
    split_dir = os.path.join(cache_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)

    for subject_id in tqdm(subj_list, desc=f"Processing {split_name}"):
        row = df[df[0] == subject_id].iloc[0]
        label = LABEL_MAP[row[4]]

        img_file = os.path.join(images_dir, f"n_mmni_fADNI_{subject_id}_1.5T_t1w.nii.gz")
        mask_file = os.path.join(masks_dir, f"mask_n_mmni_fADNI_{subject_id}_1.5T_t1w.nii.gz")
        if not os.path.exists(img_file):
            continue

        vol = nib.load(img_file).get_fdata()
        if os.path.exists(mask_file):
            mask = nib.load(mask_file).get_fdata()
            vol = vol * mask

        # Normalisation [0,1]
        vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-8)
        vol = torch.tensor(vol, dtype=torch.float32)

        # Génération des slices 2.5D
        slices_2p5d = []
        for i in range(1, vol.shape[0] - 1):
            slice_2p5d = vol[i-1:i+2, :, :]  # [3, H, W]
            if slice_2p5d.mean().item() >= threshold:
                slices_2p5d.append(slice_2p5d)

        if len(slices_2p5d) == 0:
            continue  # sujet vide

        slices_tensor = torch.stack(slices_2p5d)  # [num_slices, 3, H, W]
        label_tensor = torch.tensor(label, dtype=torch.long)

        # Sauvegarde
        torch.save((slices_tensor, label_tensor), os.path.join(split_dir, f"{subject_id}.pt"))
