import os
import shutil
import nibabel as nib
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# ==========================================
# 1. CONFIGURATION
# ==========================================
LABEL_MAP = {"CN": 0, "AD": 1}

IMAGES_DIR = "../../adni"        
MASKS_DIR = "../../adni"         
CSV_FILE = "../../adni/list_standardized_tongtong_2017.csv"
CACHE_DIR = "slices_cache_filtered"

THRESHOLD = 0.001   
STRIDE = 3          
CROP_MARGIN = 0.15  

# ==========================================
# 2. NETTOYAGE
# ==========================================
if os.path.exists(CACHE_DIR):
    print(f"🗑️ Nettoyage du dossier '{CACHE_DIR}'...")
    shutil.rmtree(CACHE_DIR)
os.makedirs(CACHE_DIR, exist_ok=True)

# ==========================================
# 3. FILTRAGE DES DONNÉES (CORRECTION)
# ==========================================
print(f"Lecture du CSV...")
df = pd.read_csv(CSV_FILE, header=None)
df.columns = ["SubjectID", "RoosterID", "Age", "Sex", "Group", "Conversion",
              "MMSE", "RAVLT", "FAQ", "CDR_SB", "ADAS11"]

# --- CN/AD ---
cn_ad_df = df[df['Group'].isin(['CN', 'AD'])].copy()

# --- MCI : ON FILTRE LES INCONNUS (5) ---
mci_df = df[df['Group'] == 'MCI'].copy()

print(f"MCI Total avant filtrage : {len(mci_df)}")
# On garde uniquement 3 (Converter) et 4 (Stable)
# On supprime 5 (Inconnu) et -1 ou autres s'il y en a
mci_clean = mci_df[mci_df['Conversion'].isin([3, 4])].copy()

# Création du Label : 4 -> 0 (Stable), 3 -> 1 (Converter)
mci_clean['Label'] = mci_clean['Conversion'].apply(lambda x: 0 if x == 4 else 1)

print(f"MCI Nettoyé (sans inconnus) : {len(mci_clean)}")
print(f" -> Stables (4) : {len(mci_clean[mci_clean['Label']==0])}")
print(f" -> Converters (3) : {len(mci_clean[mci_clean['Label']==1])}")

# ==========================================
# 4. SPLITS STRATIFIÉS
# ==========================================
print("\n--- GÉNÉRATION DES SPLITS ---")

# CN/AD
train_cn_ad, temp_cn_ad = train_test_split(
    cn_ad_df, test_size=0.3, random_state=42, stratify=cn_ad_df['Group']
)
valid_cn_ad, test_cn_ad = train_test_split(
    temp_cn_ad, test_size=0.5, random_state=42, stratify=temp_cn_ad['Group']
)

# MCI (Sur le dataset PROPRE mci_clean)
valid_mci, test_mci = train_test_split(
    mci_clean, test_size=0.5, random_state=42, stratify=mci_clean['Label']
)

# Train MCI = Tout le CN/AD
train_mci_source = cn_ad_df 

# ==========================================
# 5. GÉNÉRATION DES FICHIERS
# ==========================================
def process_subjects(subject_df, split_name, out_root_dir):
    split_dir = os.path.join(out_root_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)
    
    count = 0
    for _, row in tqdm(subject_df.iterrows(), total=len(subject_df), desc=f"Writing {split_name}"):
        subject_id = row['SubjectID']
        
        if row['Group'] in LABEL_MAP: label = LABEL_MAP[row['Group']]
        elif 'Label' in row: label = row['Label']
        else: continue

        img_file = os.path.join(IMAGES_DIR, f"n_mmni_fADNI_{subject_id}_1.5T_t1w.nii.gz")
        mask_file = os.path.join(MASKS_DIR, f"mask_n_mmni_fADNI_{subject_id}_1.5T_t1w.nii.gz")
        
        if not os.path.exists(img_file): continue

        try:
            vol = nib.load(img_file).get_fdata()
            if os.path.exists(mask_file): vol = vol * nib.load(mask_file).get_fdata()
        except: continue

        v_min, v_max = vol.min(), vol.max()
        if v_max - v_min > 0: vol = (vol - v_min) / (v_max - v_min)
        vol = torch.tensor(vol, dtype=torch.float32)

        depth = vol.shape[0]
        start = max(1, int(depth * CROP_MARGIN))
        end = min(depth - 2, int(depth * (1 - CROP_MARGIN)))
        
        slices = []
        for i in range(start, end, STRIDE):
            s = vol[i-1:i+2, :, :]
            if s.mean() >= THRESHOLD: slices.append(s)
        
        if slices:
            torch.save((torch.stack(slices), torch.tensor(label, dtype=torch.long)), 
                       os.path.join(split_dir, f"{subject_id}.pt"))
            count += 1
    return count

print("\n--- ÉCRITURE ---")
process_subjects(train_cn_ad, "train", os.path.join(CACHE_DIR, "standard"))
process_subjects(valid_cn_ad, "valid", os.path.join(CACHE_DIR, "standard"))
process_subjects(test_cn_ad, "test", os.path.join(CACHE_DIR, "standard"))

process_subjects(train_mci_source, "train", os.path.join(CACHE_DIR, "mci"))
process_subjects(valid_mci, "valid", os.path.join(CACHE_DIR, "mci"))
process_subjects(test_mci, "test", os.path.join(CACHE_DIR, "mci"))

print("\n✅ Terminé ! Dataset MCI propre (Stables vs Converters uniquement).")