import csv
import os
import shutil

#paths des dossiers
csv_file = 'adni-masked/list_standardized_tongtong_2017.csv'
train_dir = 'adni-masked/train_dataset'
valid_dir = 'adni-masked/valid_dataset'
source_dir = 'adni-masked'

# si jamais rien n'existe
os.makedirs(train_dir, exist_ok=True)
os.makedirs(valid_dir, exist_ok=True)

with open(csv_file, 'r') as f:
    csv_reader = csv.reader(f)
    for row in csv_reader:
        patient_id = row[0]  # ID du patient est dans la première colonne
        diagnosis = row[4]   # Le diagnostic est dans la 5ème colonne
        irm_filename = f'masked_n_mmni_fADNI_{patient_id}_1.5T_t1w.nii.gz'
        source_path = os.path.join(source_dir, irm_filename)
        if diagnosis in ['CN', 'AD']:
            dest_dir = train_dir
        else:
            dest_dir = valid_dir
        dest_path = os.path.join(dest_dir, irm_filename)
        if os.path.exists(source_path):
            try:
                shutil.move(source_path, dest_path)
                print(f'Déplacé {irm_filename} vers {os.path.basename(dest_dir)} (Diagnostic: {diagnosis})')
            except Exception as e:
                print(f'Erreur lors du déplacement de {irm_filename}: {str(e)}')
        else:
            print(f'Fichier non trouvé: {irm_filename}')