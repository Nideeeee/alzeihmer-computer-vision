import os
import numpy as np
from pathlib import Path
from slicing import load_nifti_image, extract_slices

def slice_3d_images_to_2d(input_dir, output_dir, num_slices_per_plane=7):
    """
    Slice toutes les images 3D d'un répertoire en 21 images 2D.

    Args:
        input_dir (str): Chemin du répertoire contenant les images 3D (.nii.gz)
        output_dir (str): Chemin du répertoire de sortie pour les images 2D
        num_slices_per_plane (int): Nombre de coupes par plan (défaut: 7)
                                     Total = 3 plans × num_slices_per_plane

    Returns:
        dict: Dictionnaire contenant les informations sur le traitement
              {'total_images': int, 'total_slices': int, 'failed': list}
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Trouver tous les fichiers .nii.gz
    nifti_files = list(input_path.glob("*.nii.gz"))

    if not nifti_files:
        print(f"Aucun fichier .nii.gz trouvé dans {input_dir}")
        return {'total_images': 0, 'total_slices': 0, 'failed': []}

    print(f"Trouvé {len(nifti_files)} images 3D à traiter...")

    total_slices = 0
    failed_files = []

    for idx, nifti_file in enumerate(nifti_files, 1):
        try:
            print(f"[{idx}/{len(nifti_files)}] Traitement de {nifti_file.name}...", end=' ')

            # Charger l'image 3D
            volume = load_nifti_image(str(nifti_file))

            # Extraire les coupes 2D
            slices, indices = extract_slices(volume, num_slices_per_plane=num_slices_per_plane)

            # Nom de base du fichier (sans extension)
            base_name = nifti_file.stem.replace('.nii', '')

            # Sauvegarder chaque coupe
            slice_count = 0
            for plane_name in ['axial', 'coronal', 'sagittal']:
                for slice_idx, slice_data in enumerate(slices[plane_name]):
                    # Nom du fichier de sortie: basename_plane_index.npy
                    output_filename = f"{base_name}_{plane_name}_{slice_idx}.npy"
                    output_filepath = output_path / output_filename

                    # Sauvegarder la coupe au format numpy
                    np.save(output_filepath, slice_data)
                    slice_count += 1

            total_slices += slice_count
            print(f"✓ {slice_count} coupes sauvegardées")

        except Exception as e:
            print(f"✗ Erreur: {str(e)}")
            failed_files.append((nifti_file.name, str(e)))

    # Résumé
    print("\n" + "="*60)
    print("RÉSUMÉ DU TRAITEMENT")
    print("="*60)
    print(f"Images 3D traitées : {len(nifti_files) - len(failed_files)}/{len(nifti_files)}")
    print(f"Coupes 2D créées : {total_slices}")
    print(f"Répertoire de sortie : {output_dir}")

    if failed_files:
        print("\nFichiers en échec :")
        for filename, error in failed_files:
            print(f"  - {filename}: {error}")

    return {
        'total_images': len(nifti_files) - len(failed_files),
        'total_slices': total_slices,
        'failed': failed_files
    }


if __name__ == "__main__":
    print("="*60)
    print("CONVERSION D'IMAGES 3D VERS 2D")
    print("="*60)

    # Définir les répertoires d'entrée et de sortie
    datasets = [
        {
            'input': 'adni-masked/train_dataset_3D',
            'output': 'adni-masked/train_dataset_2D',
            'name': 'TRAIN'
        },
        {
            'input': 'adni-masked/valid_dataset_3D',
            'output': 'adni-masked/valid_dataset_2D',
            'name': 'VALIDATION'
        }
    ]

    # Traiter chaque dataset
    results = {}
    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"TRAITEMENT DU DATASET {dataset['name']}")
        print(f"{'='*60}")
        print(f"Entrée  : {dataset['input']}")
        print(f"Sortie  : {dataset['output']}")
        print()

        result = slice_3d_images_to_2d(
            input_dir=dataset['input'],
            output_dir=dataset['output'],
            num_slices_per_plane=7
        )

        results[dataset['name']] = result

    # Afficher le résumé global
    print("\n" + "="*60)
    print("RÉSUMÉ GLOBAL")
    print("="*60)

    total_images = sum(r['total_images'] for r in results.values())
    total_slices = sum(r['total_slices'] for r in results.values())
    total_failed = sum(len(r['failed']) for r in results.values())

    print(f"Total d'images 3D traitées : {total_images}")
    print(f"Total de coupes 2D créées   : {total_slices}")
    print(f"Total d'échecs              : {total_failed}")

    for dataset_name, result in results.items():
        print(f"\n{dataset_name}:")
        print(f"  Images traitées : {result['total_images']}")
        print(f"  Coupes créées   : {result['total_slices']}")
        if result['failed']:
            print(f"  Échecs          : {len(result['failed'])}")

    print("\n" + "="*60)
    print("TRAITEMENT TERMINÉ")
    print("="*60)

