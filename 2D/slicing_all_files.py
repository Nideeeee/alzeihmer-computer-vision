import os
import numpy as np
from pathlib import Path
from slicing import load_nifti_image, extract_slices

def slice_3d_images_to_2d(input_dir, output_dir, num_slices_per_plane=7): #permet de slicer tous les fichiers
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    nifti_files = list(input_path.glob("*.nii.gz"))

    total_slices = 0
    failed_files = []

    for idx, nifti_file in enumerate(nifti_files, 1):
        print(f"[{idx}/{len(nifti_files)}] Traitement de {nifti_file.name}...", end=' ')
        volume = load_nifti_image(str(nifti_file))
        slices, indices = extract_slices(volume, num_slices_per_plane=num_slices_per_plane)
        base_name = nifti_file.stem.replace('.nii', '')
        slice_count = 0
        for plane_name in ['axial', 'coronal', 'sagittal']:
            for slice_idx, slice_data in enumerate(slices[plane_name]):
                #fichier de sortie: basename_plane_index.npy
                output_filename = f"{base_name}_{plane_name}_{slice_idx}.npy"
                output_filepath = output_path / output_filename
                #save la coupe au format numpy
                np.save(output_filepath, slice_data)
                slice_count += 1

        total_slices += slice_count

    return {
        'total_images': len(nifti_files) - len(failed_files),
        'total_slices': total_slices,
        'failed': failed_files
    }


if __name__ == "__main__":
    #répertoires d'entrée et de sortie
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
        print(f"Entrée  : {dataset['input']}")
        print(f"Sortie  : {dataset['output']}")
        print()

        result = slice_3d_images_to_2d(
            input_dir=dataset['input'],
            output_dir=dataset['output'],
            num_slices_per_plane=7
        )

        results[dataset['name']] = result

    total_images = sum(r['total_images'] for r in results.values())
    total_slices = sum(r['total_slices'] for r in results.values())
    total_failed = sum(len(r['failed']) for r in results.values())