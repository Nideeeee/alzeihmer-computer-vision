import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path

def load_nifti_image(filepath):
    img = nib.load(filepath)
    data = img.get_fdata()
    return data

def extract_slices(volume, num_slices_per_plane=7):
    x_dim, y_dim, z_dim = volume.shape

    # On ignore 10% des bords pour chaque axe
    def centered_linspace(start, stop, num):
        return np.linspace(start, stop, num, dtype=int)

    margin_x = int(0.2 * x_dim)
    margin_y = int(0.2 * y_dim)
    margin_z = int(0.2 * z_dim)

    sagittal_indices = centered_linspace(margin_x, x_dim - margin_x - 1, num_slices_per_plane)
    coronal_indices = centered_linspace(margin_y, y_dim - margin_y - 1, num_slices_per_plane)
    axial_indices = centered_linspace(margin_z, z_dim - margin_z - 1, num_slices_per_plane)

    slices = {
        'axial': [volume[:, :, i] for i in axial_indices],
        'coronal': [volume[:, i, :] for i in coronal_indices],
        'sagittal': [volume[i, :, :] for i in sagittal_indices]
    }

    return slices, {
        'axial': axial_indices,
        'coronal': coronal_indices,
        'sagittal': sagittal_indices
    }

def visualize_slices(slices, indices, volume_shape):
    fig, axes = plt.subplots(3, len(slices['axial']), figsize=(15, 10))
    
    plane_names = ['Axial', 'Coronal', 'Sagittal']
    
    for i, (plane_name, plane_key) in enumerate(zip(plane_names, ['axial', 'coronal', 'sagittal'])):
        for j, (slice_data, idx) in enumerate(zip(slices[plane_key], indices[plane_key])):
            ax = axes[i, j]
            im = ax.imshow(slice_data.T, cmap='gray', origin='lower')
            ax.set_title(f'{plane_name}\nSlice {idx}')
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('slices_overview.png', dpi=150, bbox_inches='tight')
    print("✓ Visualisation sauvegardée : slices_overview.png")
    plt.show()

def print_slice_info(volume, slices, indices):
    """
    Affiche les informations sur les coupes extraites
    """
    print("\n" + "="*60)
    print("INFORMATIONS SUR L'IMAGE 3D")
    print("="*60)
    print(f"Dimensions du volume : {volume.shape} (X×Y×Z)")
    print(f"Taille voxel : {volume.dtype}")
    print(f"Range des valeurs : [{volume.min():.2f}, {volume.max():.2f}]")
    
    print("\n" + "="*60)
    print("COUPES EXTRAITES")
    print("="*60)
    
    for plane_name in ['axial', 'coronal', 'sagittal']:
        print(f"\n{plane_name.upper()}:")
        print(f"  Nombre de coupes : {len(slices[plane_name])}")
        print(f"  Indices : {indices[plane_name]}")
        print(f"  Taille de chaque coupe : {slices[plane_name][0].shape}")

if __name__ == "__main__":

    filepath = "adni/adni-masked/masked_n_mmni_fADNI_941_S_1363_1.5T_t1w.nii.gz" 
    
    volume = load_nifti_image(filepath)

    slices_regular, indices_regular = extract_slices(volume, num_slices_per_plane=7)
    print_slice_info(volume, slices_regular, indices_regular)
    visualize_slices(slices_regular, indices_regular, volume.shape)
