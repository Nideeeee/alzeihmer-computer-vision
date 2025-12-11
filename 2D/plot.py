import numpy as np
import matplotlib.pyplot as plt
import os
dataset_path = 'adni-masked/train_dataset_2D'
files = sorted(os.listdir(dataset_path))[:21]
fig, axes = plt.subplots(3, 7, figsize=(20, 10))
axes = axes.flatten()
for idx, filename in enumerate(files):
    filepath = os.path.join(dataset_path, filename)
    img = np.load(filepath)

    axes[idx].imshow(img, cmap='gray')
    axes[idx].set_title(filename.split('_')[-2] + '_' + filename.split('_')[-1].replace('.npy', ''),
                        fontsize=8)
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig('2D/first_21_images.png', dpi=150, bbox_inches='tight')
plt.show()
