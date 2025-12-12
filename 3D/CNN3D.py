import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv3D, MaxPooling3D, Dense, Dropout, BatchNormalization,
    GlobalAveragePooling3D, SpatialDropout3D
)
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import KFold
from sklearn.manifold import TSNE
from sklearn.svm import SVC
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score
)
import os

from sklearn.linear_model import LogisticRegression

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU activé : {len(gpus)} détecté(s)")
    except RuntimeError as e:
        print(e)
else:
    print("CPU uniquement")

# 1. DÉFINITION DU MODÈLE
def build_cnn3d_model(input_shape, num_classes):
    model = Sequential()

    # Bloc 1
    model.add(Conv3D(16, (3, 3, 3), activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(SpatialDropout3D(0.2))
    model.add(MaxPooling3D((2, 2, 2)))

    # Bloc 2
    model.add(Conv3D(32, (3, 3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(SpatialDropout3D(0.2))
    model.add(MaxPooling3D((2, 2, 2)))

    # Bloc 3
    model.add(Conv3D(64, (3, 3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(SpatialDropout3D(0.2))
    model.add(MaxPooling3D((2, 2, 2)))

    # Tête de classification
    model.add(GlobalAveragePooling3D())
    
    model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.5))
    
    model.add(Dense(num_classes, activation='softmax'))

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# 2. CHARGEMENT DONNÉES

x_train = np.load('datas/x_train_hippocampi_40.npy')
y_train = np.load('datas/y_train_hippocampi_40.npy')
x_test = np.load('datas/x_test_hippocampi_40.npy')
y_test = np.load('datas/y_test_hippocampi_40.npy')

# Pour le Train
mean_train = x_train.mean(axis=(1, 2, 3), keepdims=True)
std_train = x_train.std(axis=(1, 2, 3), keepdims=True) + 1e-5
x_train = (x_train - mean_train) / std_train

# Pour le Test
mean_test = x_test.mean(axis=(1, 2, 3), keepdims=True)
std_test = x_test.std(axis=(1, 2, 3), keepdims=True) + 1e-5
x_test = (x_test - mean_test) / std_test 

x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# 3. PHASE 1 : PRÉ-ENTRAÎNEMENT AD/CN

input_shape = (80, 40, 40, 1)
model = build_cnn3d_model(input_shape, 2)
weights_path = "cnn3d_initial_weights.weights.h5"

# Mélange et split
indices = np.arange(len(x_train))
np.random.shuffle(indices)
x_train_shuff, y_train_shuff = x_train[indices], y_train[indices]
val_split = 0.2
val_size = int(len(x_train_shuff) * val_split)
history_phase1 = model.fit(
    x_train_shuff[val_size:], y_train_shuff[val_size:],
    epochs=100,
    batch_size=8,
    validation_data=(x_train_shuff[:val_size], y_train_shuff[:val_size]),
    verbose=1
)
model.save_weights(weights_path)
print(f"Entraînement terminé et poids sauvegardés.")
acc = history_phase1.history['accuracy']
val_acc = history_phase1.history['val_accuracy']
loss = history_phase1.history['loss']
val_loss = history_phase1.history['val_loss']
epochs_range = range(1, len(acc) + 1)
plt.figure(figsize=(12, 5))

# Plot Accuracy
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Train Accuracy')
plt.plot(epochs_range, val_acc, label='Val Accuracy')
plt.title('Phase 1: Accuracy (AD vs CN)')
plt.xlabel('Epochs')
plt.legend()

# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Train Loss')
plt.plot(epochs_range, val_loss, label='Val Loss')
plt.title('Phase 1: Loss (AD vs CN)')
plt.xlabel('Epochs')
plt.legend()

plt.tight_layout()
plt.show()
# 4. PHASE 2 : FINE-TUNING MCI

# Mélange du test set
indices_test = np.arange(len(x_test))
np.random.shuffle(indices_test)
x_test, y_test = x_test[indices_test], y_test[indices_test]

model.load_weights("cnn3d_initial_weights.weights.h5")

# Gel couches convolutionnelles
for layer in model.layers:
    layer.trainable = False # On gèle tout (Conv3D)
    if isinstance(layer, Dense) or isinstance(layer, Dropout):
        layer.trainable = True # On dégèle le classifieur

# Reset des poids de la couche Dense
for layer in model.layers:
    if isinstance(layer, Dense) and layer.units == 256:
        weights = layer.get_weights()
        if weights:
            layer.set_weights([
                np.random.randn(*weights[0].shape) * 0.05,
                np.zeros(weights[1].shape)
            ])

# Poids des classes
class_weights_dict = {0: 1.1764705882352942, 1: 0.8695652173913043}

# Compilation
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# K-Fold Training
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
fold_histories = []

for fold, (train_idx, val_idx) in enumerate(kf.split(x_test)):
    print(f"\n🔹 Fold {fold+1}/{n_splits}")

    history = model.fit(
        x_test[train_idx], y_test[train_idx],
        epochs=140,
        batch_size=16,
        validation_data=(x_test[val_idx], y_test[val_idx]),
        class_weight=class_weights_dict,
        verbose=1
    )
    fold_histories.append(history)

# Stats K-Fold
val_accuracies = [max(h.history['val_accuracy']) for h in fold_histories]
print(f"\nMoyenne des val_accuracy sur {n_splits} folds : {np.mean(val_accuracies):.4f}")

all_acc = [h.history['accuracy'] for h in fold_histories]
all_val_acc = [h.history['val_accuracy'] for h in fold_histories]
all_loss = [h.history['loss'] for h in fold_histories]
all_val_val_loss = [h.history['val_loss'] for h in fold_histories]

# Calcul de la moyenne
mean_acc = np.mean(all_acc, axis=0)
mean_val_acc = np.mean(all_val_acc, axis=0)
mean_loss = np.mean(all_loss, axis=0)
mean_val_loss = np.mean(all_val_val_loss, axis=0)

epochs_range = range(1, len(mean_acc) + 1)

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
# Affiche chaque pli
for i in range(len(all_acc)):
    plt.plot(epochs_range, all_acc[i], 'b-', alpha=0.15)
    plt.plot(epochs_range, all_val_acc[i], 'r-', alpha=0.15)
# Affiche la moyenne
plt.plot(epochs_range, mean_acc, 'b-o', linewidth=2, label='Mean Train Acc')
plt.plot(epochs_range, mean_val_acc, 'r-o', linewidth=2, label='Mean Val Acc')
plt.title(f'K-Fold Accuracy (Moyenne sur {n_splits} folds)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
# Affiche chaque pli
for i in range(len(all_loss)):
    plt.plot(epochs_range, all_loss[i], 'b-', alpha=0.15)
    plt.plot(epochs_range, all_val_val_loss[i], 'r-', alpha=0.15)
# Affiche la moyenne
plt.plot(epochs_range, mean_loss, 'b-o', linewidth=2, label='Mean Train Loss')
plt.plot(epochs_range, mean_val_loss, 'r-o', linewidth=2, label='Mean Val Loss')
plt.title(f'K-Fold Loss (Moyenne sur {n_splits} folds)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("kfold_learning_curves.png") # Sauvegarde auto
plt.show()
 
# 5. RÉSULTATS & VISUALISATIONS

# Prédictions
y_pred_prob = model.predict(x_test, batch_size=16)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = np.argmax(y_test, axis=1) if len(y_test.shape) > 1 else y_test
y_labels = ["pMCI" if y == 1 else "sMCI" for y in y_true]
my_palette = {"sMCI": "dodgerblue", "pMCI": "crimson"}

# Matrice de Confusion
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['sMCI (Stable)', 'pMCI (Progressif)'],
            yticklabels=['sMCI (Stable)', 'pMCI (Progressif)'])
plt.title('Matrice de Confusion')
plt.ylabel('Réel')
plt.xlabel('Prédit')
plt.savefig("matrice_confusion_cnn3d.png")
plt.show()

# t-SNE / UMAP

feature_extractor = Model(inputs=model.inputs, outputs=model.layers[-2].output)
embeddings = feature_extractor.predict(x_test, batch_size=16)

fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
X_tsne = tsne.fit_transform(embeddings)
sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y_labels, 
                palette=my_palette, ax=axes[0], s=60)
axes[0].set_title('t-SNE')
# UMAP
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
X_umap = reducer.fit_transform(embeddings)
sns.scatterplot(x=X_umap[:, 0], y=X_umap[:, 1], hue=y_labels, 
                palette=my_palette, ax=axes[1], s=60)
axes[1].set_title('UMAP')

plt.tight_layout()
plt.savefig("visualisation_cnn_embeddings.png")
plt.show()
