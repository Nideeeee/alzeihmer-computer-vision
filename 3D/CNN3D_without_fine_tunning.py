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
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.manifold import TSNE
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

import os

# 1. CONFIGURATION
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

# 2. DÉFINITION DU MODÈLE

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
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# 3. CHARGEMENT DONNÉES

x_train = np.load('datas/x_train_hippocampi_40.npy')
y_train = np.load('datas/y_train_hippocampi_40.npy')
x_test = np.load('datas/x_test_hippocampi_40.npy')
y_test = np.load('datas/y_test_hippocampi_40.npy')

# Normalisation par sujet (train)
mean_train = x_train.mean(axis=(1, 2, 3), keepdims=True)
std_train = x_train.std(axis=(1, 2, 3), keepdims=True) + 1e-5
x_train = (x_train - mean_train) / std_train

# Normalisation par sujet (test)
mean_test = x_test.mean(axis=(1, 2, 3), keepdims=True)
std_test = x_test.std(axis=(1, 2, 3), keepdims=True) + 1e-5
x_test = (x_test - mean_test) / std_test

# Ajout de la dimension canal
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

input_shape = (80, 40, 40, 1)
num_classes = 2 

# 4. ENTRAÎNEMENT DU CNN3D (AD vs CN)

model = build_cnn3d_model(input_shape, num_classes)
weights_path = "cnn3d_initial_weights.weights.h5"

# Mélange + split pour validation
indices = np.arange(len(x_train))
np.random.shuffle(indices)
x_train_shuff, y_train_shuff = x_train[indices], y_train[indices]
val_split = 0.2
val_size = int(len(x_train_shuff) * val_split)
history = model.fit(
    x_train_shuff[val_size:], y_train_shuff[val_size:],
    epochs=100,
    batch_size=8,
    validation_data=(x_train_shuff[:val_size], y_train_shuff[:val_size]),
    verbose=1
)
model.save_weights(weights_path)
print("Entraînement terminé et poids sauvegardés.")
# Courbes d'apprentissage
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(1, len(acc) + 1)
plt.figure(figsize=(12, 5))
# Accuracy
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Train Accuracy')
plt.plot(epochs_range, val_acc, label='Val Accuracy')
plt.title('Accuracy (Entraînement CNN3D)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
# Loss
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Train Loss')
plt.plot(epochs_range, val_loss, label='Val Loss')
plt.title('Loss (Entraînement CNN3D)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.savefig("learning_curves_cnn3d.png")
plt.show()

# 5. ÉVALUATION SUR LE JEU DE TEST
print("\n=== Évaluation sur le jeu de test ===")
test_loss, test_acc = model.evaluate(x_test, y_test, batch_size=16, verbose=1)
print(f"Test accuracy : {test_acc:.4f}")
print(f"Test loss     : {test_loss:.4f}")

# 6. RÉSULTATS & VISUALISATIONS
print("\n=== Génération des résultats et visualisations ===")

# Prédictions
y_pred_prob = model.predict(x_test, batch_size=16)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = y_test  # sparse labels 0 / 1

# Noms de classes à adapter à ton codage réel (0 -> CN, 1 -> AD par ex.)
class_names = ['Classe 0', 'Classe 1']  # remplace éventuellement par ['CN', 'AD']

# Matrice de confusion
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    cbar=False,
    xticklabels=class_names,
    yticklabels=class_names
)
plt.title('Matrice de confusion CNN3D')
plt.ylabel('Réel')
plt.xlabel('Prédit')
plt.savefig("matrice_confusion_cnn3d.png")
plt.show()

# 7. t-SNE / UMAP
feature_extractor = Model(inputs=model.inputs, outputs=model.layers[-2].output)
embeddings = feature_extractor.predict(x_test, batch_size=16)

y_labels = [class_names[y] for y in y_true]
palette = {class_names[0]: "dodgerblue", class_names[1]: "crimson"}

fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
X_tsne = tsne.fit_transform(embeddings)
sns.scatterplot(
    x=X_tsne[:, 0], y=X_tsne[:, 1],
    hue=y_labels, palette=palette,
    ax=axes[0], s=60
)
axes[0].set_title('t-SNE des embeddings CNN3D')

# UMAP
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
X_umap = reducer.fit_transform(embeddings)
sns.scatterplot(
    x=X_umap[:, 0], y=X_umap[:, 1],
    hue=y_labels, palette=palette,
    ax=axes[1], s=60
)
axes[1].set_title('UMAP des embeddings CNN3D')

plt.tight_layout()
plt.savefig("visualisation_cnn_embeddings.png")
plt.show()