from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling3D, SpatialDropout3D
import tensorflow as tf
import numpy as np
from sklearn.model_selection import KFold

print("\n=== Configuration GPU ===")
print(f"TensorFlow construit avec CUDA: {tf.test.is_built_with_cuda()}")
gpus = tf.config.list_physical_devices('GPU')
print(f"GPUs physiques disponibles: {gpus}")

if gpus:
    try:
        # Configuration de la croissance mémoire pour éviter d'allouer toute la mémoire GPU d'un coup
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"GPUs logiques: {logical_gpus}")
        print(f"\n✓ GPU détecté et configuré! TensorFlow utilisera automatiquement le GPU pour l'entraînement.")
    except RuntimeError as e:
        print(f"Erreur lors de la configuration du GPU: {e}")
else:
    print("\n⚠ Aucun GPU détecté. TensorFlow utilisera le CPU.")
print("========================\n")

def augment_3d_volume(volume):
    """
    Augmentation 3D sécurisée pour structures anatomiques.
    Pas de flip ni rotation.
    """
    # Ajout de bruit léger
    if np.random.rand() > 0.7:
        noise = np.random.normal(0, 0.01, size=volume.shape)
        volume = volume + noise

    # Déplacement léger
    shift = np.random.randint(-1, 2, size=3)  # -1, 0, +1 voxel
    volume = tf.roll(volume, shift=shift, axis=(0,1,2)).numpy()

    return volume

def build_cnn3d_model(input_shape, num_classes):
    """
    Construit un modèle CNN 3D pour la classification d'images médicales.

    Parameters:
    - input_shape : tuple, la forme des données d'entrée (profondeur, hauteur, largeur, canaux)
    - num_classes : int, le nombre de classes pour la classification

    Returns:
    - model : un modèle Keras CNN 3D compilé
    """
    model = Sequential()

    # Première couche de convolution 3D
    model.add(Conv3D(filters=16, kernel_size=(3, 3, 3), activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(SpatialDropout3D(0.2))  # Dropout dans les features maps
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))

    # Deuxième couche de convolution 3D
    model.add(Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(SpatialDropout3D(0.2))  # Dropout dans les features maps
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))

    # Troisième couche de convolution 3D
    model.add(Conv3D(filters=64, kernel_size=(3, 3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(SpatialDropout3D(0.2))  # Dropout dans les features maps
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))

    # Aplatir les données pour les couches denses
    model.add(GlobalAveragePooling3D())

    # Couche dense avec dropout pour éviter le surapprentissage
    model.add(Dense(units=256, activation='relu'))
    #model.add(BatchNormalization())
    model.add(Dropout(rate=0.5))

    # Couche de sortie
    model.add(Dense(units=num_classes, activation='softmax'))

    # Compiler le modèle
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)  # ou 5e-5
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

x_train = np.load('datas/x_train_hippocampi_40.npy')
y_train = np.load('datas/y_train_hippocampi_40.npy')
x_test = np.load('datas/x_test_hippocampi_40.npy')
y_test = np.load('datas/y_test_hippocampi_40.npy')
# x_train = np.load('datas/x_train_hippocampi.npy')
# y_train = np.load('datas/y_train_hippocampi.npy')
# x_test = np.load('datas/x_test_hippocampi.npy')
# y_test = np.load('datas/y_test_hippocampi.npy')

x_train = (x_train - x_train.mean(axis=(1,2,3), keepdims=True)) / (x_train.std(axis=(1,2,3), keepdims=True) + 1e-5)
x_test = (x_test - x_test.mean(axis=(1,2,3), keepdims=True)) / (x_test.std(axis=(1,2,3), keepdims=True) + 1e-5)

x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

print(x_train.shape)

# x_train = augment_3d_volume(x_train)

# --- Étape 1 : Entraînement sur AD/CN ---
input_shape = (80, 40, 40, 1)
num_classes = 2

# Crée le modèle CNN 3D
model = build_cnn3d_model(input_shape=input_shape, num_classes=num_classes)

# Fractionnement validation interne sur X_train
indices = np.arange(len(x_train))
np.random.shuffle(indices)

x_train = x_train[indices]
y_train = y_train[indices]

# --- Fractionnement validation interne ---
val_split = 0.2
val_size = int(len(x_train) * val_split)

X_train_adcn = x_train[val_size:]
y_train_adcn = y_train[val_size:]
X_val_adcn = x_train[:val_size]
y_val_adcn = y_train[:val_size]

# Entraînement initial
history = model.fit(
    X_train_adcn, y_train_adcn,
    epochs=20,
    batch_size=8,
    validation_data=(X_val_adcn, y_val_adcn)
)

indices_test = np.arange(len(x_test))
np.random.shuffle(indices_test)

x_test = x_test[indices_test]
y_test = y_test[indices_test]

# --- Étape 2 : Fine-tuning pour MCI ---
# On freeze les couches convolutives
for layer in model.layers:
    if isinstance(layer, Conv3D) or isinstance(layer, MaxPooling3D) or isinstance(layer, BatchNormalization) or isinstance(layer, GlobalAveragePooling3D) or isinstance(layer, SpatialDropout3D):
        layer.trainable = False

# Vérifie la tête
print("\nCouches gelées, seules les Dense finales sont entraînables:")
for layer in model.layers:
    print(layer.name, layer.trainable)

# Boucle sur les couches
for layer in model.layers:
    if isinstance(layer, Dense):
        # Réinitialisation des poids et biais
        old_shape = layer.get_weights()[0].shape
        old_bias_shape = layer.get_weights()[1].shape
        layer.set_weights([
            np.random.randn(*old_shape) * 0.05,
            np.zeros(old_bias_shape)
        ])

n_splits = 5  # 5-fold CV
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

fold_histories = []

# Recompilation du modèle après freeze
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# K-Fold training
for fold, (train_idx, val_idx) in enumerate(kf.split(x_test)):
    print(f"\n--- Fold {fold+1}/{n_splits} ---")

    X_train_fold = x_test[train_idx]
    y_train_fold = y_test[train_idx]
    X_val_fold = x_test[val_idx]
    y_val_fold = y_test[val_idx]

    history = model.fit(
        X_train_fold, y_train_fold,
        epochs=10,
        batch_size=4,
        validation_data=(X_val_fold, y_val_fold),
        verbose=2
    )

    fold_histories.append(history)

# Optionnel : calculer la moyenne des metrics sur les folds
val_accuracies = [max(h.history['val_accuracy']) for h in fold_histories]
print(f"\nMoyenne des val_accuracy sur {n_splits} folds : {np.mean(val_accuracies):.4f}")