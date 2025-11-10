from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout, BatchNormalization

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
    model.add(Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))

    # Deuxième couche de convolution 3D
    model.add(Conv3D(filters=64, kernel_size=(3, 3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))

    # Troisième couche de convolution 3D
    model.add(Conv3D(filters=128, kernel_size=(3, 3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))

    # Aplatir les données pour les couches denses
    model.add(Flatten())

    # Couche dense avec dropout pour éviter le surapprentissage
    model.add(Dense(units=256, activation='relu'))
    model.add(Dropout(rate=0.5))

    # Couche de sortie
    model.add(Dense(units=num_classes, activation='softmax'))

    # Compiler le modèle
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

model = build_cnn3d_model(input_shape=(64, 64, 64, 1), num_classes=2)
model.summary()
model.fit(x_train, y_train, epochs=10, batch_size=8, validation_data=(x_val, y_val))