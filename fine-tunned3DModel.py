import tensorflow as tf
from tfm.vision.backbones import ResNet3D  # ou selon version/chemin

# Définir le backbone
backbone = ResNet3D(
    model_id=50,
    include_top=False,
    input_shape=(D, H, W, C),
    weights=None  # on verra plus loin pour les poids
)

inputs = tf.keras.Input(shape=(D, H, W, C))
x = backbone(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling3D()(x)
x = tf.keras.layers.Dropout(0.5)(x)
outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)