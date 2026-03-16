import numpy as np, tensorflow as tf
from tensorflow.keras import layers, models, regularizers
reg=regularizers.l2(1e-4)

model = models.Sequential([
    layers.Input(shape=(960, 720, 3)),
    layers.Conv2D(16, 5, strides=2, padding='same', use_bias=False, kernel_regularizer=reg),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(32, 3, padding='same', use_bias=False, kernel_regularizer=reg),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, 3, padding='same', use_bias=False, kernel_regularizer=reg),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, 3, padding='same', use_bias=False, kernel_regularizer=reg),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D((2, 2)),
    layers.GlobalAveragePooling2D(),
    layers.Dense(64, activation='relu', kernel_regularizer=reg),
    layers.Dropout(0.3),
    layers.Dense(1, activation='linear')
])

total_bytes = 0
batch_size = 8
input_shape = model.input_shape
elements = np.prod([dim for dim in input_shape if dim is not None])
total_bytes += (elements * 4 * batch_size)

current_shape = (batch_size, *input_shape[1:]) if input_shape[0] is None else input_shape
for layer in model.layers:
    try:
        current_shape = layer.compute_output_shape(current_shape)
        shape = current_shape[0] if isinstance(current_shape, list) else current_shape
        elements = np.prod([dim for dim in shape if dim is not None])
        total_bytes += (elements * 4 * batch_size)
    except Exception:
        pass
print(f"Activations MB: {(total_bytes * 2) / (1024**2):.1f}")
