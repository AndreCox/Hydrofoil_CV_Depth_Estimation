import io
import tensorflow as tf
import numpy as np
import h5py
from tensorflow.keras import layers, models, regularizers
from PIL import Image

# ── GPU setup ─────────────────────────────────────────────────────────────────
physical_devices = tf.config.list_physical_devices("GPU")
for dev in physical_devices:
    try:
        tf.config.experimental.set_memory_growth(dev, True)
    except RuntimeError:
        pass  # Device already initialised — must be set before init


# ── Config ────────────────────────────────────────────────────────────────────
HDF5_PATH  = '../hydrofoil_webp.hdf5'

# Lower batch size directly reduces VRAM proportionally
BATCH_SIZE = 8  
EPOCHS     = 20


# ── Data loading ──────────────────────────────────────────────────────────────

class HydroWebpH5Loader:
    """Streams WebP-encoded images and labels from an HDF5 file."""

    def __init__(self, h5_path, img_key="colors_webp", y_key="ride_height"):
        self.h5_path = h5_path
        self.img_key = img_key
        self.y_key   = y_key

        with h5py.File(self.h5_path, "r") as f:
            self.length = f[self.y_key].shape[0]
            # Detect native image size from the first sample — no resizing needed
            first_bytes = bytes(f[self.img_key][0])
            first_img   = Image.open(io.BytesIO(first_bytes)).convert("RGB")
            self.img_w, self.img_h = first_img.size  # PIL gives (W, H)

    @property
    def img_shape(self):
        """Returns (H, W, C) — the native render resolution."""
        return (self.img_h, self.img_w, 3)

    def __len__(self):
        return self.length


def get_split_datasets(h5_path, batch_size=BATCH_SIZE, split_ratio=0.8):
    """
    Returns (train_ds, val_ds, img_shape).

    Images are decoded at their native resolution — no resizing, no data loss.
    Since all renders come from the same Blender scene they share one fixed size,
    so batching works without any padding or cropping.
    """
    loader = HydroWebpH5Loader(h5_path)
    print(f"Native image shape: {loader.img_shape}")

    # Reproducible train/val split
    rng      = np.random.default_rng(seed=42)
    indices  = rng.permutation(loader.length)
    split_at = int(loader.length * split_ratio)

    train_indices = indices[:split_at]
    val_indices   = indices[split_at:]

    def make_ds(indices_list: np.ndarray, shuffle: bool):
        idx = indices_list.copy()

        def indexed_gen():
            with h5py.File(loader.h5_path, "r") as f:
                images = f[loader.img_key]
                labels = f[loader.y_key]
                for i in idx:
                    raw = bytes(images[i])
                    img = Image.open(io.BytesIO(raw)).convert("RGB")
                    img = np.array(img, dtype=np.float32) / 255.0
                    yield img, np.array([float(labels[i])], dtype=np.float32)

        ds = tf.data.Dataset.from_generator(
            indexed_gen,
            output_signature=(
                tf.TensorSpec(shape=loader.img_shape, dtype=tf.float32),
                tf.TensorSpec(shape=(1,),             dtype=tf.float32),
            ),
        )
        
        # Adding .repeat() prevents the generator from exhausting prematurely
        ds = ds.repeat()

        if shuffle:
            ds = ds.shuffle(
                buffer_size=min(2048, len(indices_list)),
                reshuffle_each_iteration=True,
            )

        return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE), len(indices_list) // batch_size

    train_ds, train_steps = make_ds(train_indices, shuffle=True)
    val_ds, val_steps     = make_ds(val_indices,   shuffle=False)
    
    return train_ds, val_ds, loader.img_shape, train_steps, val_steps


# ── Model ─────────────────────────────────────────────────────────────────────

def residual_block(x, filters, downsample=False, reg=None):
    """A lightweight residual block."""
    stride = 2 if downsample else 1

    # Main path
    y = layers.Conv2D(filters, 3, strides=stride, padding="same",
                      use_bias=False, kernel_regularizer=reg)(x)
    y = layers.BatchNormalization()(y)
    y = layers.Activation("relu")(y)
    y = layers.Conv2D(filters, 3, strides=1, padding="same",
                      use_bias=False, kernel_regularizer=reg)(y)
    y = layers.BatchNormalization()(y)

    # Skip connection — match dimensions if needed
    if downsample or x.shape[-1] != filters:
        x = layers.Conv2D(filters, 1, strides=stride, padding="same",
                          use_bias=False, kernel_regularizer=reg)(x)
        x = layers.BatchNormalization()(x)

    return layers.Activation("relu")(layers.Add()([x, y]))


def build_model(input_shape):
    reg = regularizers.l2(1e-4)
    inputs = layers.Input(shape=input_shape)

    # Stem: modest initial downsample
    x = layers.Conv2D(32, 7, strides=2, padding="same",
                      use_bias=False, kernel_regularizer=reg)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

    # Residual stages — progressively deeper, filters double each stage
    x = residual_block(x, 32, reg=reg)
    x = residual_block(x, 32, reg=reg)

    x = residual_block(x, 64, downsample=True, reg=reg)
    x = residual_block(x, 64, reg=reg)

    x = residual_block(x, 128, downsample=True, reg=reg)
    x = residual_block(x, 128, reg=reg)

    x = residual_block(x, 256, downsample=True, reg=reg)
    x = residual_block(x, 256, reg=reg)

    # Head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=reg)(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="linear")(x)

    model = models.Model(inputs, outputs)

    # Fix the LR schedule — decay over ALL training steps
    total_steps = (EPOCHS * 320)  # train_steps * epochs — adjust 320 to your train_steps
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=1e-3,
        decay_steps=total_steps,
        alpha=1e-6,
    )

    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=1e-4),
        loss=tf.keras.losses.Huber(delta=0.1),  # tune delta to your label scale
        metrics=["mae"],
    )
    return model

# ── Training ──────────────────────────────────────────────────────────────────

train_ds, val_ds, img_shape, train_steps, val_steps = get_split_datasets(HDF5_PATH, batch_size=BATCH_SIZE)

model = build_model(input_shape=img_shape)
model.summary()

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_mae", patience=5, restore_best_weights=True
    ),
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    steps_per_epoch=train_steps,
    validation_steps=val_steps,
    epochs=EPOCHS,
    callbacks=callbacks,
)