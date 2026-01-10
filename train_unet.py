# ============================================================
# U-Net training script
# k (input) -> h (output)
# ============================================================

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import train_test_split


# ======================
# CONFIGURATIE
# ======================
IMG_SIZE = 60
BATCH_SIZE = 8
EPOCHS = 50
TEST_SIZE = 0.1
RANDOM_STATE = 42

PATH_DATA = r"C:\Users\jorri\GitHub\capstone-group-6\datasets"

POSTFIXES = [
    "0","_1000to1050","_1050to1400","_1400to2000",
    "_2000to3000","_3000to4000","_4000to5000",
    "_5000to6000","_6000to7000","_7000to8000"
]


# ======================
# DATA LOADING
# ======================
def load_dataset(path_data, postfixes):
    k_all, h_all = [], []

    for p in postfixes:
        k_file = os.path.join(path_data, f"k_set{p}.txt")
        h_file = os.path.join(path_data, f"h_set{p}.txt")

        if os.path.exists(k_file):
            k = np.loadtxt(k_file)
            k_all.extend(k if k.ndim > 1 else [k])
        else:
            print(f"⚠️ Missing: {k_file}")

        if os.path.exists(h_file):
            h = np.loadtxt(h_file)
            h_all.extend(h if h.ndim > 1 else [h])
        else:
            print(f"⚠️ Missing: {h_file}")

    k_data = np.array(k_all).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    h_data = np.array(h_all).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

    return k_data, h_data


# ======================
# NORMALISATIE
# ======================
def normalize(x):
    return (x - x.mean()) / x.std()


# ======================
# U-NET MODEL
# ======================
def conv_block(x, filters):
    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    return x


def build_unet(input_shape):
    inputs = layers.Input(shape=input_shape)

    # Encoder
    c1 = conv_block(inputs, 32)
    p1 = layers.MaxPooling2D()(c1)

    c2 = conv_block(p1, 64)
    p2 = layers.MaxPooling2D()(c2)

    # Bottleneck
    b = conv_block(p2, 128)
    b = layers.Dropout(0.3)(b)

    # Decoder
    u2 = layers.Conv2DTranspose(64, 2, strides=2, padding="same")(b)
    u2 = layers.Concatenate()([u2, c2])
    u2 = conv_block(u2, 64)

    u1 = layers.Conv2DTranspose(32, 2, strides=2, padding="same")(u2)
    u1 = layers.Concatenate()([u1, c1])
    u1 = conv_block(u1, 32)

    outputs = layers.Conv2D(1, 1, activation="linear")(u1)

    return models.Model(inputs, outputs)


# ======================
# VISUALISATIE
# ======================
def plot_prediction(model, X, y, idx):
    k = X[idx, :, :, 0]
    h_true = y[idx, :, :, 0]
    h_pred = model.predict(X[idx:idx+1], verbose=0)[0, :, :, 0]

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title("k (input)")
    plt.imshow(k)
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.title("True h")
    plt.imshow(h_true)
    plt.contour(h_true, colors="white")
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.title("Predicted h")
    plt.imshow(h_pred)
    plt.contour(h_pred, colors="white")
    plt.colorbar()

    plt.tight_layout()
    plt.show()


# ======================
# MAIN
# ======================
def main():
    print("📥 Loading data...")
    k_data, h_data = load_dataset(PATH_DATA, POSTFIXES)

    print("📏 Normalizing...")
    k_data[:] = normalize(k_data)
    h_data[:] = normalize(h_data)

    print("🔀 Train/test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        k_data, h_data, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    print("🧠 Building model...")
    model = build_unet((IMG_SIZE, IMG_SIZE, 1))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="mse",
        metrics=["mse", "mae"]
    )
    model.summary()

    callbacks = [
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, verbose=1),
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    ]

    print("🚀 Training...")
    model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks
    )

    print("📊 Plotting predictions...")
    for i in range(3):
        plot_prediction(model, X_test, y_test, i)


if __name__ == "__main__":
    main()
