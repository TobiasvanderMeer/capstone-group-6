import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import glob
import os
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping


batch_size = 8
epochs = 20
test_size = 0.1
n = 60


path_data = r"C:\Users\bramv\Downloads\capstone-group-6\datasets"


postfix = ["0","_1000to1050","_1050to1400",  "_1400to2000", "_2000to3000", 
           "_3000to4000", "_4000to5000", "_5000to6000", "_6000to7000", "_7000to8000"]


k_all = []
h_all = []

for p in postfix:
    # Construct file names (with .txt)
    k_file = os.path.join(path_data, f"k_set{p}.txt")
    h_file = os.path.join(path_data, f"h_set{p}.txt")
    
    # Load the k file
    if os.path.exists(k_file):
        k_data = np.loadtxt(k_file)
        # Assuming each row is a 3600-long array, otherwise reshape
        if k_data.ndim == 1:  # single array
            k_all.append(k_data)
        else:  # multiple arrays
            k_all.extend(k_data)
    else:
        print(f"Warning: {k_file} not found.")
    
    # Load the h file
    if os.path.exists(h_file):
        h_data = np.loadtxt(h_file)
        if h_data.ndim == 1:
            h_all.append(h_data)
        else:
            h_all.extend(h_data)
    else:
        print(f"Warning: {h_file} not found.")
k_data = np.array(k_all).reshape(-1, 60, 60, 1)
h_data = np.array(h_all).reshape(-1, 60, 60, 1)


# Pick 2 random samples from the dataset
num_samples = 2
indices = np.random.choice(len(k_data), size=num_samples, replace=False)

plt.figure(figsize=(10, num_samples * 5))

for i, idx in enumerate(indices):
    k_sample = k_data[idx, :, :, 0]
    h_sample = h_data[idx, :, :, 0]

    # Plot k
    plt.subplot(num_samples, 2, i*2 + 1)
    plt.imshow(k_sample, cmap='viridis')
    plt.title(f"Sample {idx} - k")
    plt.colorbar(fraction=0.046, pad=0.04)

    # Plot h with contour
    plt.subplot(num_samples, 2, i*2 + 2)
    plt.imshow(h_sample, cmap='viridis')
    plt.contour(h_sample, colors='white', linewidths=0.8)
    plt.title(f"Sample {idx} - h (true)")
    plt.colorbar(fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()


def unet_model(input_shape):
    inputs = layers.Input(shape=input_shape)

    # Encoder
    def conv_block(x, filters):
        x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        return x

    c1 = conv_block(inputs, 16)
    p1 = layers.MaxPooling2D((2,2))(c1)

    c2 = conv_block(p1, 32)
    p2 = layers.MaxPooling2D((2,2))(c2)

    # Bottleneck
    c3 = conv_block(p2, 64)
    c3 = layers.Dropout(0.3)(c3)

    # Decoder
    def up_block(x, skip, filters):
        x = layers.Conv2DTranspose(filters, 2, strides=2, padding='same')(x)
        x = layers.concatenate([x, skip])
        x = conv_block(x, filters)
        return x

    u2 = up_block(c3, c2, 32)
    u1 = up_block(u2, c1, 16)

    outputs = layers.Conv2D(1, 1, activation='linear')(u1)
    model = models.Model(inputs, outputs)
    return model

# Build model
model = unet_model((n, n, 1))
model.compile(optimizer='adam', loss='mse', metrics=['mse'])
model.summary()


X_train, X_test, y_train, y_test = train_test_split(
    k_data, h_data, test_size=test_size, random_state=42
)


callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1),
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
]

history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    batch_size=batch_size,
    epochs=10,
    callbacks=callbacks
)



num_samples = 4
plt.figure(figsize=(15, num_samples * 4))

for i in range(num_samples):
    k_sample = X_test[i, :, :, 0]      # 2D grid
    h_true = y_test[i, :, :, 0]        # 2D grid
    h_pred = model.predict(X_test[i:i+1], verbose=0)[0, :, :, 0]  # 2D grid

    # Plot k
    plt.subplot(num_samples, 3, i*3 + 1)
    plt.title(f"Sample {i} - k")
    plt.imshow(k_sample, cmap='viridis')
    plt.colorbar(fraction=0.046, pad=0.04)

    # Plot true h with contour
    plt.subplot(num_samples, 3, i*3 + 2)
    plt.title(f"Sample {i} - True h")
    plt.imshow(h_true, cmap='viridis')
    plt.contour(h_true, colors='white', linewidths=0.8)  # contours
    plt.colorbar(fraction=0.046, pad=0.04)

    # Plot predicted h with contour
    plt.subplot(num_samples, 3, i*3 + 3)
    plt.title(f"Sample {i} - Predicted h")
    plt.imshow(h_pred, cmap='viridis')
    plt.contour(h_pred, colors='white', linewidths=0.8)  # contours
    plt.colorbar(fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()




