

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Load mnist
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
x_test  = x_test.reshape(-1, 784).astype('float32') / 255.0

X_all = np.concatenate([x_train, x_test], axis=0)

# Center
mean_vec = np.mean(X_all, axis=0, keepdims=True)
X_centered = X_all - mean_vec

# Create dataset
batch_size = 250
dataset = tf.data.Dataset.from_tensor_slices((X_centered, X_centered))
dataset = dataset.shuffle(buffer_size=X_centered.shape[0]).batch(batch_size)

# Autoencoder


input_dim = 784
encoder_input = layers.Input(shape=(input_dim,), name='encoder_input')
e1 = layers.Dense(512, activation='relu')(encoder_input)
e2 = layers.Dense(256, activation='relu')(e1)
latent = layers.Dense(128, activation='relu', name='encoder_output')(e2)

# decoder
d1 = layers.Dense(256, activation='relu')(latent)
d2 = layers.Dense(512, activation='relu')(d1)
decoder_output = layers.Dense(784, activation='sigmoid', name='decoder_output')(d2)


autoencoder = models.Model(inputs=encoder_input, outputs=decoder_output)


autoencoder.summary()

# train
autoencoder.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='binary_crossentropy'
)

epochs = 40
history = autoencoder.fit(
    dataset,
    epochs=epochs,
    verbose=1
)


print("\nΥπολογιζόμενος αριθμός trainable παραμέτρων:",
      np.sum([np.prod(v.shape) for v in autoencoder.trainable_variables]))
