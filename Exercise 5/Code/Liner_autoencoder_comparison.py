import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

# Load mnist

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 784).astype(np.float32) / 255.
x_test = x_test.reshape(-1, 784).astype(np.float32) / 255.

all_data = np.concatenate([x_train, x_test], axis=0)  # shape (70000, 784)

# Center

mean_vec = np.mean(all_data, axis=0, keepdims=True)  # shape (1,784)
centered_data = all_data - mean_vec  # shape (70000,784)

# Create dataset

batch_size = 250
train_dataset = tf.data.Dataset.from_tensor_slices((centered_data, centered_data))
train_dataset = train_dataset.shuffle(70000).batch(batch_size)

# matrix Vl

V_L = np.random.randn(784, 128).astype(np.float32)

# Autoencoder

latent_dim = 128
input_dim = 784

input_layer = layers.Input(shape=(input_dim,), name="ae_input")
encoder_layer = layers.Dense(latent_dim,
                             activation='linear',
                             use_bias=False,
                             name='encoder')
encoded = encoder_layer(input_layer)

decoder_layer = layers.Dense(input_dim,
                             activation='sigmoid',
                             use_bias=False,
                             name='decoder')
decoded = decoder_layer(encoded)

autoencoder = models.Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='binary_crossentropy'
)


# Calculate simularity

def compute_similarity(enc_weights, v_matrix):
    norm_enc = np.sqrt(np.sum(enc_weights ** 2, axis=0, keepdims=True)) + 1e-12
    norm_v = np.sqrt(np.sum(v_matrix ** 2, axis=0, keepdims=True)) + 1e-12

    W_enc_unit = enc_weights / norm_enc
    V_unit = v_matrix / norm_v

    dot_per_col = np.sum(W_enc_unit * V_unit, axis=0)  # (128,)
    return np.mean(dot_per_col)


similarity_log = []


class SimilarityCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Παίρνουμε τα βάρη του κωδικοποιητή
        W_encoder = self.model.get_layer('encoder').get_weights()[0]
        sim_val = compute_similarity(W_encoder, V_L)
        similarity_log.append(sim_val)
        print(f"Epoch {epoch + 1}, Similarity to V_L: {sim_val:.4f}")


sim_callback = SimilarityCallback()

# 40 epochs

epochs = 40
history = autoencoder.fit(
    train_dataset,
    epochs=epochs,
    verbose=1,
    callbacks=[sim_callback]
)

plt.figure(figsize=(7, 4))
plt.plot(range(1, epochs + 1), similarity_log, marker='o')
plt.title("Ομοιότητα κωδικοποιητή με V_L (Cosine Similarity)")
plt.xlabel("Εποχή")
plt.ylabel("Ομοιότητα")
plt.grid(True)
plt.show()
