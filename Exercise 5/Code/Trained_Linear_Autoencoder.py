import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

# Load mnist

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 784).astype(np.float32) / 255.
x_test  = x_test.reshape(-1, 784).astype(np.float32) / 255.


all_data = np.concatenate([x_train, x_test], axis=0)


# Center
mean_vec = np.mean(all_data, axis=0, keepdims=True)
centered_data = all_data - mean_vec  # shape=(70000,784)

# Dataset
batch_size = 250
train_dataset = tf.data.Dataset.from_tensor_slices((centered_data, centered_data))
train_dataset = train_dataset.shuffle(buffer_size=centered_data.shape[0]).batch(batch_size)


# Vl matrix
V_L = np.random.randn(784, 128).astype(np.float32)


# no bias
 latent_dim = 128
input_dim  = 784

input_layer = layers.Input(shape=(input_dim,), name="ae_input")

# Coder
encoder_layer = layers.Dense(latent_dim,
                             activation='linear',
                             use_bias=False,
                             name='encoder')
encoded = encoder_layer(input_layer)

# Decoder
decoder_layer = layers.Dense(input_dim,
                             activation='sigmoid',
                             use_bias=False,
                             name='decoder')
decoded = decoder_layer(encoded)

autoencoder = models.Model(inputs=input_layer, outputs=decoded, name="UndercompleteAE")
autoencoder.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='binary_crossentropy'
)


# Calculate similarity
def compute_similarity(enc_weights, v_matrix):
    """
    enc_weights: shape (784,128)
    v_matrix   : shape (784,128)
    Επιστρέφει τη μέση συνημίτονο ομοιότητα στήλης προς στήλη.
    """
    norm_enc = np.sqrt(np.sum(enc_weights**2, axis=0, keepdims=True)) + 1e-12
    norm_v   = np.sqrt(np.sum(v_matrix**2,   axis=0, keepdims=True)) + 1e-12

    W_enc_unit = enc_weights / norm_enc
    V_unit     = v_matrix   / norm_v

    dot_per_col = np.sum(W_enc_unit * V_unit, axis=0)  # shape(128,)
    return np.mean(dot_per_col)


similarity_log = []

class SimilarityCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        W_encoder = self.model.get_layer('encoder').get_weights()[0]  # (784,128)
        sim_val = compute_similarity(W_encoder, V_L)
        similarity_log.append(sim_val)
        print(f"Epoch {epoch+1}, Similarity to V_L: {sim_val:.4f}")

sim_callback = SimilarityCallback()


# train
epochs = 40
history = autoencoder.fit(
    train_dataset,
    epochs=epochs,
    verbose=1,
    callbacks=[sim_callback]
)

plt.figure(figsize=(7,4))
plt.plot(range(1, epochs+1), similarity_log, marker='o')
plt.title("Ομοιότητα κωδικοποιητή με V_L (Cosine Similarity)")
plt.xlabel("Εποχή")
plt.ylabel("Ομοιότητα")
plt.grid(True)
plt.show()




x_test_centered = x_test - mean_vec

def compute_mse(model, data):
    recons = model.predict(data, verbose=0)
    mse_vals = np.mean((data - recons)**2, axis=1)
    return np.mean(mse_vals)

mse_test = compute_mse(autoencoder, x_test_centered)
print(f"\nΜέσο Τετραγωνικό Σφάλμα στο test set: {mse_test:.6f}")


def plot_reconstructions(model, data, nsamples=5):
    idxs = np.random.choice(len(data), nsamples, replace=False)
    fig, axes = plt.subplots(nsamples, 2, figsize=(6,2*nsamples))
    if nsamples==1:
        axes=[axes]

    for row,i in enumerate(idxs):
        original = data[i]
        rec = model.predict(data[i:i+1], verbose=0)
        axes[row][0].imshow(original.reshape(28,28), cmap='gray')
        axes[row][0].set_title("Original")
        axes[row][0].axis('off')
        axes[row][1].imshow(rec.reshape(28,28), cmap='gray')
        axes[row][1].set_title("Reconstructed")
        axes[row][1].axis('off')

    plt.tight_layout()
    plt.show()


plot_reconstructions(autoencoder, x_test_centered, nsamples=5)
