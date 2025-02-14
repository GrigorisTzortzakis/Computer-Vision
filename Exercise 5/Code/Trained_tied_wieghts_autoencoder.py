import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

x_train = x_train.reshape(-1, 784).astype(np.float32) / 255.0
x_test = x_test.reshape(-1, 784).astype(np.float32) / 255.0

all_data = np.concatenate([x_train, x_test], axis=0)
mean_vec = np.mean(all_data, axis=0, keepdims=True)
x_train_centered = x_train - mean_vec
x_test_centered = x_test - mean_vec

batch_size = 250
train_dataset = tf.data.Dataset.from_tensor_slices((x_train_centered, x_train_centered))
train_dataset = train_dataset.shuffle(len(x_train_centered)).batch(batch_size)

epochs = 40
latent_dim = 128


# Custom Dense + LeakyReLU

class DenseNoBiasLeaky(tf.keras.layers.Layer):
    def __init__(self, in_dim, out_dim, alpha=0.2):
        super().__init__()
        # Weight matrix
        self.W = self.add_weight(
            shape=(in_dim, out_dim),
            initializer='glorot_uniform',
            trainable=True
        )
        self.alpha = alpha

    def call(self, x):
        z = tf.matmul(x, self.W)
        # Leaky ReLU
        return tf.nn.leaky_relu(z, alpha=self.alpha)


# Custom TransposeDenseNoBiasLeaky (tied weights) + LeakyReLU

class TransposeDenseNoBiasLeaky(tf.keras.layers.Layer):
    def __init__(self, tied_layer, alpha=0.2):
        super().__init__()

        self.tied_layer = tied_layer
        self.alpha = alpha

    def call(self, x):
        W_enc = self.tied_layer.W
        W_t = tf.transpose(W_enc)
        z = tf.matmul(x, W_t)
        # Leaky ReLU
        return tf.nn.leaky_relu(z, alpha=self.alpha)


# Final decoder layer

class TransposeDenseLinear(tf.keras.layers.Layer):
    def __init__(self, tied_layer):
        super().__init__()
        self.tied_layer = tied_layer

    def call(self, x):
        W_enc = self.tied_layer.W
        W_t = tf.transpose(W_enc)
        # linear = καμία ενεργοποίηση => απλώς z
        return tf.matmul(x, W_t)


# 3 level encoder
class Tied3LayerAE(tf.keras.Model):
    def __init__(self, alpha=0.2):
        super().__init__()

        self.enc1 = DenseNoBiasLeaky(784, 512, alpha=alpha)
        self.enc2 = DenseNoBiasLeaky(512, 256, alpha=alpha)
        self.enc3 = DenseNoBiasLeaky(256, latent_dim, alpha=alpha)

        # --- Decoder (tied)

        self.dec1 = TransposeDenseNoBiasLeaky(self.enc3, alpha=alpha)
        self.dec2 = TransposeDenseNoBiasLeaky(self.enc2, alpha=alpha)
        self.dec3 = TransposeDenseLinear(self.enc1)

    def call(self, x):
        # Encoder forward
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        # Decoder forward
        d1 = self.dec1(e3)
        d2 = self.dec2(d1)
        d3 = self.dec3(d2)
        return d3


model = Tied3LayerAE(alpha=0.2)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='mean_squared_error'
)

history = model.fit(train_dataset, epochs=epochs, verbose=1)

nparams = np.sum([np.prod(v.shape) for v in model.trainable_variables])
print(f"\nΣυνολικές trainable παράμετροι (Tied 3-layer AE): {nparams}")


def compute_mse(model, data):
    recons = model.predict(data, verbose=0)
    mse_vals = np.mean((data - recons) ** 2, axis=1)
    return np.mean(mse_vals)


mse_test = compute_mse(model, x_test_centered)
print(f"Μέσο Τετραγωνικό Σφάλμα στο test set: {mse_test:.6f}")


def plot_reconstructions(model, data, nsamples=5):
    idxs = np.random.choice(len(data), nsamples, replace=False)
    fig, axes = plt.subplots(nsamples, 2, figsize=(6, 2 * nsamples))
    if nsamples == 1:
        axes = [axes]

    for row, i in enumerate(idxs):
        orig = data[i].reshape(28, 28)
        rec = model.predict(data[i:i + 1], verbose=0).reshape(28, 28)

        axes[row][0].imshow(orig, cmap='gray')
        axes[row][0].set_title("Original")
        axes[row][0].axis('off')

        axes[row][1].imshow(rec, cmap='gray')
        axes[row][1].set_title("Reconstructed")
        axes[row][1].axis('off')

    plt.tight_layout()
    plt.show()


plot_reconstructions(model, x_test_centered, nsamples=5)
