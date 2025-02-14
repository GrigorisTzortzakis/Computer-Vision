import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Load mnist
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 784).astype(np.float32) / 255.0
x_test = x_test.reshape(-1, 784).astype(np.float32) / 255.0

# Συνένωση train+test
all_data = np.concatenate([x_train, x_test], axis=0)

# Center
mean_vec = np.mean(all_data, axis=0, keepdims=True)
x_train_centered = x_train - mean_vec
x_test_centered = x_test - mean_vec

# Create dataset
batch_size = 250
train_dataset = tf.data.Dataset.from_tensor_slices((x_train_centered, x_train_centered))
train_dataset = train_dataset.shuffle(buffer_size=len(x_train_centered)).batch(batch_size)

epochs = 40
latent_dim = 128


# Leaky relu
class DenseNoBiasLeaky(tf.keras.layers.Layer):
    def __init__(self, in_dim, out_dim, alpha=0.2):
        super().__init__()
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


# Encoder and decoder
class NonLinear3LevelAE(tf.keras.Model):
    def __init__(self, alpha=0.2):
        super().__init__()
        # Encoder: 784->512->256->128
        self.enc1 = DenseNoBiasLeaky(in_dim=784, out_dim=512, alpha=alpha)
        self.enc2 = DenseNoBiasLeaky(in_dim=512, out_dim=256, alpha=alpha)
        self.enc3 = DenseNoBiasLeaky(in_dim=256, out_dim=latent_dim, alpha=alpha)

        # Decoder
        self.dec1 = DenseNoBiasLeaky(in_dim=latent_dim, out_dim=256, alpha=alpha)
        self.dec2 = DenseNoBiasLeaky(in_dim=256, out_dim=512, alpha=alpha)

        self.W_dec3 = self.add_weight(
            shape=(512, 784),
            initializer='glorot_uniform',
            trainable=True
        )

    def call(self, x):
        # Forward encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        # Forward decoder
        d1 = self.dec1(e3)
        d2 = self.dec2(d1)
        # Τελικό: linear => (None, 512) x (512, 784) => (None, 784)
        return tf.matmul(d2, self.W_dec3)  # no activation => linear


# Train
model = NonLinear3LevelAE(alpha=0.2)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='mean_squared_error'
)

history = model.fit(train_dataset, epochs=epochs, verbose=1)

nparams = np.sum([np.prod(v.shape) for v in model.trainable_variables])
print(f"\nΣυνολικές trainable παράμετροι: {nparams}")


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
        orig_img = data[i].reshape(28, 28)
        rec = model.predict(data[i:i + 1], verbose=0).reshape(28, 28)
        axes[row][0].imshow(orig_img, cmap='gray')
        axes[row][0].axis('off')
        axes[row][0].set_title("Original")
        axes[row][1].imshow(rec, cmap='gray')
        axes[row][1].axis('off')
        axes[row][1].set_title("Reconstructed")
    plt.tight_layout()
    plt.show()


plot_reconstructions(model, x_test_centered, nsamples=5)
