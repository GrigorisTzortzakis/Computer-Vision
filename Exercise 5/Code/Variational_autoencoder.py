import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Load mnist

(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 784).astype(np.float32) / 255.
x_test = x_test.reshape(-1, 784).astype(np.float32) / 255.

all_data = np.concatenate([x_train, x_test], axis=0)
mean_vec = np.mean(all_data, axis=0, keepdims=True)
x_train_centered = x_train - mean_vec
x_test_centered = x_test - mean_vec

# Create dataset
batch_size = 250
train_dataset = tf.data.Dataset.from_tensor_slices(x_train_centered)
train_dataset = train_dataset.shuffle(buffer_size=len(x_train_centered)).batch(batch_size)

epochs = 100
latent_dim = 2


# 3 level encoder and decoder
class Encoder(tf.keras.Model):
    def __init__(self, latent_dim):
        super().__init__()
        self.fc1 = tf.keras.layers.Dense(512, activation='relu')
        self.fc2 = tf.keras.layers.Dense(256, activation='relu')
        self.mu_layer = tf.keras.layers.Dense(latent_dim)
        self.logvar_layer = tf.keras.layers.Dense(latent_dim)

    def call(self, x):
        h1 = self.fc1(x)
        h2 = self.fc2(h1)
        mu = self.mu_layer(h2)
        logvar = self.logvar_layer(h2)
        return mu, logvar


class Decoder(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.fc1 = tf.keras.layers.Dense(256, activation='relu')
        self.fc2 = tf.keras.layers.Dense(512, activation='relu')
        self.out = tf.keras.layers.Dense(784, activation='sigmoid')

    def call(self, z):
        h1 = self.fc1(z)
        h2 = self.fc2(h1)
        x_recon = self.out(h2)
        return x_recon


def reparameterize(mu, logvar):
    eps = tf.random.normal(shape=tf.shape(mu))
    return mu + tf.exp(0.5 * logvar) * eps


# 4. VAE (Encoder+Decoder)

class VAE(tf.keras.Model):
    def __init__(self, latent_dim=2):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder()

    def call(self, x):
        mu, logvar = self.encoder(x)
        z = reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar


def vae_loss(x_true, x_recon, mu, logvar):
    eps = 1e-7

    x_recon_clipped = tf.clip_by_value(x_recon, eps, 1.0 - eps)
    bce_matrix = - (x_true * tf.math.log(x_recon_clipped)
                    + (1.0 - x_true) * tf.math.log(1.0 - x_recon_clipped))
    bce_per_sample = tf.reduce_sum(bce_matrix, axis=1)
    recon_loss = tf.reduce_mean(bce_per_sample)

    kl_matrix = 0.5 * (tf.exp(logvar) + tf.square(mu) - 1.0 - logvar)
    kl_per_sample = tf.reduce_sum(kl_matrix, axis=1)
    kl_loss = tf.reduce_mean(kl_per_sample)

    return recon_loss + kl_loss


# train
vae = VAE(latent_dim=latent_dim)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)


@tf.function
def train_step(x):
    with tf.GradientTape() as tape:
        x_recon, mu, logvar = vae(x)
        loss_val = vae_loss(x, x_recon, mu, logvar)
    grads = tape.gradient(loss_val, vae.trainable_variables)
    optimizer.apply_gradients(zip(grads, vae.trainable_variables))
    return loss_val


# noise
fixed_noise = np.random.randn(16, latent_dim).astype(np.float32)


def generate_and_plot(epoch):
    x_gen = vae.decoder(fixed_noise).numpy()
    fig, axes = plt.subplots(4, 4, figsize=(6, 6))
    idx = 0
    for i in range(4):
        for j in range(4):
            axes[i, j].imshow(x_gen[idx].reshape(28, 28), cmap='gray')
            axes[i, j].axis('off')
            idx += 1
    fig.suptitle(f"Epoch {epoch} - Generated from N(0,I)")
    plt.show()


for ep in range(1, epochs + 1):
    epoch_loss, nb = 0., 0
    for batch_x in train_dataset:
        loss_val = train_step(batch_x)
        epoch_loss += loss_val.numpy()
        nb += 1
    epoch_loss /= nb
    print(f"Epoch {ep}/{epochs}, Loss={epoch_loss:.4f}")

    if ep in [1, 50, 100]:
        generate_and_plot(ep)

test_ds = tf.data.Dataset.from_tensor_slices(x_test_centered).batch(batch_size)
test_loss, test_count = 0., 0
for bx in test_ds:
    x_recon_t, mu_t, logvar_t = vae(bx)
    l_val = vae_loss(bx, x_recon_t, mu_t, logvar_t)
    test_loss += l_val.numpy()
    test_count += 1
test_loss_avg = test_loss / test_count
print(f"Test Loss (avg): {test_loss_avg:.4f}")
