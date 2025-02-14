import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Load mnist
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.reshape(-1, 784).astype(np.float32) / 255.
x_test = x_test.reshape(-1, 784).astype(np.float32) / 255.

all_data = np.concatenate([x_train, x_test], axis=0)
mean_vec = np.mean(all_data, axis=0, keepdims=True)

# Center
x_train_centered = x_train - mean_vec
x_test_centered = x_test - mean_vec

# Dataset
batch_size = 250
train_dataset = tf.data.Dataset.from_tensor_slices(x_train_centered)
train_dataset = train_dataset.shuffle(buffer_size=len(x_train_centered)).batch(batch_size)

epochs = 100
latent_dim = 2


# 3 level encoder and decoder
class Encoder(tf.keras.Model):
    def __init__(self, latent_dim):
        super().__init__()
        # τρία επίπεδα: 784->512->256->(mu,logvar)
        self.fc1 = tf.keras.layers.Dense(512, activation='relu')
        self.fc2 = tf.keras.layers.Dense(256, activation='relu')
        self.mu_layer = tf.keras.layers.Dense(latent_dim)  # linear
        self.logvar_layer = tf.keras.layers.Dense(latent_dim)  # linear

    def call(self, x):
        h1 = self.fc1(x)
        h2 = self.fc2(h1)
        mu = self.mu_layer(h2)
        logvar = self.logvar_layer(h2)
        return mu, logvar


class Decoder(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # 3 levels
        self.fc1 = tf.keras.layers.Dense(256, activation='relu')
        self.fc2 = tf.keras.layers.Dense(512, activation='relu')
        self.out = tf.keras.layers.Dense(784, activation='sigmoid')

    def call(self, z):
        h1 = self.fc1(z)
        h2 = self.fc2(h1)
        x_recon = self.out(h2)  # [None,784], [0,1]
        return x_recon


def reparameterize(mu, logvar):
    """
    z = mu + exp(0.5*logvar)*eps,   eps ~ N(0,I)
    """
    eps = tf.random.normal(shape=tf.shape(mu))
    return mu + tf.exp(0.5 * logvar) * eps


# Cost function
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

    # BCE pixel-wise:
    x_recon_clipped = tf.clip_by_value(x_recon, eps, 1.0 - eps)
    bce_matrix = -(x_true * tf.math.log(x_recon_clipped)
                   + (1.0 - x_true) * tf.math.log(1.0 - x_recon_clipped))
    bce_per_sample = tf.reduce_sum(bce_matrix, axis=1)
    recon_loss = tf.reduce_mean(bce_per_sample)

    # KL Divergence vs N(0,I):

    kl_matrix = 0.5 * (tf.exp(logvar) + tf.square(mu) - 1.0 - logvar)
    kl_per_sample = tf.reduce_sum(kl_matrix, axis=1)
    kl_loss = tf.reduce_mean(kl_per_sample)

    return recon_loss + kl_loss


# Train
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


# Noise
fixed_noise = np.random.randn(16, latent_dim).astype(np.float32)


def generate_and_plot_samples(epoch):
    """
    Παράγει 16 δείγματα από decoder(fixed_noise) και τα σχεδιάζει σε 4x4.
    """
    x_gen = vae.decoder(fixed_noise).numpy()
    fig, axes = plt.subplots(4, 4, figsize=(6, 6))
    idx = 0
    for i in range(4):
        for j in range(4):
            axes[i, j].imshow(x_gen[idx].reshape(28, 28), cmap='gray')
            axes[i, j].axis('off')
            idx += 1
    fig.suptitle(f"Epoch {epoch} - Generated samples from N(0,I)")
    plt.show()


# Training Loop (100 epochs)
for ep in range(1, epochs + 1):
    epoch_loss, n_batches = 0., 0
    for batch_x in train_dataset:
        loss_val = train_step(batch_x)
        epoch_loss += loss_val.numpy()
        n_batches += 1
    epoch_loss /= n_batches
    print(f"Epoch {ep}/{epochs}, Loss={epoch_loss:.4f}")

    # Εμφανίζουμε το batch θορύβου στις εποχές 1,50,100
    if ep in [1, 50, 100]:
        generate_and_plot_samples(ep)

# Scatter plot

test_dataset = tf.data.Dataset.from_tensor_slices((x_test_centered, y_test)).batch(batch_size)

all_z = []
all_labels = []

for bx, by in test_dataset:
    mu_val, logvar_val = vae.encoder(bx)
    mu_np = mu_val.numpy()  # shape (batch, 2)
    all_z.append(mu_np)
    all_labels.append(by.numpy())

all_z = np.concatenate(all_z, axis=0)  # (N_test,2)
all_labels = np.concatenate(all_labels, axis=0)  # (N_test,)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(all_z[:, 0], all_z[:, 1], c=all_labels, cmap='tab10', s=5, alpha=0.7)
plt.colorbar(scatter, ticks=range(10))
plt.title("Latent representation of test set (colored by digit)")
plt.xlabel("z1")
plt.ylabel("z2")
plt.show()
