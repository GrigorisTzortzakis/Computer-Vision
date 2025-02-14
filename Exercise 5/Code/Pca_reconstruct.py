import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

!pip
install - q
scikit - learn
from sklearn.decomposition import PCA

# Digit 3 and 7
digits_to_select = [3, 7]
components_list = [1, 8, 16, 64, 256]

# Load mnist
transform = transforms.Compose([
    transforms.ToTensor()
])
mnist_train = torchvision.datasets.MNIST(
    root='.',
    train=True,
    download=True,
    transform=transform
)

data_list = []
for img_tensor, label in mnist_train:
    if label in digits_to_select:
        # Μετατροπή από [1, 28, 28] -> [784]
        flattened = img_tensor.view(-1).numpy()
        data_list.append(flattened)

# List
X = np.stack(data_list, axis=0)
print(f"Μέγεθος συνόλου μετά το φιλτράρισμα: {X.shape}")

# Center
mean_vec = X.mean(axis=0, keepdims=True)  # [1, 784]
X_centered = X - mean_vec

# Choose first pic
sample_idx = 0
original_sample = X_centered[sample_idx]  # [784]


# Make pic 28x28
def plot_image(vec_784, title=""):
    """Βοηθητική συνάρτηση για προβολή εικόνας 28x28."""
    plt.imshow(vec_784.reshape(28, 28), cmap='gray')
    plt.title(title)
    plt.axis('off')


# Add average
plt.figure(figsize=(3, 3))
plot_image(original_sample + mean_vec, title="Αρχική Εικόνα (Ένα Δείγμα)")
plt.show()

plt.figure(figsize=(15, 3))
for i, L in enumerate(components_list, start=1):
    # Pca
    pca = PCA(n_components=L)
    pca.fit(X_centered)

    # Reconstruct
    sample_transformed = pca.transform(original_sample.reshape(1, -1))  # [1, L]
    sample_reconstructed = pca.inverse_transform(sample_transformed)  # [1, 784]

    # Add average to recontructed
    sample_reconstructed_full = sample_reconstructed + mean_vec

    plt.subplot(1, len(components_list), i)
    plot_image(sample_reconstructed_full[0], title=f"L = {L}")
plt.tight_layout()
plt.show()