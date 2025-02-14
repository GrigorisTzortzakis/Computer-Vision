import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

!pip
install - q
scikit - learn
from sklearn.decomposition import PCA


# Convert tensor
def flatten_mnist_image(tensor_img):
    return tensor_img.view(-1).numpy()


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

# Create list
data_list = []
for img_tensor, _ in mnist_train:
    flattened = flatten_mnist_image(img_tensor)
    data_list.append(flattened)

X = np.stack(data_list, axis=0)
print(f"Μέγεθος συνόλου εκπαίδευσης (X): {X.shape}")

# Center
mean_vec = X.mean(axis=0, keepdims=True)  # [1, 784]
X_centered = X - mean_vec

# Pca
pca = PCA(n_components=128)
pca.fit(X_centered)

V_L = pca.components_.T
print(f"Σχήμα του πίνακα V_L: {V_L.shape}")

# Reconstruct
X_transformed = pca.transform(X_centered)  # [N, 128]

# Inverse transform
X_reconstructed_centered = pca.inverse_transform(X_transformed)
X_reconstructed = X_reconstructed_centered + mean_vec

print("Σχήμα των ανακατασκευασμένων δεδομένων:", X_reconstructed.shape)

mse = np.mean((X - X_reconstructed) ** 2)
print(f"Μέσο Σφάλμα Ανακατασκευής (MSE) για L=128: {mse:.4f}")
