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
n_components = 8

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
        flattened = img_tensor.view(-1).numpy()
        data_list.append(flattened)

# Convert matrix
X = np.stack(data_list, axis=0)  # N samples, 784 features

print(f"Σχήμα dataset (μετά το φιλτράρισμα): {X.shape}")

# Calculate average
mean_vec = X.mean(axis=0, keepdims=True)
X_centered = X - mean_vec  # κεντραρισμένο dataset

# Pca
pca = PCA(n_components=n_components)
pca.fit(X_centered)

# 8 first principles
principal_components = pca.components_

print(f"Σχήμα προκύπτοντων κύριων συνιστωσών: {principal_components.shape}")

fig, axes = plt.subplots(1, n_components, figsize=(2 * n_components, 2))
for i in range(n_components):
    component_2D = principal_components[i].reshape(28, 28)
    ax = axes[i]
    ax.imshow(component_2D, cmap='gray')
    ax.set_title(f"PC #{i + 1}")
    ax.axis('off')

plt.tight_layout()
plt.show()

explained_variance_ratio = pca.explained_variance_ratio_
print("Ποσοστό εξηγούμενης διασποράς ανά συνιστώσα:")
for i, ratio in enumerate(explained_variance_ratio):
    print(f"PC {i + 1}: {ratio * 100:.2f}%")