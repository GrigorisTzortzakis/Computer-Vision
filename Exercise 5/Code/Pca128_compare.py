import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


!pip
install - q
scikit - learn
from sklearn.decomposition import PCA


def flatten_mnist_image(tensor_img):
    return tensor_img.view(-1).numpy()


def show_image(vec_784, title=""):
    # Convert to 28x28
    plt.imshow(vec_784.reshape(28, 28), cmap='gray')
    plt.axis('off')
    if title:
        plt.title(title)


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

train_data_list = []
for img_tensor, _ in mnist_train:
    flattened = flatten_mnist_image(img_tensor)
    train_data_list.append(flattened)

X_train = np.stack(train_data_list, axis=0)  # [N_train, 784]
mean_train = X_train.mean(axis=0, keepdims=True)
X_train_centered = X_train - mean_train

print(f"Σχήμα train set: {X_train.shape}")

# PCA
pca = PCA(n_components=128)
pca.fit(X_train_centered)

V_L = pca.components_.T  # [784, 128]
print(f"Σχήμα του V_L: {V_L.shape}")

# Compressed
mnist_test = torchvision.datasets.MNIST(
    root='.',
    train=False,
    download=True,
    transform=transform
)

test_data_list = []
for img_tensor, _ in mnist_test:
    flattened = flatten_mnist_image(img_tensor)
    test_data_list.append(flattened)

X_test = np.stack(test_data_list, axis=0)  # [N_test, 784]

X_test_centered = X_test - mean_train

print(f"Σχήμα test set: {X_test.shape}")

X_test_transformed = pca.transform(X_test_centered)  # [N_test, 128]

# Reconstruct
X_test_reconstructed_centered = pca.inverse_transform(X_test_transformed)  # [N_test, 784]
X_test_reconstructed = X_test_reconstructed_centered + mean_train

# See results
num_samples_to_show = 5
sample_indices = np.random.choice(len(X_test), num_samples_to_show, replace=False)

plt.figure(figsize=(10, 4))
for i, idx in enumerate(sample_indices, start=1):
    # Αρχική εικόνα
    plt.subplot(2, num_samples_to_show, i)
    show_image(X_test[idx], title=f"Original #{idx}")
    # Ανακατασκευασμένη εικόνα
    plt.subplot(2, num_samples_to_show, i + num_samples_to_show)
    show_image(X_test_reconstructed[idx], title=f"Reconstructed #{idx}")

plt.tight_layout()
plt.show()
