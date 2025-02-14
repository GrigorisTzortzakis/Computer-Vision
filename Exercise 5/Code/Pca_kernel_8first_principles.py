import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

!pip install -q scikit-learn
from sklearn.decomposition import KernelPCA

# Load mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Choose digits
digit_a, digit_b = 3, 7

# Filter
inds_a = np.where(y_train == digit_a)[0]
inds_b = np.where(y_train == digit_b)[0]

X_a = X_train[inds_a]   # σχήμα [Na, 28,28]
X_b = X_train[inds_b]   # σχήμα [Nb, 28,28]
X_two = np.concatenate([X_a, X_b], axis=0)  # [N, 28,28]
N = X_two.shape[0]

print(f"Digit {digit_a} παραδείγματα:", len(X_a))
print(f"Digit {digit_b} παραδείγματα:", len(X_b))
print("Συνολικά δείγματα =", N)

# Transform
X_two_flat = X_two.reshape(N, 28*28).astype(np.float32) / 255.0


# Kernel PCA
kpca = KernelPCA(
    n_components=8,
    kernel='rbf',
    gamma=10.0,
    fit_inverse_transform=False
)
X_kpca = kpca.fit_transform(X_two_flat)   # [N, 8]

print("Σχήμα του X_kpca:", X_kpca.shape)
print("Παραδείγματα:", X_kpca[:5, :], "...")



plt.figure(figsize=(6,6))
plt.scatter(X_kpca[:,0], X_kpca[:,1], c=np.concatenate([np.zeros(len(X_a)), np.ones(len(X_b))]))
plt.colorbar()
plt.title("Scatter των 2 πρώτων μη γραμμικών συνιστωσών (Kernel PCA)")
plt.show()
