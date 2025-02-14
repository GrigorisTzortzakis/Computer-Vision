import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

# Load mnist
(dX_train, dY_train), (dX_test, dY_test) = mnist.load_data()

digitA = 0  # πρώτο ψηφίο
digitB = 1  # δεύτερο ψηφίο

inds_A = np.where(dY_train == digitA)[0]
inds_B = np.where(dY_train == digitB)[0]

X_A = dX_train[inds_A]  # π.χ. όλες οι εικόνες με label = digitA
X_B = dX_train[inds_B]  # όλες οι εικόνες με label = digitB

X_two_digits = np.concatenate([X_A, X_B], axis=0)  # Σχήμα [N, 28,28]
N = X_two_digits.shape[0]

print(f"Digit {digitA} παραδείγματα:", len(X_A))
print(f"Digit {digitB} παραδείγματα:", len(X_B))
print("Συνολικό μέγεθος N =", N)


# Non linear transform
def k_map(x_img):
    x_float = x_img.astype(np.float32) / 255.0

    x_flat = x_float.reshape(-1)

    norm_sq = np.sum(x_flat ** 2)
    return np.exp(-norm_sq / 0.1)


# Calculate kx
k_values = np.array([k_map(img) for img in X_two_digits])  # [N]

print("Σχήμα k_values:", k_values.shape)  # π.χ. (N,)

# Calculate matrix

# Reshape
K_1D = k_values.reshape(-1, 1)  # (N,1)

# Average
mean_col = np.mean(K_1D, axis=0)  # shape (1,)

# Center
K_centered = K_1D - mean_col  # [N,1]

# Matrix

S_k = (K_centered.T @ K_centered) / (N - 1)

print("Σχήμα του μητρώου συνδιασποράς:", S_k.shape)
print("Μητρώο συνδιασποράς =", S_k)

variance_k = S_k[0, 0]
print(f"Διακύμανση των k(x): {variance_k:.6f}")
