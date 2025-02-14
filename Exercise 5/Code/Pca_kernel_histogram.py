
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

# Load mnist
(X_train, y_train), _ = mnist.load_data()
digitA, digitB = 3, 7

A_data = X_train[y_train == digitA]
B_data = X_train[y_train == digitB]

rng = np.random.default_rng(42)
idxA = rng.choice(len(A_data), 200, replace=False)
idxB = rng.choice(len(B_data), 200, replace=False)
A_sub = A_data[idxA]
B_sub = B_data[idxB]

# Flatten images
A_sub = A_sub.reshape(A_sub.shape[0], -1).astype(np.float32)
B_sub = B_sub.reshape(B_sub.shape[0], -1).astype(np.float32)

X = np.vstack([A_sub, B_sub])  # shape = (400, 784)
N, D = X.shape

# Build kernel
def rbf_kernel_matrix(Xdata, sigma_sq=0.1):
    N = Xdata.shape[0]
    K = np.zeros((N ,N), dtype=np.float32)
    for i in range(N):
        diff = Xdata - Xdata[i]
        dist_sq = np.sum(dif f *diff, axis=1)
        K[i ,:] = np.exp(-dist_sq / sigma_sq)
    return K

K = rbf_kernel_matrix(X, sigma_sq=0.1)

# Center
oneN = np.ones((N ,N), dtype=np.float32 ) /N
Kc = K - one N @K - K@ oneN + oneN @ K @ oneN

# Eigen-decomposition of the centered kernel
vals, vecs = np.linalg.eigh(Kc)  # ascending order
vals = vals[::-1]
vecs = vecs[:, ::-1]
pos_mask = vals > 1e-12
vals = vals[pos_mask]
vecs = vecs[:, pos_mask]

# Pick L for reconstruction
L = 16
vals_L = vals[:L]
vecs_L = vecs[:, :L]


# Projection
def project_kpca(Kc_row, vecs_eig, vals_eig):
    zcoords = []
    for l in range(len(vals_eig)):
        proj = np.dot(Kc_row, vecs_eig[:, l])
        zcoords.append(proj / np.sqrt(vals_eig[l] + 1e-12))
    return np.array(zcoords, dtype=np.float32)


def preimage_rbf(zcoords, Xdata, vecs_eig, vals_eig, max_iter=50):
    N = Xdata.shape[0]
    alpha = np.zeros(N, dtype=np.float32)
    for j in range(N):
        s = 0.
        for l in range(len(vals_eig)):
            s += zcoords[l] * vecs_eig[j, l] / np.sqrt(vals_eig[l] + 1e-12)
        alpha[j] = s

    # Start from mean of data
    x = np.mean(Xdata, axis=0).copy()

    for _ in range(max_iter):
        diff = Xdata - x
        dist_sq = np.sum(diff * diff, axis=1)
        k_x = np.exp(-dist_sq / 0.1)
        num = np.zeros_like(x)
        denom = 0.
        for j in range(N):
            w = alpha[j] * k_x[j]
            num += w * Xdata[j]
            denom += w
        if denom < 1e-12:
            break
        x = num / denom

    return x


X_rec = np.zeros_like(X)
for i in range(N):
    z_i = project_kpca(Kc[i, :], vecs_L, vals_L)
    X_rec[i] = preimage_rbf(z_i, X, vecs_L, vals_L, max_iter=50)

mse = np.mean((X - X_rec) ** 2, axis=1)

labels = np.array([digitA] * 200 + [digitB] * 200)
mse_digitA = mse[labels == digitA]
mse_digitB = mse[labels == digitB]

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.hist(mse_digitA, bins=30, color='blue', alpha=0.7)
plt.title(f"Digit {digitA} - Reconstruction Errors, L={L}")
plt.xlabel("MSE");
plt.ylabel("Count")

plt.subplot(1, 2, 2)
plt.hist(mse_digitB, bins=30, color='green', alpha=0.7)
plt.title(f"Digit {digitB} - Reconstruction Errors, L={L}")
plt.xlabel("MSE");
plt.ylabel("Count")

plt.tight_layout()
plt.show()

print(f"Avg MSE for digit {digitA}: {np.mean(mse_digitA):.2f}")
print(f"Avg MSE for digit {digitB}: {np.mean(mse_digitB):.2f}")
