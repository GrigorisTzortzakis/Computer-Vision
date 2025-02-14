
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

# Load mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
digitA, digitB = 3, 7

A_data = X_train[y_train == digitA]
B_data = X_train[y_train == digitB]


rng = np.random.default_rng(12345)
idxA = rng.choice(len(A_data), 200, replace=False)
idxB = rng.choice(len(B_data), 200, replace=False)
A_sub = A_data[idxA]
B_sub = B_data[idxB]

# Flatten each image
A_sub = A_sub.reshape(A_sub.shape[0], -1).astype(np.float32)
B_sub = B_sub.reshape(B_sub.shape[0], -1).astype(np.float32)

X = np.vstack([A_sub, B_sub])  # shape = (400, 784)
N = X.shape[0]

# Kernel matrix
K = np.zeros((N ,N), dtype=np.float32)
for i in range(N):
    xi = X[i]
    diff = X - xi  # shape (N,784)
    dist_sq = np.sum(dif f *diff, axis=1)  # shape (N,)
    K[i ,:] = np.exp(-dist_sq / 0.1)

# Center
oneN = np.ones((N ,N), dtype=np.float32 ) /N
Kc = K - one N @K - K@ oneN + oneN @ K @ oneN

# Decomposition
vals, vecs = np.linalg.eigh(Kc)
vals = vals[::-1]
vecs = vecs[:, ::-1]

# Keep positive values
pos_idx = vals > 1e-9
vals = vals[pos_idx]
vecs = vecs[:, pos_idx]


def project_to_kpca(Kc_row, vecs_full, vals_full, L):
    z = []
    for l in range(L):
        proj = np.dot(Kc_row, vecs_full[:, l])
        z.append(proj / np.sqrt(vals_full[l] + 1e-12))
    return np.array(z, dtype=np.float32)


def preimage_rbf(zcoords, Xdata, vecs_full, vals_full, L, max_iter=100, lr=1.0):
    N = Xdata.shape[0]
    alpha = np.zeros(N, dtype=np.float32)
    for j in range(N):
        s = 0.0
        for l in range(L):
            s += zcoords[l] * vecs_full[j, l] / np.sqrt(vals_full[l] + 1e-12)
        alpha[j] = s

    # Initialize x randomly
    x = np.mean(Xdata, axis=0).copy()

    # Fixed-point iteration
    for _ in range(max_iter):
        # Compute K
        diff = Xdata - x
        dist_sq = np.sum(diff * diff, axis=1)
        k_x = np.exp(-dist_sq / 0.1)

        num = np.zeros_like(x)
        denom = 0.0
        for j in range(N):
            w = alpha[j] * k_x[j]
            num += w * Xdata[j]
            denom += w
        if denom < 1e-12:
            break
        x_new = num / denom

        x = x_new
    return x


# Reconstruct
Ls = [1, 8, 16, 64, 256]
sample_indices = [0, 200]  # 0 -> first digit (3), 200 -> second digit (7)

fig, axes = plt.subplots(len(sample_indices), len(Ls) + 1, figsize=(12, 5))
fig.suptitle("Kernel PCA Pre-image Reconstruction")

for row_idx, sidx in enumerate(sample_indices):
    original_img = X[sidx].reshape(28, 28)
    axes[row_idx, 0].imshow(original_img, cmap='gray')
    axes[row_idx, 0].set_title("Original")
    axes[row_idx, 0].axis('off')

    # The row of Kc needed for this sample
    Kc_row = Kc[sidx, :]

    col_idx = 1
    for L_ in Ls:
        z_sample = project_to_kpca(Kc_row, vecs, vals, L_)
        x_rec = preimage_rbf(z_sample, X, vecs, vals, L_)
        rec_img = x_rec.reshape(28, 28)
        axes[row_idx, col_idx].imshow(rec_img, cmap='gray')
        axes[row_idx, col_idx].set_title(f"L={L_}")
        axes[row_idx, col_idx].axis('off')
        col_idx += 1

plt.tight_layout()
plt.show()
