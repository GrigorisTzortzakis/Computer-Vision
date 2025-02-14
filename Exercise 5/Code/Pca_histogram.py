import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

!pip
install - q
scikit - learn
from sklearn.decomposition import PCA

# Load mnist
digits_to_select = [3, 7]
components_list = [1, 8, 16, 64, 256]

transform = transforms.Compose([
    transforms.ToTensor()
])

mnist_train = torchvision.datasets.MNIST(
    root='.',
    train=True,
    download=True,
    transform=transform
)

X_list = []
labels_list = []

for img_tensor, label in mnist_train:
    if label in digits_to_select:
        # Μετατροπή σε μονοδιάστατο διάνυσμα 784 στοιχείων
        flattened = img_tensor.view(-1).numpy()
        X_list.append(flattened)
        labels_list.append(label)

X = np.stack(X_list, axis=0)  # [N, 784]
labels_arr = np.array(labels_list)  # [N]

print("Σχήμα X:", X.shape)
print("Μοναδικές ετικέτες:", np.unique(labels_arr))

# Center
mean_vec = X.mean(axis=0, keepdims=True)  # [1, 784]
X_centered = X - mean_vec


# Find reconstruction
def reconstruction_error(original, reconstructed):
    """
    Υπολογίζει το MSE (mean squared error) για κάθε δείγμα.
    original & reconstructed: [N, 784]
    Επιστρέφει έναν πίνακα [N] με τα σφάλματα ανά δείγμα.
    """
    mse = np.mean((original - reconstructed) ** 2, axis=1)
    return mse


def plot_hist_of_errors(error_arr, digit_label, L_value):
    """
    Δημιουργεί ένα απλό histogram των σφαλμάτων για ένα συγκεκριμένο digit και τιμή L.
    """
    plt.hist(error_arr, bins=50, alpha=0.7, edgecolor='black')
    plt.title(f"Ιστόγραμμα Σφαλμάτων\nΨηφίο = {digit_label}, L = {L_value}")
    plt.xlabel("Σφάλμα Ανακατασκευής (MSE)")
    plt.ylabel("Συχνότητα")


# Pca error
plt.figure(figsize=(14, 8))

num_rows = len(digits_to_select)
num_cols = len(components_list)

for col_idx, L in enumerate(components_list, start=1):
    # Pca
    pca = PCA(n_components=L)
    pca.fit(X_centered)

    # Transform samples
    X_transformed = pca.transform(X_centered)  # [N, L]
    X_reconstructed_centered = pca.inverse_transform(X_transformed)  # [N, 784]
    X_reconstructed = X_reconstructed_centered + mean_vec

    # Find error of samples
    errs = reconstruction_error(X, X_reconstructed)  # [N]

    # Calculate histogram
    for row_idx, digit in enumerate(digits_to_select, start=1):
        plt.subplot(num_rows, num_cols, (row_idx - 1) * num_cols + col_idx)

        digit_mask = (labels_arr == digit)
        digit_errs = errs[digit_mask]

        plot_hist_of_errors(digit_errs, digit_label=digit, L_value=L)

        mean_err = np.mean(digit_errs)
        plt.title(f"Ψηφίο={digit}, L={L}\nMean Error={mean_err:.4f}")

plt.tight_layout()
plt.show()
