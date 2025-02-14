
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist

# Load MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Choose digits
digit0 = 3
digit1 = 7

# Filter the training set for each digit
X0 = X_train[y_train == digit0]
X1 = X_train[y_train == digit1]


# Compute the mean digit in the original space

def compute_mean_digit(images):
    """
    images is an array of shape (num_samples, 28, 28).
    Returns a single 28x28 array representing the average.
    """
    images = images.astype(np.float32)
    mean_img = np.mean(images, axis=0)
    return mean_img

mean0 = compute_mean_digit(X0)
mean1 = compute_mean_digit(X1)

# Display the mean digit in original space

def show_mean_digit(mean_img, digit_label):
    plt.figure()
    plt.imshow(mean_img, cmap='gray')
    plt.title(f"Mean Digit for label = {digit_label}")
    plt.colorbar()
    plt.show()

show_mean_digit(mean0, digit0)
show_mean_digit(mean1, digit1)


# Apply the non-linear mapping k(x)

def k_map(x):
    """
    Given a single image x (2D array),
    flatten to 1D and apply:
        k(x) = exp( - (||x||^2 / 0.1) )
    """
    x_flat = x.flatten().astype(np.float32)
    norm_sq = np.sum(x_flat**2)
    return np.exp(-norm_sq / 0.1)

# Compute k(x) for all images
mapped_vals_0 = [k_map(img) for img in X0]
mapped_vals_1 = [k_map(img) for img in X1]

# Compute mean of these mapped scalars
mean_k_0 = np.mean(mapped_vals_0)
mean_k_1 = np.mean(mapped_vals_1)

print(f"Average k(x) for digit={digit0} is {mean_k_0:.6f}")
print(f"Average k(x) for digit={digit1} is {mean_k_1:.6f}")
