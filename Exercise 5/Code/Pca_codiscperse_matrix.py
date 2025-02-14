import torch
import torchvision
import torchvision.transforms as transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Χρησιμοποιούμε συσκευή: {device}")

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

# digit 3 and 7
selected_digits = [3, 7]
all_images = []

for img_tensor, label in mnist_train:
    if label in selected_digits:
        img_tensor = img_tensor.to(device)
        # convert dimension
        flattened = img_tensor.view(-1)
        all_images.append(flattened)

X = torch.stack(all_images)  # σχήμα [N, 784]

# Find average
mean_vec = X.mean(dim=0, keepdim=True)  # [1, 784]
X_centered = X - mean_vec  # σχήμα [N, 784]

# Matrix
N = X.shape[0]
cov_matrix = (X_centered.T @ X_centered) / (N - 1)

cov_matrix_cpu = cov_matrix.detach().cpu()

print(f"Σχήμα του μητρώου συνδιασπορών: {cov_matrix_cpu.shape}")
print("Ενδεικτικά, μερικές τιμές από την κύρια διαγώνιο:")
print(cov_matrix_cpu.diagonal()[:10])