
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Χρησιμοποιούμε συσκευή: {device}")

# Load mnist
transform = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = torchvision.datasets.MNIST(
    root='.',
    train=True,
    download=True,
    transform=transform
)

digit_a, digit_b = 3, 7
images_a, images_b = [], []

for img_tensor, label in train_dataset:
    # Transfer tensor
    img_tensor = img_tensor.to(device)
    if label == digit_a:
        images_a.append(img_tensor)
    elif label == digit_b:
        images_b.append(img_tensor)




# Align tensors
imgs_a_tensor = torch.stack(images_a)  # [N_a, 1, 28, 28]
imgs_b_tensor = torch.stack(images_b)  # [N_b, 1, 28, 28]

mean_a = imgs_a_tensor.mean(dim=0)     # [1, 28, 28]
mean_b = imgs_b_tensor.mean(dim=0)     # [1, 28, 28]


mean_a_cpu = mean_a.squeeze().detach().cpu().numpy()
mean_b_cpu = mean_b.squeeze().detach().cpu().numpy()


plt.figure(figsize=(6,3))

plt.subplot(1,2,1)
plt.imshow(mean_a_cpu, cmap='gray')
plt.title(f"Μέσο Ψηφίο {digit_a}")
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(mean_b_cpu, cmap='gray')
plt.title(f"Μέσο Ψηφίο {digit_b}")
plt.axis('off')

plt.show()