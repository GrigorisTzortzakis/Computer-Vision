import cv2
import numpy as np
import matplotlib.pyplot as plt

def gaussian_pyramid(image, levels):
    gp = [image.astype(np.float32)]  # Convert to float32 at the start
    for i in range(levels):
        image = cv2.pyrDown(gp[-1])
        gp.append(image)
    return gp

def laplacian_pyramid(gp):
    lp = [gp[-1]]
    for i in range(len(gp) - 1, 0, -1):
        size = (gp[i-1].shape[1], gp[i-1].shape[0])
        expanded = cv2.pyrUp(gp[i], dstsize=size)
        laplacian = cv2.subtract(gp[i-1], expanded)
        lp.append(laplacian)
    return lp[::-1]

def blend_pyramids(lp1, lp2, gp_mask):
    blended = []
    for l1, l2, g_mask in zip(lp1, lp2, gp_mask):
        # Ensure proper broadcasting for 3-channel images
        if g_mask.ndim == 2:
            g_mask = np.expand_dims(g_mask, axis=-1)
        blended_layer = l1 * g_mask + l2 * (1.0 - g_mask)
        blended.append(blended_layer)
    return blended

def reconstruct_image(pyramid):
    image = pyramid[-1]
    for layer in pyramid[-2::-1]:
        size = (layer.shape[1], layer.shape[0])
        image = cv2.pyrUp(image, dstsize=size)
        image = cv2.add(image, layer)
    return np.clip(image, 0, 255).astype(np.uint8)  # Ensure valid range

# Load and prepare images
apple = cv2.cvtColor(cv2.imread(r"D:\ceid\computer vision\askhsh 1\photos\apple.jpg"), cv2.COLOR_BGR2RGB)
orange = cv2.cvtColor(cv2.imread(r"D:\ceid\computer vision\askhsh 1\photos\orange.jpg"), cv2.COLOR_BGR2RGB)

# Convert to float32 at the start
apple = apple.astype(np.float32)
orange = orange.astype(np.float32)

# Create mask
rows, cols = apple.shape[:2]
mask = np.zeros((rows, cols), dtype=np.float32)
for i in range(cols):
    mask[:, i] = 1 - (i / cols)  # Create a smooth gradient from 1 to 0

# Number of levels
levels = 5

# Generate pyramids
gp_apple = gaussian_pyramid(apple, levels)
gp_orange = gaussian_pyramid(orange, levels)
gp_mask = gaussian_pyramid(mask, levels)

# Generate Laplacian pyramids
lp_apple = laplacian_pyramid(gp_apple)
lp_orange = laplacian_pyramid(gp_orange)

# Blend pyramids
blended_pyramid = blend_pyramids(lp_apple, lp_orange, gp_mask)

# Reconstruct final image
blended_image = reconstruct_image(blended_pyramid)

# Display results
def display_pyramid(pyramid, title):
    plt.figure(figsize=(15, 3))
    for i, layer in enumerate(pyramid):
        plt.subplot(1, len(pyramid), i + 1)
        if layer.ndim == 2:
            plt.imshow(cv2.convertScaleAbs(layer), cmap='gray')
        else:
            plt.imshow(cv2.convertScaleAbs(layer))
        plt.title(f'{title}\nLevel {i}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Display pyramids
display_pyramid(gp_apple, "Gaussian Pyramid - Apple")
display_pyramid(lp_apple, "Laplacian Pyramid - Apple")
display_pyramid(gp_orange, "Gaussian Pyramid - Orange")
display_pyramid(lp_orange, "Laplacian Pyramid - Orange")

# Display both direct blend and pyramid blend
plt.figure(figsize=(15, 5))

# Direct blend (with visible line)
plt.subplot(121)
direct_blend = np.zeros_like(apple)
direct_blend[:, :cols//2] = apple[:, :cols//2]
direct_blend[:, cols//2:] = orange[:, cols//2:]
plt.imshow(direct_blend.astype(np.uint8))
plt.title("Direct Blending (With Line)")
plt.axis("off")

# Pyramid blend
plt.subplot(122)
plt.imshow(blended_image)
plt.title("Pyramid Blending (Smooth Transition)")
plt.axis("off")

plt.tight_layout()
plt.show()