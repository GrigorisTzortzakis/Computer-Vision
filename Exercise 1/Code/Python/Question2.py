import cv2
import numpy as np
import matplotlib.pyplot as plt


def gaussian_pyramid(image, levels):
    gp = [image.astype(np.float32)]
    for i in range(levels):
        image = cv2.pyrDown(gp[-1])
        gp.append(image)
    return gp


def laplacian_pyramid(gp):
    lp = [gp[-1]]
    for i in range(len(gp) - 1, 0, -1):
        size = (gp[i - 1].shape[1], gp[i - 1].shape[0])
        expanded = cv2.pyrUp(gp[i], dstsize=size)
        laplacian = cv2.subtract(gp[i - 1], expanded)
        lp.append(laplacian)
    return lp[::-1]


def blend_pyramids(lp1, lp2, gp_mask):
    blended = []
    for l1, l2, g_mask in zip(lp1, lp2, gp_mask):
        blended_layer = l1 * g_mask + l2 * (1.0 - g_mask)
        blended.append(blended_layer)
    return blended


def reconstruct_image(pyramid):
    image = pyramid[-1]
    for layer in pyramid[-2::-1]:
        size = (layer.shape[1], layer.shape[0])
        image = cv2.pyrUp(image, dstsize=size)
        image = cv2.add(image, layer)
    return np.clip(image, 0, 255).astype(np.uint8)


def create_circular_mask(img_shape):
    height, width = img_shape
    center_x = width // 2
    center_y = height // 2
    Y, X = np.ogrid[:height, :width]
    dist_from_center = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)

    radius = min(center_x, center_y)
    mask = 1 - (dist_from_center / radius)
    mask = np.clip(mask, 0, 1)
    return mask


def process_images():
    # Load images
    woman = cv2.imread(r"D:\ceid\computer vision\askhsh 1\photos\woman.png",
                       cv2.IMREAD_GRAYSCALE)
    hand = cv2.imread(r"D:\ceid\computer vision\askhsh 1\photos\hand.png",
                      cv2.IMREAD_GRAYSCALE)

    # Ensure same size
    hand = cv2.resize(hand, (woman.shape[1], woman.shape[0]))

    # Create mask
    mask = create_circular_mask(woman.shape)

    # Number of levels
    levels = int(np.log2(min(woman.shape[0], woman.shape[1]))) - 3

    # Generate pyramids
    gp_woman = gaussian_pyramid(woman, levels)
    gp_hand = gaussian_pyramid(hand, levels)
    gp_mask = gaussian_pyramid(mask, levels)

    # Generate Laplacian pyramids
    lp_woman = laplacian_pyramid(gp_woman)
    lp_hand = laplacian_pyramid(gp_hand)

    # Blend pyramids
    blended_pyramid = blend_pyramids(lp_woman, lp_hand, gp_mask)

    # Reconstruct final image
    blended_image = reconstruct_image(blended_pyramid)

    # Display results
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.imshow(woman, cmap='gray')
    plt.title('Woman')
    plt.axis('off')

    plt.subplot(132)
    plt.imshow(hand, cmap='gray')
    plt.title('Hand')
    plt.axis('off')

    plt.subplot(133)
    plt.imshow(blended_image, cmap='gray')
    plt.title('Blended Result')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    return blended_image


# Execute the solution
blended_result = process_images()