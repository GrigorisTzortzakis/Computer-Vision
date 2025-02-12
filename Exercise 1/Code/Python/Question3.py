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


def blend_pyramids(lp_list, gp_masks):
    blended = []
    num_levels = len(lp_list[0])

    for level in range(num_levels):
        # Get current level size from first image pyramid
        current_size = lp_list[0][level].shape
        level_result = np.zeros(current_size)

        # Blend all images at current level
        for i, (laplacian, mask) in enumerate(zip(lp_list, gp_masks)):
            # Resize mask to match current level size
            resized_mask = cv2.resize(mask[level], (current_size[1], current_size[0]))
            if len(current_size) == 3:  # For RGB images
                resized_mask = np.expand_dims(resized_mask, axis=2)
            level_result += laplacian[level] * resized_mask

        blended.append(level_result)
    return blended


def reconstruct_image(pyramid):
    image = pyramid[-1]
    for layer in pyramid[-2::-1]:
        size = (layer.shape[1], layer.shape[0])
        image = cv2.pyrUp(image, dstsize=size)
        image = cv2.add(image, layer)
    return np.clip(image, 0, 255).astype(np.uint8)


def create_position_masks(img_shape, positions):
    """Create masks for different image positions"""
    masks = []
    base_mask = np.zeros((img_shape[0], img_shape[1]))

    for pos in positions:
        mask = base_mask.copy()
        x1, y1, x2, y2 = pos
        mask[y1:y2, x1:x2] = 1
        masks.append(mask)

    # Ensure masks sum to 1
    masks_sum = np.sum(masks, axis=0)
    masks = [mask / np.maximum(masks_sum, 1) for mask in masks]

    return masks


def process_multiple_images():
    # Load all images
    image_paths = [
        r"D:\ceid\computer vision\askhsh 1\photos\P200.jpg",
        r"D:\ceid\computer vision\askhsh 1\photos\dog1.jpg",
        r"D:\ceid\computer vision\askhsh 1\photos\dog2.jpg",
        r"D:\ceid\computer vision\askhsh 1\photos\cat.jpg",
        r"D:\ceid\computer vision\askhsh 1\photos\bench.jpg",
        r"D:\ceid\computer vision\askhsh 1\photos\My_image.jpg"
    ]

    # Define target size
    target_size = (800, 800)

    # Load and resize images
    images = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, target_size)
            images.append(img)

    # Define positions for each image (x1, y1, x2, y2)
    positions = [
        (0, 0, 400, 400),  # P200
        (400, 0, 800, 400),  # dog1
        (0, 400, 400, 800),  # dog2
        (400, 400, 800, 800),  # cat
        (200, 200, 600, 600),  # bench
        (300, 300, 500, 500)  # My_image
    ]

    # Create masks
    masks = create_position_masks((target_size[0], target_size[1]), positions)

    # Number of levels
    levels = 4

    # Generate Gaussian pyramids for all images and masks
    gp_images = [gaussian_pyramid(img, levels) for img in images]
    gp_masks = [gaussian_pyramid(mask, levels) for mask in masks]

    # Generate Laplacian pyramids for all images
    lp_images = [laplacian_pyramid(gp) for gp in gp_images]

    # Blend pyramids
    blended_pyramid = blend_pyramids(lp_images, gp_masks)

    # Reconstruct final image
    blended_image = reconstruct_image(blended_pyramid)

    # Display results
    plt.figure(figsize=(20, 10))

    # Display original images
    for i, img in enumerate(images):
        plt.subplot(2, 3, i + 1)
        plt.imshow(img)
        plt.title(f'Image {i + 1}')
        plt.axis('off')

    # Display final blended result
    plt.figure(figsize=(10, 10))
    plt.imshow(blended_image)
    plt.title('Blended Result')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    return blended_image


# Execute the solution
blended_result = process_multiple_images()