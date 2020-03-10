import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm


def setup_subplot(subplt, name):
    subplt.set_title(name)
    subplt.get_xaxis().set_visible(False)
    subplt.get_yaxis().set_visible(False)


def generate_gaussian_noise(img, mean=0, var=0.01):
    row, col = img.shape
    sigma = var**0.5
    gauss = 255 * np.array(np.random.normal(mean, sigma, (row, col)))
    noisy_image = np.clip((img + gauss), 0, 255)
    noisy_image = noisy_image.astype(np.uint8)
    return noisy_image


def generate_sandp_noise(img, density=0.05):
    row, col = img.shape
    thres = 1 - density
    noisy_image = img.copy()
    for i in range(row):
        for j in range(col):
            prob = np.random.rand()
            if density < prob < thres:
                pass
            elif prob < density:
                noisy_image.itemset(i, j, 0)
            elif prob > thres:
                noisy_image.itemset(i, j, 255)
    return noisy_image


def generate_speckle_noise(img, var=0.01):
    row, col = img.shape
    sigma = var**0.5
    gauss = np.array(np.random.normal(0, sigma, (row, col)))
    noisy_image = np.clip((img + img*gauss), 0, 255)
    noisy_image = noisy_image.astype(np.uint8)
    return noisy_image


if __name__ == "__main__":
    img = cv2.imread("moon.jpg", cv2.IMREAD_GRAYSCALE)

    gaussian = generate_gaussian_noise(img)
    salt_pepper = generate_sandp_noise(img)
    speckle = generate_speckle_noise(img)

    fig, axs = plt.subplots(2, 2)
    setup_subplot(axs[0, 0], "Original")
    axs[0, 0].imshow(img, cmap=cm.gray, vmin=0, vmax=255)
    setup_subplot(axs[0, 1], "Gaussian")
    axs[0, 1].imshow(gaussian, cmap=cm.gray, vmin=0, vmax=255)
    setup_subplot(axs[1, 0], "Salt and pepper")
    axs[1, 0].imshow(salt_pepper, cmap=cm.gray, vmin=0, vmax=255)
    setup_subplot(axs[1, 1], "Speckle")
    axs[1, 1].imshow(speckle, cmap=cm.gray, vmin=0, vmax=255)

    fig.tight_layout()
    plt.show()