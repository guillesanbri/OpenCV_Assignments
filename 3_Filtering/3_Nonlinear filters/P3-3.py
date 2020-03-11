import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def setup_subplot(subplt, name):
    subplt.set_title(name)
    subplt.get_xaxis().set_visible(False)
    subplt.get_yaxis().set_visible(False)


def setup_row(row, imgs):
    setup_subplot(row[0], "Original")
    row[0].imshow(imgs[0], cmap=cm.gray, vmin=0, vmax=255)
    setup_subplot(row[1], "Median 9x9 filter")
    row[1].imshow(imgs[1], cmap=cm.gray, vmin=0, vmax=255)
    setup_subplot(row[2], "Max 9x9 filter")
    row[2].imshow(imgs[2], cmap=cm.gray, vmin=0, vmax=255)
    setup_subplot(row[3], "Min 9x9 filter")
    row[3].imshow(imgs[3], cmap=cm.gray, vmin=0, vmax=255)


def iteration(img, ksize, func):
    row, col = img.shape
    output = img.copy()
    for i in range(row):
        for j in range(col):
            try:
                pixels = img[i - int(ksize / 2):i + int(ksize / 2) + 1, j - int(ksize / 2):j + int(ksize / 2) + 1]
                output[i, j] = func(pixels)
            except:
                pass
    return output


def median_filter(img, ksize):
    return iteration(img, ksize, np.median)


def max_filter(img, ksize):
    return iteration(img, ksize, np.max)


def min_filter(img, ksize):
    return iteration(img, ksize, np.min)


if __name__ == "__main__":
    gaussian_noise = cv2.imread("gaussian.png", cv2.IMREAD_GRAYSCALE)
    sp_noise = cv2.imread("saltpepper.png", cv2.IMREAD_GRAYSCALE)
    speckle_noise = cv2.imread("speckle.png", cv2.IMREAD_GRAYSCALE)

    gaussian_median_5x5 = median_filter(gaussian_noise, 9)
    sp_median_5x5 = median_filter(sp_noise, 9)
    speckle_median_5x5 = median_filter(speckle_noise, 9)

    gaussian_max_filtered = max_filter(gaussian_noise, 9)
    sp_max_filtered = max_filter(sp_noise, 9)
    speckle_max_filtered = max_filter(speckle_noise, 9)

    gaussian_min_filtered = min_filter(gaussian_noise, 9)
    sp_min_filtered = min_filter(sp_noise, 9)
    speckle_min_filtered = min_filter(speckle_noise, 9)

    gauss = [gaussian_noise, gaussian_median_5x5, gaussian_max_filtered, gaussian_min_filtered]
    sp = [sp_noise, sp_median_5x5, sp_max_filtered, sp_min_filtered]
    speckle = [speckle_noise, sp_median_5x5, speckle_max_filtered, speckle_min_filtered]

    fig, axs = plt.subplots(3, 4)
    setup_row(axs[0], gauss)
    setup_row(axs[1], sp)
    setup_row(axs[2], speckle)
    fig.tight_layout()
    plt.show()