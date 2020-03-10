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
    setup_subplot(row[1], "Gaussian 9x9 kernel")
    row[1].imshow(imgs[1], cmap=cm.gray, vmin=0, vmax=255)
    setup_subplot(row[2], "Mean 9x9 kernel")
    row[2].imshow(imgs[2], cmap=cm.gray, vmin=0, vmax=255)


def get_mean_filter(size):
    return np.ones((size, size)) / size**2


if __name__ == "__main__":
    gaussian_noise = cv2.imread("gaussian.png", cv2.IMREAD_GRAYSCALE)
    sp_noise = cv2.imread("saltpepper.png", cv2.IMREAD_GRAYSCALE)
    speckle_noise = cv2.imread("speckle.png", cv2.IMREAD_GRAYSCALE)

    gaussian_9x1 = cv2.getGaussianKernel(9, -1)  # 9x1 kernel
    gaussian_gaussian_9x9 = cv2.sepFilter2D(gaussian_noise, -1, gaussian_9x1, gaussian_9x1)
    sp_gaussian_9x9 = cv2.sepFilter2D(sp_noise, -1, gaussian_9x1, gaussian_9x1)
    speckle_gaussian_9x9 = cv2.sepFilter2D(speckle_noise, -1, gaussian_9x1, gaussian_9x1)

    mean_9x9 = get_mean_filter(9)
    gaussian_mean_9x9 = cv2.filter2D(gaussian_noise, -1, mean_9x9)
    sp_mean_9x9 = cv2.filter2D(sp_noise, -1, mean_9x9)
    speckle_mean_9x9 = cv2.filter2D(speckle_noise, -1, mean_9x9)

    gauss = [gaussian_noise, gaussian_gaussian_9x9, gaussian_mean_9x9]
    sp = [sp_noise, sp_gaussian_9x9, sp_mean_9x9]
    speckle = [speckle_noise, speckle_gaussian_9x9, speckle_mean_9x9]

    fig, axs = plt.subplots(3, 3)
    setup_row(axs[0], gauss)
    setup_row(axs[1], sp)
    setup_row(axs[2], speckle)
    fig.tight_layout()
    plt.show()
