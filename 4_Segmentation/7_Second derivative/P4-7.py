import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def setup_subplot(subplt, name):
    subplt.set_title(name)
    subplt.get_xaxis().set_visible(False)
    subplt.get_yaxis().set_visible(False)


def generate_LoG_kernel(ksize, sigma, laplacian_kernel):
    gaussian_kernel = cv2.getGaussianKernel(ksize, sigma)
    return cv2.filter2D(gaussian_kernel*gaussian_kernel.T, -1, laplacian_kernel)


def generate_DoG_kernel(ksize, sigma1, sigma2):
    gaussian_kernel_1 = cv2.getGaussianKernel(ksize, sigma1)
    gaussian_kernel_2 = cv2.getGaussianKernel(ksize, sigma2)
    return gaussian_kernel_1*gaussian_kernel_1.T - gaussian_kernel_2*gaussian_kernel_2.T


if __name__ == "__main__":
    img = cv2.imread("window.jpg")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    laplacian_kernel_1 = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    laplacian_kernel_2 = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    laplacian_of_gaussian_kernel = generate_LoG_kernel(5, 0.55, laplacian_kernel_1)
    difference_of_gaussian_kernel = generate_DoG_kernel(5, 0.05, 1)

    laplacian_1 = cv2.filter2D(img_gray, cv2.CV_16S, laplacian_kernel_1)
    laplacian_2 = cv2.filter2D(img_gray, cv2.CV_16S, laplacian_kernel_2)
    laplacian_of_gaussian = cv2.filter2D(img_gray, cv2.CV_16S, laplacian_of_gaussian_kernel)
    difference_of_gaussian = cv2.filter2D(img_gray, cv2.CV_16S, difference_of_gaussian_kernel)
    zero_crossings = laplacian_of_gaussian.copy()
    zero_crossings[zero_crossings >= 0] = 255
    zero_crossings[zero_crossings < 0] = -255

    laplacian_1_abs = cv2.convertScaleAbs(laplacian_1)
    laplacian_2_abs = cv2.convertScaleAbs(laplacian_2)
    laplacian_of_gaussian_abs = cv2.convertScaleAbs(laplacian_of_gaussian)
    difference_of_gaussian_abs = cv2.convertScaleAbs(difference_of_gaussian)

    fig, axs = plt.subplots(1, 1)
    axs.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), vmin=0, vmax=255)
    setup_subplot(axs, "Original")
    plt.show(block=False)

    fig, axs = plt.subplots(1, 4)
    setup_subplot(axs[0], "Laplacian 1")
    axs[0].imshow(laplacian_kernel_1,
                  cmap=cm.gray,
                  vmin=np.min(laplacian_kernel_1),
                  vmax=np.max(laplacian_kernel_1))
    setup_subplot(axs[1], "Laplacian 2")
    axs[1].imshow(laplacian_kernel_2,
                  cmap=cm.gray,
                  vmin=np.min(laplacian_kernel_2),
                  vmax=np.max(laplacian_kernel_2))
    setup_subplot(axs[2], "Laplacian of Gaussian (LoG)")
    axs[2].imshow(laplacian_of_gaussian_kernel,
                  cmap=cm.gray,
                  vmin=np.min(laplacian_of_gaussian_kernel),
                  vmax=np.max(laplacian_of_gaussian_kernel))
    setup_subplot(axs[3], "Difference of Gaussian (DoG)")
    axs[3].imshow(difference_of_gaussian_kernel,
                  cmap=cm.gray,
                  vmin=np.min(difference_of_gaussian_kernel),
                  vmax=np.max(difference_of_gaussian_kernel))
    plt.tight_layout()
    plt.show(block=False)

    fig, axs = plt.subplots(1, 4)
    setup_subplot(axs[0], "Laplacian 1")
    axs[0].imshow(laplacian_1, cmap=cm.gray, vmin=-255, vmax=255)
    setup_subplot(axs[1], "Laplacian 2")
    axs[1].imshow(laplacian_2, cmap=cm.gray, vmin=-255, vmax=255)
    setup_subplot(axs[2], "Laplacian of Gaussian (LoG)")
    axs[2].imshow(laplacian_of_gaussian, cmap=cm.gray, vmin=-255, vmax=255)
    setup_subplot(axs[3], "Difference of Gaussian (DoG)")
    axs[3].imshow(difference_of_gaussian, cmap=cm.gray, vmin=-255, vmax=255)
    plt.tight_layout()
    plt.show(block=False)

    fig, axs = plt.subplots(1, 4)
    setup_subplot(axs[0], "Laplacian 1 - Abs")
    axs[0].imshow(laplacian_1_abs, cmap=cm.gray, vmin=0, vmax=255)
    setup_subplot(axs[1], "Laplacian 2 - Abs")
    axs[1].imshow(laplacian_2_abs, cmap=cm.gray, vmin=0, vmax=255)
    setup_subplot(axs[2], "Laplacian of Gaussian (LoG) - Abs")
    axs[2].imshow(laplacian_of_gaussian_abs, cmap=cm.gray, vmin=0, vmax=255)
    setup_subplot(axs[3], "Difference of Gaussian (DoG) - Abs")
    axs[3].imshow(difference_of_gaussian_abs, cmap=cm.gray, vmin=0, vmax=255)
    plt.tight_layout()
    plt.show(block=False)

    fig, axs = plt.subplots(1, 2)
    setup_subplot(axs[0], "Laplacian of Gaussian (LoG)")
    axs[0].imshow(laplacian_of_gaussian, cmap=cm.gray, vmin=-255, vmax=255)
    setup_subplot(axs[1], "Zero-crossing")
    axs[1].imshow(zero_crossings, cmap=cm.gray, vmin=-255, vmax=255)
    plt.tight_layout()
    plt.show()