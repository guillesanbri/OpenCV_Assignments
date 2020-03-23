import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def setup_subplot(subplt, name):
    subplt.set_title(name)
    subplt.get_xaxis().set_visible(False)
    subplt.get_yaxis().set_visible(False)


if __name__ == "__main__":
    img = cv2.imread("bilbao.jpg", cv2.IMREAD_GRAYSCALE)

    # _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    sobel_custom_x = cv2.sepFilter2D(img,
                                     cv2.CV_16S,
                                     kernelX=np.array([-1, 0, 1]),
                                     kernelY=np.array([1, 2, 1]))  # 16bits
    sobel_custom_y = cv2.sepFilter2D(img,
                                     cv2.CV_16S,
                                     kernelX=np.array([1, 2, 1]),
                                     kernelY=np.array([-1, 0, 1]))  # 16bits

    sobel_custom_x_abs = cv2.convertScaleAbs(sobel_custom_x)  # 8bits
    sobel_custom_y_abs = cv2.convertScaleAbs(sobel_custom_y)  # 8bits

    sobel_x_y = cv2.convertScaleAbs(abs(sobel_custom_x) + abs(sobel_custom_y))

    sobel_x_2 = sobel_custom_x_abs.astype(float)**2
    sobel_y_2 = sobel_custom_y_abs.astype(float)**2
    sobel_x_y_2 = np.sqrt(sobel_x_2 + sobel_y_2).astype(int)
    sobel_x_y_2 = cv2.convertScaleAbs(sobel_x_y_2)

    _, sobel_binary = cv2.threshold(sobel_x_y, 250, 255, cv2.THRESH_BINARY)

    fig, axs = plt.subplots(1, 3)
    setup_subplot(axs[0], "Original")
    axs[0].imshow(img, cmap=cm.gray, vmin=0, vmax=255)
    setup_subplot(axs[1], "Sobel x")
    axs[1].imshow(sobel_custom_x, cmap=cm.gray, vmin=-255, vmax=255)
    setup_subplot(axs[2], "Sobel y")
    axs[2].imshow(sobel_custom_y, cmap=cm.gray, vmin=-255, vmax=255)
    plt.tight_layout()
    plt.show(block=False)

    fig, axs = plt.subplots(1, 3)
    setup_subplot(axs[0], "Original")
    axs[0].imshow(img, cmap=cm.gray, vmin=0, vmax=255)
    setup_subplot(axs[1], "Sobel x abs")
    axs[1].imshow(sobel_custom_x_abs, cmap=cm.gray, vmin=0, vmax=255)
    setup_subplot(axs[2], "Sobel y abs")
    axs[2].imshow(sobel_custom_y_abs, cmap=cm.gray, vmin=0, vmax=255)
    plt.tight_layout()
    plt.show(block=False)

    fig, axs = plt.subplots(1, 2)
    setup_subplot(axs[0], "sqrt(dx^2 + dy^2)")
    axs[0].imshow(sobel_x_y_2, cmap=cm.gray, vmin=0, vmax=255)
    setup_subplot(axs[1], "abs(dx) + abs(dy)")
    axs[1].imshow(sobel_x_y, cmap=cm.gray, vmin=0, vmax=255)
    plt.tight_layout()
    plt.show(block=False)

    fig, axs = plt.subplots(1, 2)
    setup_subplot(axs[0], "abs(dx) + abs(dy)")
    axs[0].imshow(sobel_x_y, cmap=cm.gray, vmin=0, vmax=255)
    setup_subplot(axs[1], "Threshold")
    axs[1].imshow(sobel_binary, cmap=cm.gray, vmin=0, vmax=255)
    plt.tight_layout()
    plt.show()