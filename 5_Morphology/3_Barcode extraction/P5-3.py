import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def setup_subplot(subplt, name):
    subplt.set_title(name)
    subplt.get_xaxis().set_visible(False)
    subplt.get_yaxis().set_visible(False)


def opening(img, vertical):
    _, img_binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    k = np.ones((1, 70))
    img_open = cv2.morphologyEx(img_binary, cv2.MORPH_OPEN, kernel=k if vertical else k.T)
    return img_open


if __name__ == "__main__":
    barcode_horizontal = cv2.imread("barcode_horizontal.png", cv2.IMREAD_GRAYSCALE)
    barcode_vertical = cv2.imread("barcode_vertical.png", cv2.IMREAD_GRAYSCALE)

    barcode_horizontal_open = opening(barcode_horizontal, False)
    barcode_vertical_open = opening(barcode_vertical, True)

    fig, axs = plt.subplots(2, 2)
    setup_subplot(axs[0, 0], "Original")
    axs[0, 0].imshow(barcode_horizontal, cmap=cm.gray, vmin=0, vmax=255)
    setup_subplot(axs[0, 1], "Original")
    axs[0, 1].imshow(barcode_vertical, cmap=cm.gray, vmin=0, vmax=255)
    setup_subplot(axs[1, 0], "Vertical kernel")
    axs[1, 0].imshow(barcode_horizontal_open, cmap=cm.gray, vmin=0, vmax=255)
    setup_subplot(axs[1, 1], "Horizontal kernel")
    axs[1, 1].imshow(barcode_vertical_open, cmap=cm.gray, vmin=0, vmax=255)
    plt.tight_layout()
    plt.show()
