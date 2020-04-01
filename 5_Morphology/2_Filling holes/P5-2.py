import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def setup_subplot(subplt, name):
    subplt.set_title(name)
    subplt.get_xaxis().set_visible(False)
    subplt.get_yaxis().set_visible(False)


def closing(img, size):
    _, img_binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    k = np.ones((size, size))
    filled = cv2.morphologyEx(img_binary, cv2.MORPH_CLOSE, kernel=k)
    return img_binary, filled


def imfill(img):  # https://www.learnopencv.com/filling-holes-in-an-image-using-opencv-python-c/
    _, img_binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask = np.zeros((img.shape[0] + 2, img.shape[1] + 2), np.uint8)
    flood = img_binary.copy()
    cv2.floodFill(flood, mask, (0, 0), 255)
    inv_flood = 255 - flood
    return img_binary | inv_flood


if __name__ == "__main__":
    img = cv2.imread("corona.png", cv2.IMREAD_GRAYSCALE)

    binary, close = closing(img, 50)
    fill = imfill(img)

    fig, axs = plt.subplots(1, 3)
    fig.suptitle("Morphology closing")
    setup_subplot(axs[0], "Original")
    axs[0].imshow(img, cmap=cm.gray, vmin=0, vmax=255)
    setup_subplot(axs[1], "Binary")
    axs[1].imshow(binary, cmap=cm.gray, vmin=0, vmax=255)
    setup_subplot(axs[2], "Closing")
    axs[2].imshow(close, cmap=cm.gray, vmin=0, vmax=255)
    plt.tight_layout()
    plt.show(block=False)

    fig, axs = plt.subplots(1, 2)
    fig.suptitle("Imfill closing")
    setup_subplot(axs[0], "Original")
    axs[0].imshow(img, cmap=cm.gray, vmin=0, vmax=255)
    setup_subplot(axs[1], "Closing")
    axs[1].imshow(fill, cmap=cm.gray, vmin=0, vmax=255)
    plt.tight_layout()
    plt.show()