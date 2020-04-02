import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def setup_subplot(subplt, name):
    subplt.set_title(name)
    subplt.get_xaxis().set_visible(False)
    subplt.get_yaxis().set_visible(False)


def opening(img, size):
    k = np.ones((size, size))
    img_open = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel=k)
    return img_open


def imfill(img):  # https://www.learnopencv.com/filling-holes-in-an-image-using-opencv-python-c/
    _, img_binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask = np.zeros((img.shape[0] + 2, img.shape[1] + 2), np.uint8)
    flood = img_binary.copy()
    cv2.floodFill(flood, mask, (0, 0), 255)
    inv_flood = 255 - flood
    return img_binary | inv_flood


def gen_lut(num):
    lut = np.zeros((num, 3))
    for i in range(1, num):
        lut[i] = random_color()
    return lut


def labels2rgb(labeled_img, lut):
    h, w = labeled_img.shape
    out = cv2.merge((labeled_img.copy(), labeled_img.copy(), labeled_img.copy()))
    for i in range(0, h):
        for j in range(0, w):
            out[i, j] = lut[labeled_img[i, j]]
    return out


def random_color():
    return np.array(np.random.choice(range(256), size=3))


if __name__ == "__main__":
    img_gray = cv2.imread("corona.bmp", cv2.IMREAD_GRAYSCALE)
    binary_fill = imfill(img_gray)

    img_open = opening(binary_fill, 15)

    difference = binary_fill - img_open

    object_number, labels = cv2.connectedComponents(difference, connectivity=4)
    print(object_number - 1)
    lut = gen_lut(object_number)
    labels = labels2rgb(labels, lut)

    fig, axs = plt.subplots(2, 2)
    setup_subplot(axs[0, 0], "Original")
    axs[0, 0].imshow(img_gray, cmap=cm.gray, vmin=0, vmax=255)
    setup_subplot(axs[0, 1], "Opening")
    axs[0, 1].imshow(img_open, cmap=cm.gray, vmin=0, vmax=255)
    setup_subplot(axs[1, 0], "Difference")
    axs[1, 0].imshow(difference, cmap=cm.gray, vmin=0, vmax=255)
    setup_subplot(axs[1, 1], "Labels")
    axs[1, 1].imshow(labels, vmin=0, vmax=255)
    plt.tight_layout()
    plt.show()