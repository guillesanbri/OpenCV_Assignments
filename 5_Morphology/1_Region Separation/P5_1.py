import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def setup_subplot(subplt, name):
    subplt.set_title(name)
    subplt.get_xaxis().set_visible(False)
    subplt.get_yaxis().set_visible(False)


def object_count(img, radius, connectivity):
    _, img_binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    img_erode = cv2.erode(img_binary, cv2.getStructuringElement(cv2.MORPH_RECT, (radius, radius)))
    object_number, _ = cv2.connectedComponents(img_erode, connectivity=connectivity)
    print(object_number - 1)  # Minus background

    return img_binary, img_erode


if __name__ == "__main__":
    img = cv2.imread("tirafondos2.png", cv2.IMREAD_GRAYSCALE)

    img_binary, img_erode_10 = object_count(img, 10, 8)
    img_binary, img_erode_30 = object_count(img, 30, 8)

    fig, axs = plt.subplots(2, 2)
    setup_subplot(axs[0, 0], "Original")
    axs[0, 0].imshow(img, cmap=cm.gray, vmin=0, vmax=255)
    setup_subplot(axs[0, 1], "Binary")
    axs[0, 1].imshow(img_binary, cmap=cm.gray, vmin=0, vmax=255)
    setup_subplot(axs[1, 0], "Erode 10")
    axs[1, 0].imshow(img_erode_10, cmap=cm.gray, vmin=0, vmax=255)
    setup_subplot(axs[1, 1], "Erode 25")
    axs[1, 1].imshow(img_erode_30, cmap=cm.gray, vmin=0, vmax=255)
    plt.tight_layout()
    plt.show()