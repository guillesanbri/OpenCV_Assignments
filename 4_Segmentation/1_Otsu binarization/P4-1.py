import cv2
import matplotlib.cm as cm
import matplotlib.pyplot as plt


def setup_subplot(subplt, name):
    subplt.set_title(name)
    subplt.get_xaxis().set_visible(False)
    subplt.get_yaxis().set_visible(False)


def get_histogram(img):
    return cv2.calcHist([img], channels=[0], mask=None, histSize=[256], ranges=[0, 256])


if __name__ == "__main__":
    img = cv2.imread("road.jpg", cv2.IMREAD_GRAYSCALE)

    fig, axs = plt.subplots(2, 2)
    setup_subplot(axs[0, 0], "Original")
    axs[0, 0].imshow(img, cmap=cm.gray, vmin=0, vmax=255)
    axs[0, 1].set_title("Histogram")
    axs[0, 1].plot(get_histogram(img), color="gray")

    th, otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    correction_factor = 1.3
    _, otsu_correction = cv2.threshold(img, min(correction_factor*th, 255), 255, cv2.THRESH_BINARY)

    setup_subplot(axs[1, 0], "Otsu " + str(th))
    axs[1, 0].imshow(otsu, cmap=cm.gray, vmin=0, vmax=255)
    setup_subplot(axs[1, 1], "Otsu * " + str(correction_factor) + " = " + str(th*correction_factor))
    axs[1, 1].imshow(otsu_correction, cmap=cm.gray, vmin=0, vmax=255)

    fig.tight_layout()
    plt.show()
