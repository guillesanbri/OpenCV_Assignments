import cv2
from matplotlib import pyplot as plt
from matplotlib import cm


def image_histogram(img, title, channels):
    fig, axs = plt.subplots(1, 2)
    fig.suptitle(title)

    axs[0].get_xaxis().set_visible(False)
    axs[0].get_yaxis().set_visible(False)
    axs[0].imshow(img, cmap=cm.gray, vmin=0, vmax=255)

    if channels == "rgb":
        hist_blue = cv2.calcHist([img], channels=[0], mask=None, histSize=[256], ranges=[0, 256])
        hist_green = cv2.calcHist([img], channels=[1], mask=None, histSize=[256], ranges=[0, 256])
        hist_red = cv2.calcHist([img], channels=[2], mask=None, histSize=[256], ranges=[0, 256])
        axs[1].plot(hist_blue, color="blue")
        axs[1].plot(hist_green, color="green")
        axs[1].plot(hist_red, color="red")
    elif channels == "gray":
        hist_gray = cv2.calcHist([img], channels=[0], mask=None, histSize=[256], ranges=[0, 256])
        axs[1].plot(hist_gray, color="gray")

    fig.tight_layout()
    plt.show(block=False)


if __name__ == "__main__":
    img = cv2.imread("washers.jpg", cv2.IMREAD_GRAYSCALE)
    image_histogram(img, "Original", "gray")

    imgInv = 255 - img
    image_histogram(imgInv, "Inv", "gray")

    img50 = img + 50
    image_histogram(img50, "+50", "gray")

    img[img > 160] = 255
    img[img <= 160] = 0
    image_histogram(img, "Threshold", "gray")

    plt.show()
