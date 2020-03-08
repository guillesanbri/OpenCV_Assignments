import cv2
from matplotlib import pyplot as plt


def show_histogram(img, channels):
    if channels == "rgb":
        hist_blue = cv2.calcHist([img], channels=[0], mask=None, histSize=[256], ranges=[0, 256])
        hist_green = cv2.calcHist([img], channels=[1], mask=None, histSize=[256], ranges=[0, 256])
        hist_red = cv2.calcHist([img], channels=[2], mask=None, histSize=[256], ranges=[0, 256])
        plt.plot(hist_blue, color="blue")
        plt.plot(hist_green, color="green")
        plt.plot(hist_red, color="red")
    elif channels == "gray":
        hist_gray = cv2.calcHist([img], channels=[0], mask=None, histSize=[256], ranges=[0, 256])
        plt.plot(hist_gray, color="gray")
    plt.xlim([0, 256])
    plt.show()


if __name__ == "__main__":
    lenna = cv2.imread("lenna.png")
    parrots = cv2.imread("Parrots.jpg", cv2.IMREAD_GRAYSCALE)
    camera_man = cv2.imread("cameraman.tif", cv2.IMREAD_GRAYSCALE)

    cv2.imshow("Lenna window", lenna)
    cv2.imshow("Parrots window", parrots)
    cv2.imshow("Camera man window", camera_man)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    show_histogram(lenna, "rgb")
    show_histogram(parrots, "gray")
    show_histogram(camera_man, "gray")
