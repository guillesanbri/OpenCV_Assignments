import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def setup_subplot(subplt, name):
    subplt.set_title(name)
    subplt.get_xaxis().set_visible(False)
    subplt.get_yaxis().set_visible(False)


def generate_integral_image(img):
    integral = np.zeros((img.shape[0]+1, img.shape[1]+1), dtype="uint64")
    for r in range(1, integral.shape[0]):
        for c in range(1, integral.shape[1]):
            up = integral[r-1, c] if r > 0 else 0
            left = integral[r, c-1] if c > 0 else 0
            corner = integral[r-1, c-1] if r > 0 and c > 0 else 0
            actual = img[r-1, c-1] if r > 0 and c > 0 else 0
            integral[r, c] = actual + left + up - corner
    return integral


def get_window_mean(integral, row, column, width, height):
    x1, y1 = max(int(column-width/2), 0), max(int(row-height/2), 0)
    x2, y2 = min(int(column+width/2), integral.shape[1]-2), min(int(row+height/2), integral.shape[0]-2)
    window_sum = int(integral[y2, x2]) - integral[y2, x1] - integral[y1, x2] + integral[y1, x1]
    return window_sum // ((x2-x1) * (y2-y1))


def window_mean_threshold(pixel_value, mean, correction):
    return 0 if pixel_value < correction*mean else 255


def adaptative_threshold(img, size, correction):
    binary = np.zeros((img.shape[0], img.shape[1]), dtype="uint8")
    integral = generate_integral_image(img)
    for r in range(1, binary.shape[0]):
        for c in range(1, binary.shape[1]):
            window_mean = get_window_mean(integral, r, c, size[0], size[1])
            binary[r, c] = window_mean_threshold(img[r, c], window_mean, correction)
    return binary


if __name__ == "__main__":
    img = cv2.imread("text.bmp", cv2.IMREAD_GRAYSCALE)

    threshold_opencv = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 75, 0)
    threshold = adaptative_threshold(img, (75, 75), correction=0.95)

    fig, axs = plt.subplots(1, 3)
    setup_subplot(axs[0], "Original")
    axs[0].imshow(img, cmap=cm.gray, vmin=0, vmax=255)

    setup_subplot(axs[1], "OpenCV")
    axs[1].imshow(threshold_opencv, cmap=cm.gray, vmin=0, vmax=255)

    setup_subplot(axs[2], "adaptative_threshold")
    axs[2].imshow(threshold, cmap=cm.gray, vmin=0, vmax=255)

    fig.tight_layout()
    plt.show()

    fig, axs = plt.subplots(1, 2)

    setup_subplot(axs[0], "Otsu")
    axs[0].imshow(cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1], cmap=cm.gray, vmin=0, vmax=255)

    setup_subplot(axs[1], "adaptative_threshold")
    axs[1].imshow(threshold, cmap=cm.gray, vmin=0, vmax=255)

    fig.tight_layout()
    plt.show()
