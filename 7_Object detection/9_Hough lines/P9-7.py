import cv2
import numpy as np
import matplotlib.pyplot as plt


def setup_subplot(subplt, name):
    subplt.set_title(name)
    subplt.get_xaxis().set_visible(False)
    subplt.get_yaxis().set_visible(False)


def draw_hough_lines(img_hough, edges_img, threshold):
    lines = cv2.HoughLines(edges_img, 1, np.pi / 180, threshold)
    if lines is not None:
        for line in lines:
            for rho, theta in line:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                cv2.line(img_hough, (x1, y1), (x2, y2), (0, 255, 0), 2)


if __name__ == "__main__":
    img = cv2.imread("bridge.jpg")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.GaussianBlur(img_gray, (5, 5), 1)
    edges = cv2.Canny(img_gray, 175, 200)

    img_100 = img.copy()
    img_150 = img.copy()
    img_250 = img.copy()

    draw_hough_lines(img_100, edges, 100)
    draw_hough_lines(img_150, edges, 150)
    draw_hough_lines(img_250, edges, 250)

    fig, axs = plt.subplots(3, 1)
    setup_subplot(axs[0], "Threshold 100")
    axs[0].imshow(cv2.cvtColor(img_100, cv2.COLOR_BGR2RGB), vmin=0, vmax=255)
    setup_subplot(axs[1], "Threshold 150")
    axs[1].imshow(cv2.cvtColor(img_150, cv2.COLOR_BGR2RGB), vmin=0, vmax=255)
    setup_subplot(axs[2], "Threshold 250")
    axs[2].imshow(cv2.cvtColor(img_250, cv2.COLOR_BGR2RGB), vmin=0, vmax=255)
    plt.tight_layout()
    plt.show()