import cv2
import numpy as np
import matplotlib.pyplot as plt


def setup_subplot(subplt, name):
    subplt.set_title(name)
    subplt.get_xaxis().set_visible(False)
    subplt.get_yaxis().set_visible(False)


# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_houghcircles/py_houghcircles.html
def draw_hough_circles(img_draw, img_gray):
    circles = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT, 1, 50,
                               param1=50, param2=35, minRadius=40, maxRadius=125)
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # Outer circle
        cv2.circle(img_draw, (i[0], i[1]), i[2], (0, 0, 255), 6)
        # Center of the circle
        cv2.circle(img_draw, (i[0], i[1]), 2, (0, 0, 255), 3)


if __name__ == "__main__":
    img = cv2.imread("grapes.jpg")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.medianBlur(img_gray, 15)

    draw_hough_circles(img, img_gray)

    fig, axs = plt.subplots(1, 1)
    setup_subplot(axs, "Hough circles")
    axs.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), vmin=0, vmax=255)
    plt.tight_layout()
    plt.show()