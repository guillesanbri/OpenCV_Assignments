import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def setup_subplot(subplt, name):
    subplt.set_title(name)
    subplt.get_xaxis().set_visible(False)
    subplt.get_yaxis().set_visible(False)


# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_features_harris/py_features_harris.html
if __name__ == "__main__":
    img = cv2.imread("rubik.jpg")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_gray = np.float32(img_gray)
    harris = cv2.cornerHarris(img_gray, 3, 3, 0.04)

    # Dilatamos para ensanchar los puntos
    dst = cv2.dilate(harris, None)

    corners = img.copy()
    corners[dst > 0.01 * dst.max()] = [0, 0, 255]

    fig, axs = plt.subplots(1, 2)
    setup_subplot(axs[0], "Original")
    axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), vmin=0, vmax=255)
    setup_subplot(axs[1], "Corners")
    axs[1].imshow(cv2.cvtColor(corners, cv2.COLOR_BGR2RGB), vmin=0, vmax=255)

    plt.tight_layout()
    plt.show()