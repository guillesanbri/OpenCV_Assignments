import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def setup_subplot(subplt, name):
    subplt.set_title(name)
    subplt.get_xaxis().set_visible(False)
    subplt.get_yaxis().set_visible(False)


# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html
if __name__ == "__main__":
    ref = cv2.imread("cereales_1.JPG")
    ref = cv2.resize(ref, (ref.shape[1]//4, ref.shape[0]//4))
    img1 = cv2.imread("cereales_2.jpg")
    img1 = cv2.resize(img1, (img1.shape[1]//4, img1.shape[0]//4))

    orb = cv2.ORB_create()
    keypoints0, descriptors0 = orb.detectAndCompute(ref, None)
    keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
    ref = cv2.drawKeypoints(ref, keypoints0, None)

    # Brute Force
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches1 = bf.match(descriptors0, descriptors1)
    matches1 = sorted(matches1, key=lambda x: x.distance)
    img1 = cv2.drawMatches(ref, keypoints0, img1, keypoints1, matches1[:25], outImg=img1, flags=2)

    fig, axs = plt.subplots(1, 2)
    setup_subplot(axs[0], "Original")
    axs[0].imshow(cv2.cvtColor(ref, cv2.COLOR_BGR2RGB), vmin=0, vmax=255)
    setup_subplot(axs[1], "Img")
    axs[1].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB), vmin=0, vmax=255)

    plt.tight_layout()
    plt.show()