import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def setup_subplot(subplt, name):
    subplt.set_title(name)
    subplt.get_xaxis().set_visible(False)
    subplt.get_yaxis().set_visible(False)


def find_endpoints(img):
    row, col = img.shape
    ends = []
    for i in range(1, col):
        for j in range(1, row):
            try:
                if img[j, i] == 255:
                    neighbours = img[j-1:j+2, i-1:i+2]
                    num_neigh = np.sum(neighbours)/255
                    if int(num_neigh) == 2:
                        ends.append((i, j))
            except:
                pass
    return ends


# TODO
def find_branchpoints(img):
    row, col = img.shape
    branches = []
    for i in range(2, col):
        for j in range(2, row):
            try:
                if img[j, i] == 255:
                    neighbours = img[j-2:j+3, i-2:i+3]
                    num_neigh = np.sum(neighbours)/255
                    if int(num_neigh) >= 7:
                        branches.append((i, j))
            except:
                pass
    return branches


if __name__ == "__main__":
    img_gray = cv2.imread("estrella2.jpg", cv2.IMREAD_GRAYSCALE)
    _, img_binary = cv2.threshold(img_gray, 120, 255, cv2.THRESH_BINARY_INV)

    skeleton = cv2.ximgproc.thinning(img_binary, cv2.ximgproc.THINNING_GUOHALL)
    end_points = find_endpoints(skeleton)
    branch_points = find_branchpoints(skeleton)
    print(branch_points)

    ends_highlight = cv2.merge((img_binary.copy(), img_binary.copy(), img_binary.copy()))
    branches_highlight = ends_highlight.copy()
    for point in end_points:
        cv2.circle(ends_highlight, point, 5, (0, 0, 255), thickness=10)

    for point in branch_points:
        cv2.circle(branches_highlight, point, 5, (0, 255, 0), thickness=10)

    print("Closed" if len(end_points) == 0 else "Open")

    fig, axs = plt.subplots(2, 2)
    setup_subplot(axs[0, 0], "Original")
    axs[0, 0].imshow(img_gray, cmap=cm.gray, vmin=0, vmax=255)
    setup_subplot(axs[0, 1], "Skeleton")
    axs[0, 1].imshow(skeleton, cmap=cm.gray, vmin=0, vmax=255)
    setup_subplot(axs[1, 0], "End points")
    axs[1, 0].imshow(ends_highlight, vmin=0, vmax=255)
    #setup_subplot(axs[1, 1], "Branch points")
    #axs[1, 1].imshow(branches_highlight, vmin=0, vmax=255)
    plt.tight_layout()
    plt.show()