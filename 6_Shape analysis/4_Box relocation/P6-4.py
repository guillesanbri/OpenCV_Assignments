import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def setup_subplot(subplt, name):
    subplt.set_title(name)
    subplt.get_xaxis().set_visible(False)
    subplt.get_yaxis().set_visible(False)


def transform(gray, transf_matrix):
    transformed = np.zeros(gray.shape, np.uint8)
    inv_transf_matrix = np.linalg.inv(transf_matrix)
    for row in range(transformed.shape[0]):
        for col in range(transformed.shape[1]):
            p = np.array([row, col, 1])
            transf_p = np.matmul(inv_transf_matrix, p)
            if 0 < int(transf_p[0]) < gray.shape[0] and 0 < int(transf_p[1]) < gray.shape[1]:
                transformed[p[0]][p[1]] = gray[int(transf_p[0])][int(transf_p[1])]
    return transformed


def inertia_angle(binary):
    M = cv2.moments(binary)
    return 0.5 * np.arctan2(2 * M["mu11"], (M["mu20"] - M["mu02"]))


def centroid(binary):
    M = cv2.moments(binary)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return cX, cY


if __name__ == "__main__":
    img_gray = cv2.imread("caja08.png", cv2.IMREAD_GRAYSCALE)
    _, img_binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    img_binary = 255 - img_binary
    k = np.ones((9, 9))
    img_binary = cv2.morphologyEx(img_binary, cv2.MORPH_CLOSE, kernel=k, iterations=5)

    center = centroid(img_binary)
    offset_x = img_gray.shape[1]//2 - center[0]
    offset_y = img_gray.shape[0]//2 - center[1]

    theta = inertia_angle(img_binary)

    affine_matrix_offset = np.array([[1, 0, offset_y], [0, 1, offset_x], [0, 0, 1]])
    img_centered = transform(img_gray, affine_matrix_offset)
    mask_centered = transform(img_binary, affine_matrix_offset)

    affine_matrix_rotation = cv2.getRotationMatrix2D((img_centered.shape[1]//2,
                                                      img_centered.shape[0]//2),
                                                     theta*180/np.pi, 1.0)
    img_rotated = cv2.warpAffine(img_centered, affine_matrix_rotation,
                                 (img_centered.shape[1], img_centered.shape[0]))
    mask_rotated = cv2.warpAffine(mask_centered, affine_matrix_rotation,
                                  (img_centered.shape[1], img_centered.shape[0]))

    img_fixed = np.multiply(img_rotated, mask_rotated//255)

    if np.sum(img_fixed[:, :img_centered.shape[0]//2]) > 2000000:
        affine_matrix_fix = cv2.getRotationMatrix2D((img_centered.shape[1] // 2,
                                                    img_centered.shape[0] // 2),
                                                    180,
                                                    1.0)
        img_fixed = cv2.warpAffine(img_fixed, affine_matrix_fix, (img_centered.shape[1], img_centered.shape[0]))

    fig, axs = plt.subplots(1, 4)
    setup_subplot(axs[0], "Original")
    axs[0].imshow(img_gray, cmap=cm.gray, vmin=0, vmax=255)
    setup_subplot(axs[1], "Centered")
    axs[1].imshow(img_centered, cmap=cm.gray, vmin=0, vmax=255)
    setup_subplot(axs[2], "Rotated")
    axs[2].imshow(img_rotated, cmap=cm.gray, vmin=0, vmax=255)
    setup_subplot(axs[3], "Fixed")
    axs[3].imshow(img_fixed, cmap=cm.gray, vmin=0, vmax=255)

    plt.tight_layout()
    plt.show()