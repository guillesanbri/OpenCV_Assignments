import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def setup_subplot(subplt, name):
    subplt.set_title(name)
    subplt.get_xaxis().set_visible(False)
    subplt.get_yaxis().set_visible(False)


def generate_label_images(labels_img, num_labels, threshold):
    clean_image = labels_img.copy().astype(np.uint8)
    separated_labels = []
    for k in range(1, num_labels):
        unique_label = labels_img.copy().astype(np.uint8)
        unique_label[unique_label != k] = 0
        unique_label[unique_label == k] = 255
        # print(np.sum(unique_label) // 255)
        if np.sum(unique_label) // 255 > threshold:
            separated_labels.append(unique_label)
        else:
            clean_image[clean_image == k] = 0
    return clean_image, separated_labels


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


def center_and_rotate(img_binary):
    center = centroid(img_binary)
    offset_x = img_binary.shape[1] // 2 - center[0]
    offset_y = img_binary.shape[0] // 2 - center[1]
    theta = inertia_angle(img_binary)

    affine_matrix_offset = np.float32([[1, 0, offset_x], [0, 1, offset_y]])
    img_centered = cv2.warpAffine(img_binary, affine_matrix_offset,
                                  (img_binary.shape[1], img_binary.shape[0]))

    affine_matrix_rotation = cv2.getRotationMatrix2D((img_centered.shape[1] // 2,
                                                      img_centered.shape[0] // 2),
                                                     theta * 180 / np.pi, 1.0)
    img_rotated = cv2.warpAffine(img_centered, affine_matrix_rotation,
                                 (img_centered.shape[1], img_centered.shape[0]))
    return img_rotated


def are_equal(img1, img2):
    w, h = img2.shape[0], img2.shape[1]
    rotation_180 = cv2.getRotationMatrix2D((h // 2, w // 2), 180, 1.0)

    img2_1 = img2.copy()
    img2_2 = cv2.warpAffine(img2, rotation_180, (h, w))
    img2_3 = cv2.flip(img2_1, +1)
    img2_4 = cv2.flip(img2_2, +1)

    fig, axs = plt.subplots(2, 2)
    setup_subplot(axs[0, 0], "Pixeles de diferencia: {}".format(np.sum(cv2.absdiff(img1, img2_1))//255))
    axs[0, 0].imshow(cv2.absdiff(img1, img2_1), cmap=cm.gray, vmin=0, vmax=255)
    setup_subplot(axs[0, 1], "Pixeles de diferencia: {}".format(np.sum(cv2.absdiff(img1, img2_2))//255))
    axs[0, 1].imshow(cv2.absdiff(img1, img2_2), cmap=cm.gray, vmin=0, vmax=255)
    setup_subplot(axs[1, 0], "Pixeles de diferencia: {}".format(np.sum(cv2.absdiff(img1, img2_3))//255))
    axs[1, 0].imshow(cv2.absdiff(img1, img2_3), cmap=cm.gray, vmin=0, vmax=255)
    setup_subplot(axs[1, 1], "Pixeles de diferencia: {}".format(np.sum(cv2.absdiff(img1, img2_4))//255))
    axs[1, 1].imshow(cv2.absdiff(img1, img2_4), cmap=cm.gray, vmin=0, vmax=255)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    img = cv2.imread("parllaves2.JPG")
    img = cv2.resize(img, (img.shape[1]//4, img.shape[0]//4))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.GaussianBlur(img_gray, (9, 9), 1)
    _, img_binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    object_number, labels = cv2.connectedComponents(img_binary, connectivity=4)
    clean_labels_img, separated_labels = generate_label_images(labels, object_number, threshold=10000)

    fixed_pieces = []
    for piece in separated_labels:
        centered_rotated = center_and_rotate(piece)
        fixed_pieces.append(centered_rotated)

    fig, axs = plt.subplots(1, 3)
    setup_subplot(axs[0], "Original")
    axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), vmin=0, vmax=255)
    setup_subplot(axs[1], "Piece 1")
    axs[1].imshow(fixed_pieces[0], cmap=cm.gray, vmin=0, vmax=255)
    setup_subplot(axs[2], "Piece 2")
    axs[2].imshow(fixed_pieces[1], cmap=cm.gray, vmin=0, vmax=255)
    plt.tight_layout()
    plt.show()

    are_equal(fixed_pieces[0], fixed_pieces[1])