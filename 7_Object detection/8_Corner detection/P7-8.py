import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def setup_subplot(subplt, name):
    subplt.set_title(name)
    subplt.get_xaxis().set_visible(False)
    subplt.get_yaxis().set_visible(False)


def centroid(binary):
    M = cv2.moments(binary)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return cX, cY


def area(binary):
    contour, _ = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return cv2.contourArea(contour[0])


def draw_cross(img, center, lengthx, lengthy, thickness, color=(0, 0, 0)):
    img_copy = img.copy()
    cv2.line(img_copy,
             (int(center[0] - lengthy), int(center[1])),
             (int(center[0] + lengthy), int(center[1])),
             color,
             thickness=thickness)
    cv2.line(img_copy,
             (int(center[0]), int(center[1] - lengthx)),
             (int(center[0]), int(center[1] + lengthx)),
             color,
             thickness=thickness)
    return img_copy


def generate_label_images(labels_img, num_labels, threshold):
    clean_image = labels_img.copy().astype(np.uint8)
    separated_labels = []
    for k in range(1, num_labels):
        unique_label = labels_img.copy().astype(np.uint8)
        unique_label[unique_label != k] = 0
        unique_label[unique_label == k] = 255
        if np.sum(unique_label) // 255 > threshold:
            separated_labels.append(unique_label)
        else:
            clean_image[clean_image == k] = 0
    return clean_image, separated_labels


def find_corners(pixels, center):
    pixelsq1 = np.array([pixel for pixel in pixels if pixel[0] < center[1] and pixel[1] < center[0]])
    q1_distances = np.array([np.linalg.norm((pixel[1] - center[0], pixel[0] - center[1])) for pixel in pixelsq1])
    pixelsq2 = np.array([pixel for pixel in pixels if pixel[0] < center[1] and pixel[1] > center[0]])
    q2_distances = np.array([np.linalg.norm((pixel[1] - center[0], pixel[0] - center[1])) for pixel in pixelsq2])
    pixelsq3 = np.array([pixel for pixel in pixels if pixel[0] > center[1] and pixel[1] < center[0]])
    q3_distances = np.array([np.linalg.norm((pixel[1] - center[0], pixel[0] - center[1])) for pixel in pixelsq3])
    pixelsq4 = np.array([pixel for pixel in pixels if pixel[0] > center[1] and pixel[1] > center[0]])
    q4_distances = np.array([np.linalg.norm((pixel[1] - center[0], pixel[0] - center[1])) for pixel in pixelsq4])
    i1, _ = max(enumerate(q1_distances), key= lambda dist: dist[1])
    i2, _ = max(enumerate(q2_distances), key= lambda dist: dist[1])
    i3, _ = max(enumerate(q3_distances), key= lambda dist: dist[1])
    i4, _ = max(enumerate(q4_distances), key= lambda dist: dist[1])
    return pixelsq1[i1], pixelsq2[i2], pixelsq3[i3], pixelsq4[i4]


if __name__ == "__main__":
    img_og = cv2.imread("matricula_coche_1.jpg")
    img = img_og.copy()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img_binary = cv2.threshold(img_gray, 170, 255, cv2.THRESH_BINARY)

    k = np.ones((5, 5))
    img_open = cv2.morphologyEx(img_binary, cv2.MORPH_OPEN, kernel=k, iterations=2)
    img_close = cv2.morphologyEx(img_open, cv2.MORPH_CLOSE, kernel=k, iterations=5)

    object_number, labels = cv2.connectedComponents(img_close, connectivity=4)
    clean_labels_img, separated_labels = generate_label_images(labels, object_number, threshold=35000)

    separated_labels.sort(key=lambda label: area(label))
    license_plate = separated_labels[-1]

    center = centroid(license_plate)
    img = draw_cross(img, center, 85, 300, 4, (255, 0, 0))
    white_pixels = np.argwhere(license_plate == 255)
    c1, c2, c3, c4 = find_corners(white_pixels, center)
    img = draw_cross(img, (c1[1], c1[0]), 10, 10, 4, (0, 255, 0))
    img = draw_cross(img, (c2[1], c2[0]), 10, 10, 4, (0, 255, 0))
    img = draw_cross(img, (c3[1], c3[0]), 10, 10, 4, (0, 255, 0))
    img = draw_cross(img, (c4[1], c4[0]), 10, 10, 4, (0, 255, 0))

    scr_points_perspective = np.array([
        (c1[1], c1[0]),
        (c2[1], c2[0]),
        (c3[1], c3[0]),
        (c4[1], c4[0])
    ], dtype=np.float32)
    dst_points_perspective = np.array([
        (100, 200),
        (700, 200),
        (100, 300),
        (700, 300)
    ], dtype=np.float32)
    M_perspective = cv2.getPerspectiveTransform(scr_points_perspective, dst_points_perspective)
    img_warped_perspective = cv2.warpPerspective(img_og.copy(), M_perspective, (img.shape[1], img.shape[0]))

    fig, axs = plt.subplots(2, 2)
    setup_subplot(axs[0, 0], "Original")
    axs[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), vmin=0, vmax=255)
    setup_subplot(axs[0, 1], "Binary closed")
    axs[0, 1].imshow(img_close, cmap=cm.gray, vmin=0, vmax=255)
    setup_subplot(axs[1, 0], "Filtered blob")
    axs[1, 0].imshow(license_plate, cmap=cm.gray, vmin=0, vmax=255)
    setup_subplot(axs[1, 1], "Warped")
    axs[1, 1].imshow(cv2.cvtColor(img_warped_perspective, cv2.COLOR_BGR2RGB), vmin=0, vmax=255)

    plt.tight_layout()
    plt.show()