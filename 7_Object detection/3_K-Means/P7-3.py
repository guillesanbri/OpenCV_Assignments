import cv2
import numpy as np


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
        if np.sum(unique_label) // 255 > threshold:
            separated_labels.append(unique_label)
        else:
            clean_image[clean_image == k] = 0
    return clean_image, separated_labels


def perimeter(binary):
    contour, _ = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return cv2.arcLength(contour[0], True)


def area(binary):
    contour, _ = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return cv2.contourArea(contour[0])


def circularity(binary):
    return 4 * np.pi * area(binary) / (perimeter(binary)**2)


def centroid(binary):
    M = cv2.moments(binary)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return cX, cY


def number_of_holes(binary):
    inverse = 255 - binary
    object_number, _ = cv2.connectedComponents(inverse, connectivity=4)
    return object_number - 2  # Background and object


def gen_lut(num):
    lut = np.zeros((num, 3))
    for i in range(0, num):
        lut[i] = random_color()
    return lut


def random_color():
    return np.array(np.random.choice(range(256), size=3))


def draw_cross(img, center, length, thickness, color=(0, 0, 0)):
    img_copy = img.copy()
    cv2.line(img_copy,
             (int(center[0] - length), int(center[1])),
             (int(center[0] + length), int(center[1])),
             color,
             thickness=thickness)
    cv2.line(img_copy,
             (int(center[0]), int(center[1] - length)),
             (int(center[0]), int(center[1] + length)),
             color,
             thickness=thickness)
    return img_copy


def classify(img, individual_blob, labels, k):
    img_copy = img.copy()
    line_length = 4
    line_thickness = 2
    colors = gen_lut(k)
    for i, blob in enumerate(individual_blob):
        img_copy = draw_cross(img_copy, centroid(blob), line_length, line_thickness, colors[labels[i][0]])
    return img_copy


if __name__ == "__main__":

    k = int(input("Introduce el numero de tipos que hay: "))

    img = cv2.imread("tuercas_tornillos.bmp")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.GaussianBlur(img_gray, (9, 9), 1)
    _, img_binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((7, 7))
    img_binary = cv2.morphologyEx(img_binary, cv2.MORPH_CLOSE, kernel=kernel, iterations=2)

    object_number, labels = cv2.connectedComponents(img_binary, connectivity=4)
    _, individual_labels = generate_label_images(labels, object_number, threshold=250)

    X = np.array([number_of_holes(individual_label) for individual_label in individual_labels])
    Y = np.array([np.sum(individual_label) for individual_label in individual_labels])

    Z = np.array([X, Y]).T
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_MAX_ITER, 100, 1)
    ret, result, center = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    classified = classify(img, individual_labels, result, k)

    cv2.imshow("Result", classified)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
