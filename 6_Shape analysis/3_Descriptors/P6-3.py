import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def setup_subplot(subplt, name):
    subplt.set_title(name)
    subplt.get_xaxis().set_visible(False)
    subplt.get_yaxis().set_visible(False)


def gen_lut(num):
    lut = np.zeros((num, 3))
    for i in range(1, num):
        lut[i] = random_color()
    return lut


def labels2rgb(labeled_img, lut):
    h, w = labeled_img.shape
    out = cv2.merge((labeled_img.copy(), labeled_img.copy(), labeled_img.copy()))
    for i in range(0, h):
        for j in range(0, w):
            out[i, j] = lut[labeled_img[i, j]]
    return out


def random_color():
    return np.array(np.random.choice(range(256), size=3))


def generate_label_images(labeled_img, num_labels, threshold):
    h, w = labeled_img.shape
    label_images = []
    for k in range(1, num_labels):
        extraction = labeled_img.copy().astype(np.uint8)
        extraction[extraction != k] = 0
        extraction[extraction == k] = 255
        # print(np.sum(extraction)//255)
        if np.sum(extraction)//255 > threshold:
            label_images.append(extraction)
    return label_images


def perimeter(binary):
    contour, _ = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return cv2.arcLength(contour[0], True)


def area(binary):
    contour, _ = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return cv2.contourArea(contour[0])


def inertia_angle(binary):
    M = cv2.moments(binary)
    return 0.5 * np.arctan2(2 * M["mu11"], (M["mu20"] - M["mu02"]))


def circularity(binary):
    return 4 * np.pi * area(binary) / (perimeter(binary)**2)


def number_of_holes(binary):
    inverse = 255 - binary
    object_number, _ = cv2.connectedComponents(inverse, connectivity=4)
    return object_number - 2  # Background and object


def centroid(binary):
    M = cv2.moments(binary)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return cX, cY


def compacity(binary):
    return area(binary) / (perimeter(binary)**2)


def random_descriptors(pieces):  # pieces -> Array de blobs
    index = np.random.randint(low=0, high=len(pieces))
    fig, axs = plt.subplots(1, 2)
    setup_subplot(axs[0], "Piece")
    axs[0].imshow(pieces[index], cmap=cm.gray, vmin=0, vmax=255)
    axs[1].set_xlim([0, 10])
    setup_subplot(axs[1], "")
    axs[1].text(0.2, 0.9, "Perimeter: {:.3f}".format(perimeter(pieces[index])), fontsize=10)
    axs[1].text(0.2, 0.8, "Area: {:.3f}".format(area(pieces[index])), fontsize=10)
    axs[1].text(0.2, 0.7, "Inertia axis angle: {:.3f}".format(inertia_angle(pieces[index])), fontsize=10)
    axs[1].text(0.2, 0.6, "Circularity: {:.3f}".format(circularity(pieces[index])), fontsize=10)
    axs[1].text(0.2, 0.5, "Holes: {}".format(number_of_holes(pieces[index])), fontsize=10)
    axs[1].text(0.2, 0.4, "Centroid: {}".format(centroid(pieces[index])), fontsize=10)
    axs[1].text(0.2, 0.3, "Compacity: {:.3f}".format(compacity(pieces[index])), fontsize=10)
    plt.show()


if __name__ == "__main__":
    img_gray = cv2.imread("cuadradas-redondas.JPG", cv2.IMREAD_GRAYSCALE)
    img_gray = cv2.imread("rotas-enteras.JPG", cv2.IMREAD_GRAYSCALE)
    img_gray = cv2.imread("estrella2.JPG", cv2.IMREAD_GRAYSCALE)
    img_gray = cv2.GaussianBlur(img_gray, (9, 9), 1)
    _, img_binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    k = np.ones((9, 9))
    img_binary = cv2.morphologyEx(img_binary, cv2.MORPH_CLOSE, kernel=k, iterations=2)

    object_number, labels = cv2.connectedComponents(img_binary, connectivity=4)

    lut = gen_lut(object_number)
    color_labels = labels2rgb(labels, lut)

    individual_labels = generate_label_images(labels, object_number, threshold=50000)

    while True:
        random_descriptors(individual_labels)