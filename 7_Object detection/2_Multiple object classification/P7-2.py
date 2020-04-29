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


def number_of_holes(binary):
    inverse = 255 - binary
    object_number, _ = cv2.connectedComponents(inverse, connectivity=4)
    return object_number - 2  # Background and object


def count_elements(individual_labels, thresholds):
    elements = {}
    for individual_label in individual_labels:
        _area = area(individual_label)
        _circularity = circularity(individual_label)
        _holes = number_of_holes(individual_label)
        _perimeter = perimeter(individual_label)
        if _area > thresholds["area_llaves"]:  # Es llave
            if _holes == 2:
                key_pieza = "Llaves de boca cerrada"
            else:
                key_pieza = "Llaves de boca abierta"
        else:  # No es llave
            if _circularity > thresholds["circularidad_tuercas_arandelas"]:  # Es tuerca o arandela
                if _perimeter > thresholds["perimetro_tuerca"]:  # Es tuerca
                    key_pieza = "Tuercas"
                else:  # Es arandela
                    key_pieza = "Arandelas"
            else:  # Es tornillo
                if _area > thresholds["area_tornillo_largo"]:  # Es tornillo largo
                    key_pieza = "Tornillos largos"
                else:  # Es tornillo corto
                    key_pieza = "Tornillos cortos"
        elements[key_pieza] = elements[key_pieza] + 1 if elements.get(key_pieza) is not None else 1
    return elements


def show_info(img, elements):
    fig, axs = plt.subplots(1, 2)
    setup_subplot(axs[0], "Original")
    axs[0].imshow(img, cmap=cm.gray, vmin=0, vmax=255)
    axs[1].set_xlim([0, 10])
    setup_subplot(axs[1], "")
    for i, element in enumerate(elements):
        axs[1].text(0.2, 0.9-0.1*i, "{}: {}".format(element, elements[element]), fontsize=15)
    plt.show()


if __name__ == "__main__":
    img_gray = cv2.imread("arandelas_tuercas.bmp", cv2.IMREAD_GRAYSCALE)
    img_gray = cv2.imread("llaves.bmp", cv2.IMREAD_GRAYSCALE)
    img_gray = cv2.imread("tuercas_tornillos.bmp", cv2.IMREAD_GRAYSCALE)
    img_gray = cv2.GaussianBlur(img_gray, (9, 9), 1)
    _, img_binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    k = np.ones((7, 7))
    img_binary = cv2.morphologyEx(img_binary, cv2.MORPH_CLOSE, kernel=k, iterations=2)

    thresholds = {"area_llaves": 15000,
                  "circularidad_tuercas_arandelas": 0.7,
                  "area_tornillo_largo": 1500,
                  "perimetro_tuerca": 100}

    object_number, labels = cv2.connectedComponents(img_binary, connectivity=4)

    _, individual_labels = generate_label_images(labels, object_number, threshold=250)

    elementos = count_elements(individual_labels, thresholds)

    show_info(img_gray, elementos)
