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


def perimeter(binary):
    contour, _ = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return cv2.arcLength(contour[0], True)


def area(binary):
    contour, _ = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return cv2.contourArea(contour[0])


def circularity(binary):
    return 4 * np.pi * area(binary) / (perimeter(binary) ** 2)


def centroid(binary):
    M = cv2.moments(binary)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return cX, cY


def circularity(binary):
    return 4 * np.pi * area(binary) / (perimeter(binary)**2)


def get_label_information(separated_labels):
    return [{"img": label,
             "centroid": centroid(label),
             "circularity": circularity(label)}
            for label in separated_labels]


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


def display_info(element_info_dict, unique_labels, color=(0, 0, 0)):
    complete_img = np.zeros(unique_labels[0].shape).astype(np.uint8)
    print(unique_labels[0].shape)
    for element_info, label in zip(element_info_dict, unique_labels):
        label_copy = label.copy()
        line_length = 40
        line_thickness = 10
        label_copy = draw_cross(label_copy, element_info["centroid"], line_length, line_thickness)
        cv2.putText(label_copy,
                    "{:.2f}".format(element_info["circularity"]),
                    (int(element_info["centroid"][0] - 2*line_length), int(element_info["centroid"][1] - 1.1*line_length)),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX,
                    fontScale=3,
                    color=color,
                    thickness=7)
        complete_img = cv2.add(complete_img, label_copy)
    return complete_img


def classify(img, element_info_dict, threshold):
    img_copy = img.copy()
    line_length = 40
    line_thickness = 15
    for element_info in element_info_dict:
        color = (0, 255, 0) if element_info["circularity"] < threshold else (0, 0, 255)
        img_copy = draw_cross(img_copy, element_info["centroid"], line_length, line_thickness, color)
    return img_copy


if __name__ == "__main__":
    img = cv2.imread("cuadradas-redondas.JPG")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.GaussianBlur(img_gray, (9, 9), 1)
    _, img_binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    k = np.ones((15, 15))
    img_binary = cv2.morphologyEx(img_binary, cv2.MORPH_CLOSE, kernel=k, iterations=2)

    object_number, labels = cv2.connectedComponents(img_binary, connectivity=4)
    clean_labels_img, separated_labels = generate_label_images(labels, object_number, threshold=50000)

    lut = gen_lut(object_number)
    color_labels = labels2rgb(clean_labels_img, lut)

    element_info_dict = get_label_information(separated_labels)
    info_img = display_info(element_info_dict, separated_labels)

    classified = classify(img, element_info_dict, 0.65)

    fig, axs = plt.subplots(2, 2)
    setup_subplot(axs[0, 0], "Original")
    axs[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), vmin=0, vmax=255)
    setup_subplot(axs[0, 1], "Labels")
    axs[0, 1].imshow(color_labels, vmin=0, vmax=255)
    setup_subplot(axs[1, 0], "Circularity")
    axs[1, 0].imshow(info_img, cmap=cm.gray, vmin=0, vmax=255)
    setup_subplot(axs[1, 1], "Classified")
    axs[1, 1].imshow(cv2.cvtColor(classified, cv2.COLOR_BGR2RGB), vmin=0, vmax=255)

    plt.tight_layout()
    plt.show()
