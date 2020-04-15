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


def generate_label_images(labeled_img, num_labels):
    h, w = labeled_img.shape
    label_images = []
    for k in range(1, num_labels):
        extraction = labeled_img.copy().astype(np.uint8)
        extraction[extraction != k] = 0
        extraction[extraction == k] = 255
        if np.sum(extraction)//255 > 75:
            label_images.append(extraction)
    return label_images


def get_centroids_and_inertia_theta(individual_images):
    out = {}
    centroids = []
    thetas = []
    for piece in individual_images:
        M = cv2.moments(piece)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        theta = 0.5 * np.arctan2(2*M["mu11"], (M["mu20"] - M["mu02"]))
        centroids.append((cX, cY))
        thetas.append(theta)
    out["centroids"] = centroids
    out["inertia_angles"] = thetas
    return out


def draw_centroids_inertia(img, elements):
    color_img = cv2.merge((img.copy(), img.copy(), img.copy()))
    for centroid, theta in zip(elements["centroids"], elements["inertia_angles"]):
        cv2.line(color_img,
                 (int(centroid[0] - 20 * np.cos(theta)), int(centroid[1] - 20 * np.sin(theta))),
                 (int(centroid[0] + 20 * np.cos(theta)), int(centroid[1] + 20 * np.sin(theta))),
                 (255, 0, 0),
                 thickness=2)
        cv2.line(color_img,
                 (int(centroid[0] - 7.5 * np.cos(theta+np.pi/2)), int(centroid[1] - 7.5 * np.sin(theta+np.pi/2))),
                 (int(centroid[0] + 7.5 * np.cos(theta+np.pi/2)), int(centroid[1] + 7.5 * np.sin(theta+np.pi/2))),
                 (255, 0, 0),
                 thickness=2)
        cv2.circle(color_img, centroid, 5, (0, 0, 255), -1)
    return color_img


if __name__ == "__main__":
    img_gray = cv2.imread("tuercas_tornillos3.bmp", cv2.IMREAD_GRAYSCALE)
    img_gray = cv2.GaussianBlur(img_gray, (9, 9), 1)
    _, img_binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    object_number, labels = cv2.connectedComponents(img_binary, connectivity=4)

    lut = gen_lut(object_number)
    color_labels = labels2rgb(labels, lut)

    individual_labels = generate_label_images(labels, object_number)
    centroids_and_inertia = get_centroids_and_inertia_theta(individual_labels)
    color_img = draw_centroids_inertia(img_binary, centroids_and_inertia)

    fig, axs = plt.subplots(2, 2)
    setup_subplot(axs[0, 0], "Original")
    axs[0, 0].imshow(img_gray, cmap=cm.gray, vmin=0, vmax=255)
    setup_subplot(axs[0, 1], "Binary")
    axs[0, 1].imshow(img_binary, cmap=cm.gray, vmin=0, vmax=255)
    setup_subplot(axs[1, 0], "Labels")
    axs[1, 0].imshow(color_labels, vmin=0, vmax=255)
    setup_subplot(axs[1, 1], "Centroids and inertia axis")
    axs[1, 1].imshow(color_img, vmin=0, vmax=255)
    plt.tight_layout()
    plt.show()