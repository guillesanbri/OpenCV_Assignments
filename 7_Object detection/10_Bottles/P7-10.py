import cv2
import numpy as np
import matplotlib.pyplot as plt


def setup_subplot(subplt, name):
    subplt.set_title(name)
    subplt.get_xaxis().set_visible(False)
    subplt.get_yaxis().set_visible(False)


def closing(binary):
    k = np.ones((9, 9))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel=k, iterations=3)
    return binary


def generate_mask(binary, threshold=20000):
    mask = binary.copy()
    for i, row in enumerate(binary):
        if np.sum(row) > threshold:
            mask[i] = 255 * np.ones(row.shape)
        else:
            mask[i] = np.zeros(row.shape)
    return mask


def draw_bottle_level(img_hough, edges_img, threshold):
    lines = cv2.HoughLines(edges_img, 1, np.pi / 180, threshold)
    if lines is not None:
        for rho, theta in lines[-1]:  # Draw lower line
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(img_hough, (x1, y1), (x2, y2), (255, 0, 0), 4)
            cv2.putText(img_hough, str(img_hough.shape[0] - y1), (530, y1-20),
                        cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), thickness=2)


def bottle_level(img):
    _, binary = cv2.threshold(img, 125, 255, cv2.THRESH_BINARY_INV)
    binary = cv2.medianBlur(binary, 15)

    binary = closing(binary)
    generated_mask = generate_mask(binary)
    edges = cv2.Canny(generated_mask, 25, 75)

    result = cv2.merge((img.copy(), img.copy(), img.copy()))
    draw_bottle_level(result, edges, 200)

    return result, generated_mask, edges


if __name__ == "__main__":
    img = cv2.imread("botella6.bmp", cv2.IMREAD_GRAYSCALE)

    result, generated_mask, edges = bottle_level(img)

    fig, axs = plt.subplots(3, 1)
    setup_subplot(axs[0], "Generated mask")
    axs[0].imshow(cv2.cvtColor(generated_mask, cv2.COLOR_BGR2RGB), vmin=0, vmax=255)
    setup_subplot(axs[1], "Edges")
    axs[1].imshow(cv2.cvtColor(edges, cv2.COLOR_BGR2RGB), vmin=0, vmax=255)
    setup_subplot(axs[2], "Result")
    axs[2].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), vmin=0, vmax=255)
    plt.tight_layout()
    plt.show()

    bottles = {}
    for i in range(1, 7):
        name = "botella" + str(i) + ".bmp"
        bottle = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
        bottles[name] = bottle
    fig, axs = plt.subplots(3, 2)
    i = j = 0
    for name, bottle in bottles.items():
        setup_subplot(axs[j, i], name)
        level, _, _ = bottle_level(bottle)
        axs[j, i].imshow(cv2.cvtColor(level, cv2.COLOR_BGR2RGB), vmin=0, vmax=255)
        j += 1
        if j == 3:
            i += 1
            j = 0
    plt.tight_layout()
    plt.show()