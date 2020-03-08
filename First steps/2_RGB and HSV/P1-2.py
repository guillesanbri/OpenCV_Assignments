import cv2
from matplotlib import pyplot as plt
from matplotlib import cm


'''
    Calculamos la imagen en escala de grises a partir de la formula 0.30R + 0.59G + 0.11B 
'''
def generate_grayscale(img):
    return cv2.convertScaleAbs(0.3*img[:, :, 2] + 0.59*img[:, :, 1] + 0.11*img[:, :, 0])


def plot_grayscale(img):
    fig, axs = plt.subplots(1, 2)

    axs[0].set_title("Original")
    axs[0].get_xaxis().set_visible(False)
    axs[0].get_yaxis().set_visible(False)
    axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), vmin=0, vmax=255)

    axs[1].set_title("Grayscale")
    axs[1].get_xaxis().set_visible(False)
    axs[1].get_yaxis().set_visible(False)
    axs[1].imshow(generate_grayscale(img), cmap=cm.gray, vmin=0, vmax=255)

    fig.tight_layout()
    plt.show(block=False)


def plot_channels(img, title, labels):
    fig, axs = plt.subplots(2, 2)
    fig.suptitle(title)

    axs[0, 0].set_title(labels[0])
    axs[0, 0].get_xaxis().set_visible(False)
    axs[0, 0].get_yaxis().set_visible(False)
    axs[0, 0].imshow(img[:, :, 0], cmap=cm.gray, vmin=0, vmax=255)

    axs[0, 1].set_title(labels[1])
    axs[0, 1].get_xaxis().set_visible(False)
    axs[0, 1].get_yaxis().set_visible(False)
    axs[0, 1].imshow(img[:, :, 1], cmap=cm.gray, vmin=0, vmax=255)

    axs[1, 0].set_title(labels[2])
    axs[1, 0].get_xaxis().set_visible(False)
    axs[1, 0].get_yaxis().set_visible(False)
    axs[1, 0].imshow(img[:, :, 2], cmap=cm.gray, vmin=0, vmax=255)

    axs[1, 1].set_title("Original")
    axs[1, 1].get_xaxis().set_visible(False)
    axs[1, 1].get_yaxis().set_visible(False)
    axs[1, 1].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), vmin=0, vmax=255)

    fig.tight_layout()
    plt.show(block=False)


if __name__ == "__main__":
    img = cv2.imread("street-market.jpg")
    plot_channels(img, "RGB", ["Blue", "Green", "Red"])

    plot_grayscale(img)

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    plot_channels(img_hsv, "HSV", ["Hue", "Saturation", "Value"])

    plt.show()
