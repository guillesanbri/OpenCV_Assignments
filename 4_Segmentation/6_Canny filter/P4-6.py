import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def setup_subplot(subplt, name):
    subplt.set_title(name)
    subplt.get_xaxis().set_visible(False)
    subplt.get_yaxis().set_visible(False)


def generate_gaussian_noise(img, mean=0, var=0.01):
    row, col = img.shape
    sigma = var**0.5
    gauss = 255 * np.array(np.random.normal(mean, sigma, (row, col)))
    noisy_image = np.clip((img + gauss), 0, 255)
    noisy_image = noisy_image.astype(np.uint8)
    return noisy_image


if __name__ == "__main__":
    img = cv2.imread("burano.jpg")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_gaussian_noise = generate_gaussian_noise(img_gray)

    canny_50_80 = cv2.Canny(img_gray, 50, 80)
    canny_300_350 = cv2.Canny(img_gray, 300, 350)
    canny_450_500 = cv2.Canny(img_gray, 450, 500)

    canny_450_500_noise = cv2.Canny(img_gaussian_noise, 450, 500)

    fig, axs = plt.subplots(2, 2)
    setup_subplot(axs[0, 0], "Original")
    axs[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), cmap=cm.gray, vmin=0, vmax=255)
    setup_subplot(axs[0, 1], "Canny th1=50, th2=80")
    axs[0, 1].imshow(canny_50_80, cmap=cm.gray, vmin=0, vmax=255)
    setup_subplot(axs[1, 0], "Canny th1=300, th2=350")
    axs[1, 0].imshow(canny_300_350, cmap=cm.gray, vmin=0, vmax=255)
    setup_subplot(axs[1, 1], "Canny th1=450, th2=500")
    axs[1, 1].imshow(canny_450_500, cmap=cm.gray, vmin=0, vmax=255)
    plt.tight_layout()
    plt.show(block=False)

    fig, axs = plt.subplots(2, 2)
    setup_subplot(axs[0, 0], "Original grayscale")
    axs[0, 0].imshow(img_gray, cmap=cm.gray, vmin=0, vmax=255)
    setup_subplot(axs[0, 1], "Original grayscale + gaussian noise")
    axs[0, 1].imshow(img_gaussian_noise, cmap=cm.gray, vmin=0, vmax=255)
    setup_subplot(axs[1, 0], "Canny th1=450, th2=500")
    axs[1, 0].imshow(canny_450_500, cmap=cm.gray, vmin=0, vmax=255)
    setup_subplot(axs[1, 1], "Canny th1=450, th2=500 (Gaussian noise)")
    axs[1, 1].imshow(canny_450_500_noise, cmap=cm.gray, vmin=0, vmax=255)
    plt.tight_layout()
    plt.show()