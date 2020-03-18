import cv2


def red_threshold(red):
    global img, original
    img[:, :, 2] = original[:, :, 2]
    img[:, :, 2][img[:, :, 2] > red] = 0
    cv2.imshow(window_title, img)


def green_threshold(green):
    global img, original
    img[:, :, 1] = original[:, :, 1]
    img[:, :, 1][img[:, :, 1] > green] = 0
    cv2.imshow(window_title, img)


def blue_threshold(blue):
    global img, original
    img[:, :, 0] = original[:, :, 0]
    img[:, :, 0][img[:, :, 0] > blue] = 0
    cv2.imshow(window_title, img)


if __name__ == "__main__":
    window_title = "Interactive thresholder"
    cv2.namedWindow(window_title)

    original = cv2.imread("superman.jpg")
    img = original.copy()

    cv2.createTrackbar("R", window_title, 255, 255, red_threshold)
    cv2.createTrackbar("G", window_title, 255, 255, green_threshold)
    cv2.createTrackbar("B", window_title, 255, 255, blue_threshold)

    cv2.imshow(window_title, img)
    cv2.waitKey()
