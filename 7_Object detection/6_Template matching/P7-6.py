import cv2
import numpy as np


# https://www.pyimagesearch.com/2015/03/09/capturing-mouse-click-events-with-python-and-opencv/
def click_and_crop(event, x, y, flags, param):
    global img, template, selected, points, drawing
    if drawing:
        rectangle_copy = img.copy()
        cv2.rectangle(rectangle_copy, points[0], (x, y), (0, 255, 0), 2)
        cv2.imshow("template matching", rectangle_copy)
    if event == cv2.EVENT_LBUTTONDOWN:
        points = [(x, y)]
        drawing = True
    elif event == cv2.EVENT_LBUTTONUP:
        points.append((x, y))
        selected = True
        drawing = False
        template = img[points[0][1]:points[1][1], points[0][0]:points[1][0]]


if __name__ == "__main__":
    img = cv2.imread("dibujosanimados.png")
    cv2.namedWindow("template matching")
    cv2.setMouseCallback("template matching", click_and_crop)
    cv2.imshow("template matching", img)
    selected = drawing = False
    template = points = None
    while not selected:
        cv2.waitKey(0)
    cv2.imshow("template", template)

    res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    _, w, h = template.shape[::-1]
    loc = np.where(res >= 0.8)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img, (pt[0], pt[1]), (pt[0] + w, pt[1] + h), (255, 0, 0), 2)
    cv2.imshow("template matching", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
