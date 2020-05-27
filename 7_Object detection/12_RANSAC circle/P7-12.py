import cv2
import numpy as np


def inliers_line(X, Y, line, tol):
    num_inliers = 0
    for x, y in zip(X, Y):
        distance = np.abs(float(y) - float(line[0]*x+line[1]))
        if distance < tol:
            num_inliers += 1
    return num_inliers


def inliers_circle(X, Y, circle, tol):
    num_inliers = 0
    for x, y in zip(X, Y):
        distance = np.abs(circle[2] - np.linalg.norm([x - circle[0], y - circle[1]]))
        if distance < tol:
            num_inliers += 1
    return num_inliers


def ransac_line(X, Y, iterations):
    best_model = (0, 0)
    max_inliers = 0
    for i in range(iterations):
        random1 = np.random.randint(X.shape[0])
        p1 = (X[random1], Y[random1])
        random2 = np.random.randint(X.shape[0])
        p2 = (X[random2], Y[random2])
        slope = (p1[1] - p2[1])/(p1[0] - p2[0])
        line = (slope, slope*-p1[0] + p1[1])  # m, b # y = mx + b
        inliers = inliers_line(X, Y, line, 1)
        if inliers > max_inliers:
            max_inliers = inliers
            best_model = line
    return best_model


def circle_three_points(p1, p2, p3):
    a = (p1[0]**2 + p1[1]**2)
    b = (p2[0]**2 + p2[1]**2)
    c = (p3[0]**2 + p3[1]**2)
    denominator = (2 * (p1[0] * (p2[1] - p3[1]) - p1[1] * (p2[0] - p3[0]) + p2[0] * p3[1] - p3[0] * p2[1]))
    xc = (a * (p2[1] - p3[1]) + b * (p3[1] - p1[1]) + c * (p1[1] - p2[1])) // denominator
    yc = (a * (p3[0] - p2[0]) + b * (p1[0] - p3[0]) + c * (p2[0] - p1[0])) // denominator
    r = int(np.sqrt((xc - p1[0])**2 + (yc - p1[1])**2))
    return xc, yc, r  # center x, center y, radius


def ransac_circle(X, Y, iterations):
    best_model = (0, 0)
    max_inliers = 0
    for i in range(iterations):
        random1 = np.random.randint(X.shape[0])
        p1 = (X[random1], Y[random1])
        random2 = np.random.randint(X.shape[0])
        p2 = (X[random2], Y[random2])
        random3 = np.random.randint(X.shape[0])
        p3 = (X[random3], Y[random3])
        try:
            circle = circle_three_points(p1, p2, p3)  # center x, center y, radiu
            inliers = inliers_circle(X, Y, circle, 2)
            if inliers > max_inliers:
                max_inliers = inliers
                best_model = circle
        except:
            pass
    return best_model


if __name__ == "__main__":

    img = cv2.imread("circulo.png")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    white_pixels = np.argwhere(binary == 255)
    X = np.array([[p[1]] for p in white_pixels])
    Y = np.array([[p[0]] for p in white_pixels])

    circle = ransac_circle(X, Y, 200)
    center = (circle[1][0], circle[0][0])
    cv2.circle(img, center, circle[2], (255, 0, 0), 3)

    '''
    line = ransac_line(X, Y, 50)
    line_x = np.arange(X.min(), X.max())[:, np.newaxis]
    line_y = np.array([line[0] * x + line[1] for x in line_x])
    for y, x in zip(line_y, line_x):
        img[int(y), int(x)] = (255, 0, 0)
    '''

    cv2.imshow("RANSAC", img)
    cv2.waitKey(0)