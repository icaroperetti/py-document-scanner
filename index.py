import cv2
import numpy as np
import pyzbar.pyzbar as pyzbar

img = cv2.imread("20240304_232810.jpg", 1)

width = min(int(img.shape[0] / 6), int(img.shape[1] / 6))
hight = max(int(img.shape[0] / 6), int(img.shape[1] / 6))

resize = cv2.resize(img, (width, (hight)))
img_ctr = resize.copy()

screen_res = 1280, 720
scale_width = screen_res[0] / min(img.shape[1], img.shape[0])
scale_hight = screen_res[1] / max(img.shape[1], img.shape[0])
scale = min(scale_width, scale_hight)

output_width = min(int(img.shape[1] * scale), int(img.shape[0] * scale))
output_hight = max(int(img.shape[1] * scale), int(img.shape[0] * scale))

# cv2.namedWindow('Output', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('Output', output_width, output_hight)


def empty(a):
    pass


# cv2.namedWindow("screen")
# cv2.createTrackbar("v1","screen",1,200,empty)
# cv2.createTrackbar("v2","screen",1,200,empty)
kernal = np.ones((5, 5))


def process(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 1)
    # v1 = cv2.getTrackbarPos("v1","screen")
    # v2 = cv2.getTrackbarPos("v2","screen")
    canny = cv2.Canny(blur, 50, 50)
    dilate = cv2.dilate(canny, kernal, iterations=2)
    erode = cv2.erode(dilate, kernal, iterations=1)

    return canny, erode, gray


def getcontour(img):
    large = np.array([])
    maxArea = 0
    contour, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for ctr in contour:
        area = cv2.contourArea(ctr)
        if area > 14000:
            cv2.drawContours(img_ctr, ctr, -1, (55, 255, 55), 3)
            peri = cv2.arcLength(ctr, True)
            approx = cv2.approxPolyDP(ctr, 0.02 * peri, True)
            if len(approx) == 4 and area > maxArea:
                large = approx
                maxArea = area
    cv2.drawContours(img_ctr, large, -1, (55, 0, 255), 15)
    return large


def reorder(points):
    points = points.reshape((4, 2))
    new_points = np.zeros((4, 1, 2), np.int32)
    add = np.sum(points, axis=1)
    # print('add = ',add)
    new_points[0] = points[np.argmin(add)]
    new_points[3] = points[np.argmax(add)]

    sub = np.diff(points, axis=1)
    new_points[1] = points[np.argmin(sub)]
    new_points[2] = points[np.argmax(sub)]
    # print('rearranged points =',new_points)

    return new_points


def cvtBlack(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # v1 = cv2.getTrackbarPos("v1","screen")
    (thres, black) = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY)
    return black


def crop(img, large):
    # print(large.shape)
    large = reorder(large)
    pt1 = np.float32(large)
    pt2 = np.float32([[0, 0], [width, 0], [0, hight], [width, hight]])
    matrix = cv2.getPerspectiveTransform(pt1, pt2)
    output = cv2.warpPerspective(img, matrix, (width, hight))
    cropped = output[10 : output.shape[0] - 10, 10 : output.shape[1] - 10]

    black = cvtBlack(cropped)

    return black, cropped


while True:
    canny, erode, gray = process(resize)
    large = getcontour(erode)
    img_crop, black = crop(resize, large)
    # cv2.imshow("img", resize)
    # cv2.imshow("gray", gray)
    # cv2.imshow("canny", canny)
    # cv2.imshow("erode", erode)
    # cv2.imshow("img_ctr", img_ctr)

    for barcode in pyzbar.decode(img_crop):
        myData = barcode.data.decode("utf-8")
        print(myData)
        pts = np.array([barcode.polygon], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(img_crop, [pts], True, (0, 255, 0), 3)
        pts2 = barcode.rect
        cv2.putText(
            img_crop,
            myData,
            (pts2[0], pts2[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 0, 0),
            2,
        )

        print("Barcode: ", myData)

    cv2.imshow("Output", img_crop)
    cv2.imshow("cropped", black)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break
    elif key == ord("s"):
        cv2.imwrite(name + "converted.jpg", img_crop)
        print("saved!!!!")
        break

cv2.destroyAllWindows()
