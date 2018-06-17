import cv2
from glob import glob


def proc(img):
    img = img[525:, :, :]
    img = cv2.resize(img, None, fx=0.5, fy=0.5)
    img = 255 - img
    oimg = img
    _, img = cv2.threshold(img, 63, 0, cv2.THRESH_TOZERO)
    img = 255 - img
    _, img = cv2.threshold(img, 63, 0, cv2.THRESH_TOZERO)
    img = 255 - img
    img = img * 2 - 128
    _, img = cv2.threshold(img, 128, 0, cv2.THRESH_TOZERO)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    h, w, _ = img.shape
    for i in range(h):
        for j in range(w):
            if img[i, j, 0] < 88:
                oimg[i, j] = 0
    img[:, :, 2] = img[:, :, 0]
    img[:, :, 1] = img[:, :, 0]

    # img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    oimg = cv2.dilate(oimg, kernel, iterations=2)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    oimg = cv2.erode(oimg, kernel)
    return oimg


if __name__=='__main__':
    filenames = glob("*.jpg")
    for f in filenames:
        img = cv2.imread(f)

        oimg = proc(img)

        out = oimg[-900:-1, 0:1600, :]
        cv2.imshow(f, out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        cv2.imwrite('../preproc/'+f, oimg)

        # break
