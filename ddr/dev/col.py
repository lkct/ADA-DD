import cv2
from glob import glob

filenames = glob("./*.jpg")
for f in filenames:
    img = cv2.imread(f)
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
            if img[i, j, 0] < 80:
                oimg[i, j] = 0
    img[:, :, 0] = img[:, :, 2]
    img[:, :, 1] = img[:, :, 2]

    # img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    out = oimg[0:900, 0:1600, :]
    cv2.imshow(f, out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #break
