#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PIL import Image, ImageFont
import cv2
import numpy as np

from detect import detect, crop_detection, annotate_detection
from load_labels import get_data
from recognize import cvtrain, sktrain, preprocess
from recognize import annotate_recognition
import pred
from glob import glob
import os

SAMPLE_SIZE = (28, 28)
SZ = 28
LABEL_FILE = '../MNIST/train-labels.idx1-ubyte'
IMAGE_FILE = '../MNIST/train-images.idx3-ubyte'
CASCADE_FILE = '../classifier/cascade.xml'
TEST_FILES = '../preproc/'
RESULT_FILES = '../results/'

FONT_FILE = 'UbuntuMono-R.ttf'
FONT_SIZE = 24
TEST_FONT = '5'
TRAIN_SIZE = 10000

bin_n = 16  # Number of bins
# svm_params = dict(kernel_type=cv2.SVM_LINEAR,
#                  svm_type=cv2.SVM_C_SVC,
#                  C=2.67, gamma=5.383)

affine_flags = cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR


def overlap(d1, d2):
    x11, y11, x12, y12 = (d1[0], d1[1], d1[0]+d1[2], d1[1]+d1[3])
    x21, y21, x22, y22 = (d2[0], d2[1], d2[0]+d2[2], d2[1]+d2[3])
    return (x11 <= x21) and (y11 <= y21) and(x12 >= x22) and (y12 >= y22)


def choose(digits, lab, prob):
    chosen = []
    n = lab.size
    for i in range(n):
        if prob[i] >= 0.5:
            continue
        flag = False
        for j in range(n):
            if j == i:
                continue
            if overlap(digits[i], digits[j]):
                flag = True
                break
        if not flag:
            chosen.append(i)
    chosen = np.uint8(chosen)
    digits = digits[chosen]
    lab = lab[chosen]
    prob = prob[chosen]
    return digits, lab, prob


def main():
    images, labels, num, rows, cols = get_data(LABEL_FILE,
                                               IMAGE_FILE)

    filenames = glob(TEST_FILES + "*.jpg")
    for filename in filenames:
        print 'Processing', filename
        img = cv2.imread(filename)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        im = Image.open(filename)
        im = im.convert('L')
        digits = detect(gray, CASCADE_FILE)
        results = crop_detection(im.copy(), digits)
        test = np.float32([np.float32(i.resize(SAMPLE_SIZE)) for i in results])
        test = np.tile(test.reshape((-1, 1, 28, 28)), (1, 3, 1, 1))

        # yhat:list of str=label+prob
        lab, prob = pred.main(test)

        digits, lab, prob = choose(digits, lab, prob)

        yhat = ['%d,%.2f' % (lab[i], prob[i]) for i in range(lab.size)]

        font = ImageFont.truetype(FONT_FILE, FONT_SIZE)
        detected = annotate_detection(im.copy(), digits)

        basename = os.path.basename(filename)
        resultname = RESULT_FILES + '/' + basename

        print 'OpenCV results'
        recognized = annotate_recognition(detected, digits, yhat, font)
        recognized.show()
        recognized.save(resultname.replace('.jpg', '-cv.jpg'))


if __name__ == '__main__':
    main()
