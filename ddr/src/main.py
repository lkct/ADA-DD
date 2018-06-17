#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from glob import glob

import cv2
import numpy as np
from PIL import Image, ImageFont

import col
import pred
from detect import annotate_detection, crop_detection, detect
from load_labels import get_data
from recognize import annotate_recognition, cvtrain, preprocess, sktrain

SAMPLE_SIZE = (28, 28)
SZ = 28
script_dir = os.path.dirname(__file__)
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
    s = min(d1[2]*d1[3], d2[2]*d2[3])
    x1 = max(x11, x21)
    y1 = max(y11, y21)
    x2 = min(x12, x22)
    y2 = min(y12, y22)
    dx = max(0, x2-x1)
    dy = max(0, y2-y1)
    return dx*dy / float(s)


def choose(digits, lab, prob):
    chosen = []
    n = lab.size
    for i in range(n):
        if prob[i] < 0.4:
            continue
        flag = False
        for j in range(i - 1):
            if overlap(digits[i], digits[j]) > 0.9:  # ==1
                flag = True
                break
        if not flag:
            chosen.append(i)
    chosen = np.uint8(chosen)
    digits = digits[chosen]
    lab = lab[chosen]
    prob = prob[chosen]

    chosen = []
    n = lab.size
    for i in range(n):
        flag = False
        for j in range(n):
            if j == i:
                continue
            if overlap(digits[i], digits[j]) >= 0.4 and prob[i] < prob[j]:
                flag = True
                break
        if not flag:
            chosen.append(i)
    chosen = np.uint8(chosen)
    digits = digits[chosen]
    lab = lab[chosen]
    prob = prob[chosen]

    return digits, lab, prob


def recog(img, oriimg,  resname):
    script_dir = os.path.dirname(__file__)
    CASCADE_FILE = os.path.join(script_dir, '../classifier/cascade.xml')

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    im = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    im = im.convert('L')
    digits = detect(gray, CASCADE_FILE)
    digits = np.array(digits)
    if digits.shape[0] == 0:
        return digits, np.array([]), np.array([])
    results = crop_detection(im.copy(), digits)
    test = np.float32([np.float32(i.resize(SAMPLE_SIZE)) for i in results])
    test = np.tile(test.reshape((-1, 1, 28, 28)), (1, 3, 1, 1))

    # yhat:list of str=label+prob
    lab, prob = pred.main(test)
    digits, lab, prob = choose(digits, lab, prob)
    yhat = ['%d,%.2f' % (lab[i], prob[i]) for i in range(lab.size)]

    digits[:, 1] += 525 / 2
    oriimg = cv2.resize(oriimg, None, fx=0.5, fy=0.5)

    im = Image.fromarray(cv2.cvtColor(oriimg, cv2.COLOR_BGR2RGB))
    im = im.convert('L')
    font = ImageFont.truetype(FONT_FILE, FONT_SIZE)
    detected = annotate_detection(im.copy(), digits)
    recognized = annotate_recognition(detected, digits, yhat, font)
    recognized.show()
    recognized.save(resname)

    return digits, lab, prob


def smain(img, resname):
    pimg = col.proc(img)
    return recog(pimg, img, resname)


def main():
    images, labels, num, rows, cols = get_data(LABEL_FILE,
                                               IMAGE_FILE)

    filenames = glob(TEST_FILES + "*.jpg")
    for filename in filenames:
        print 'Processing', filename
        img = cv2.imread(filename)

        print 'results'
        basename = os.path.basename(filename)
        resultname = RESULT_FILES + '/' + basename
        recog(img, resultname.replace('.jpg', '-result.jpg'))


if __name__ == '__main__':
    main()
