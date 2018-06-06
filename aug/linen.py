<<<<<<< HEAD
import argparse
import struct
import numpy as np
import random as rd

parser = argparse.ArgumentParser(description="transpose ubyte image")
parser.add_argument('--file', type=str,
                    default='emnist-digits-train-images-idx3-ubyte', help='ubyte file')
args = parser.parse_args()

img = open(args.file, 'rb+')

img.seek(16)
s = np.fromstring(img.read(), dtype=np.uint8).reshape((-1, 28, 28))
for i in range(s.shape[0]):
    n = rd.randint(0, 2)
    if n == 1:
        l = rd.randint(0, 26)
        for k in range(28):
            s[i][l][k] = s[i][l + 1][k] = 255
    elif n == 2:
        ls = rd.randint(0, 12)
        for k in range(28):
            s[i][ls][k] = s[i][ls + 1][k] = s[i][ls + 14][k] = s[i][ls + 15][k] = 255
img.seek(16)
img.write(np.transpose(s, (0, 2, 1)).reshape((-1, )).tostring())
=======
import argparse
import struct
import numpy as np
import random as rd

parser = argparse.ArgumentParser(description="transpose ubyte image")
parser.add_argument('--file', type=str,
                    default='emnist-digits-train-images-idx3-ubyte', help='ubyte file')
args = parser.parse_args()

img = open(args.file, 'rb+')

img.seek(16)
s = np.fromstring(img.read(), dtype=np.uint8).reshape((-1, 28, 28))
for i in range(s.shape[0]):
    n = rd.randint(0, 2)
    if n == 1:
        l = rd.randint(0, 26)
        for k in range(28):
            s[i][l][k] = s[i][l + 1][k] = 255
    elif n == 2:
        ls = rd.randint(0, 12)
        for k in range(28):
            s[i][ls][k] = s[i][ls + 1][k] = s[i][ls + 14][k] = s[i][ls + 15][k] = 255
img.seek(16)
img.write(np.transpose(s, (0, 2, 1)).reshape((-1, )).tostring())
>>>>>>> upstream/master
