import argparse
import struct

import numpy as np

parser = argparse.ArgumentParser(description="transpose ubyte image")
parser.add_argument('--file', type=str,
                    default='emnist-digits-train-images-idx3-ubyte', help='ubyte file')
args = parser.parse_args()

img = open(args.file, 'rb+')

img.seek(16)
s = np.fromstring(img.read(), dtype=np.uint8).reshape((-1, 28, 28))
img.seek(16)
img.write(np.transpose(s, (0, 2, 1)).reshape((-1, )).tostring())
