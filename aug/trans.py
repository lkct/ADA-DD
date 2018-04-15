import argparse
import struct

import numpy as np

parser = argparse.ArgumentParser(description="transpose ubyte image")
parser.add_argument('--file', type=str,
                    default='emnist-digits-train-images-idx3-ubyte', help='ubyte file')
args = parser.parse_args()

img = open(args.file, 'rb+')

_, len, _, _ = struct.unpack('>IIII', img.read(16))
for i in range(len):
    if i % 1000 == 0:
        print i
    img.seek(16 + i * 784)
    s, = struct.unpack('784s', img.read(784))
    s = [ord(c) for c in s]
    s = np.array(s, dtype=np.uint8).reshape((28, 28))
    s = np.transpose(s).reshape((-1, ))
    s = [chr(c) for c in s]
    s = "".join(s)
    img.seek(-784, 1)
    img.write(struct.pack('784s', s))
