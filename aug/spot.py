import argparse
import struct
import numpy as np
import random as rd

class spot:
    kind = -1
    px = -1
    py = -1
    def __init__(self):
        self.kind = rd.randint(0, 3)
        self.px = rd.randint(0, 24)
        self.py = rd.randint(0, 24)

s1 = np.array([0, 0, 0, 0, 122, 255, 255, 0, 
           167, 255, 122, 0, 87, 122, 0, 0]).reshape(4, 4)
s2 = np.array([0, 0, 0, 0, 122, 255, 255, 0, 
           87, 209, 255, 0, 0, 87, 122, 0]).reshape(4, 4)
s3 = np.array([0, 122, 255, 255, 122, 255, 209, 255, 
           87, 186, 255, 122, 0, 87, 122, 0]).reshape(4, 4)
s4 = np.array([0, 0, 0, 0, 0, 122, 255, 255, 
           122, 255, 209, 255, 87, 122, 87, 122]).reshape(4, 4)
sp = np.array([s1, s2, s3, s4])

parser = argparse.ArgumentParser(description="transpose ubyte image")
parser.add_argument('--file', type=str,
                    default='emnist-digits-train-images-idx3-ubyte', 
                    help='ubyte file')
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
    n = rd.randint(0, 5)
    for r in range(n):
        spe = spot()
        sk = sp[spe.kind]
        for j in range(4):
            for k in range(4):
                tmp =  max(s[spe.px + j][spe.py + k], sk[j][k])
                s[spe.px + j][spe.py + k] = tmp
    s = np.transpose(s).reshape((-1, ))
    s = [chr(c) for c in s]
    s = "".join(s)
    img.seek(-784, 1)
    img.write(struct.pack('784s', s))
