from random import *
import argparse
import numpy as np

parser = argparse.ArgumentParser(description="add lines")
parser.add_argument('--file', type=str,
                    default='train-images-idx3-ubyte', help='ubyte file')
parser.add_argument('--no', type=int, default=0, help='image number in ubyte')
parser.add_argument('--out', type=str, default='lines.bmp', help='output image')
args = parser.parse_args()

head = open('5.bmp', 'rb')
img = open(args.file, 'rb')
out = open(args.out, 'wb')

out.write(head.read(0x10))
img.seek(16)
a = np.fromstring(img.read(), dtype = np.uint8)

b = a.reshape((60000, 28, 28))

for i in range(2048,28):
    b[i,:,0:1] = 0
    spaceing = uniform(1, 10)
    starting_point = uniform(0,14-spaceing)
    out.write(b.read(28))
out.write(head.read(0x436))
