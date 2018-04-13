import argparse

parser = argparse.ArgumentParser(description="visualize ubyte image")
parser.add_argument('--file', type=str,
                    default='train-images.idx3-ubyte', help='ubyte file')
parser.add_argument('--no', type=int, default=0, help='image number in ubyte')
parser.add_argument('--out', type=str, default='vis.bmp', help='output image')
args = parser.parse_args()

head = open('5.bmp', 'rb')
img = open(args.file, 'rb')
out = open(args.out, 'wb')

out.write(head.read(0x436))
img.seek(16 + (args.no + 1) * 784 - 28)
for i in range(28):
    if i != 0:
        img.seek(-56, 1)
    out.write(img.read(28))
