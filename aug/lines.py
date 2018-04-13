from random import *
import argparse
from PIL import Image, ImageDraw
parser = argparse.ArgumentParser(description="add lines")
parser.add_argument('--file', type=str,
                    default='train-images-idx3-ubyte', help='ubyte file')
parser.add_argument('--no', type=int, default=0, help='image number in ubyte')
parser.add_argument('--out', type=str, default='lines.bmp', help='output  image')
args = parser.parse_args()

head = Image.open('5.bmp')



for i in range(randint(1,3)) :
    draw = ImageDraw.Draw(head)
    num = (randint (1,3))%3
    spaceing = uniform(1, 10)
    starting_point = uniform(0,14-spaceing)
    draw.line((0, starting_point+spaceing*num, 28+starting_point+spaceing*num, starting_point+spaceing*num), fill=256,width=randint(2,3))
    del draw

head.show()
