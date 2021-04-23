from dimensions import markdims
import cv2
import numpy as np
from args.args import cmd_args

fn = "kitchen.mp4"
w = 1400
h = 1600

if __name__ == '__main__':
  args = cmd_args()

  perspective_transform = markdims.get_perspective_matrix(args.width, args.depth)

  

