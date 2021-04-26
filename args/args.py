import argparse
from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np

def cmd_args():
  ap = argparse.ArgumentParser()

  ap.add_argument("-w", "--width", type=float, required=True, help="Width of the viewing area, meters")
  ap.add_argument("-d", "--depth", type=float, required=True, help="Depth of the viewing area, meters")

  ap.add_argument("-i", "--input", type=str, default="0", help="path to optional input video file")
  ap.add_argument("-o", "--output", type=str, help="path to optional output file")
  ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
  ap.add_argument("-s", "--skip-frames", type=int, default=30,    help="# of skip frames between detections")

  ap.add_argument("-j", "--jitter", type=float, default=0.1, help="how much jitter do we tolerate when tracking? E.g. a move of 0.1 meters is not considered a move.")
  ap.add_argument("--scale", type=float, default=100, help="Scale the width and height to arrive at the pixel values of coordinates")
  
  ap.add_argument("--test", action="store_true", default=False, help="Test run with a known video source and parameters")
  return ap.parse_args()
