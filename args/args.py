import argparse
from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np

def cmd_args():
  ap = argparse.ArgumentParser()

  ap.add_argument("-w", "--width", type=float, required=True, help="Width of the viewing area")
  ap.add_argument("-d", "--depth", type=float, required=True, help="Depth of the viewing area")

  ap.add_argument("-i", "--input", type=str,
    help="path to optional input video file")
  ap.add_argument("-c", "--confidence", type=float, default=0.5,
    help="minimum probability to filter weak detections")
  ap.add_argument("-s", "--skip-frames", type=int, default=30,
    help="# of skip frames between detections")

  return ap.parse_args()
