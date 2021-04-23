from dimensions import markdims
from detections.detector import ObjectDetector
from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
import cv2
import sys, os
import numpy as np
from args.args import cmd_args
import logging
from imutils.video import FPS

logging.basicConfig(format='%(asctime)s  %(levelname)-10s %(message)s', datefmt="%Y-%m-%d-%H-%M-%S",
                    level=logging.INFO)

fn = "kitchen.mp4"
w = 1400
h = 1600

if __name__ == '__main__':
  args = cmd_args()

  W = args.width * args.scale
  H = args.height * args.scale

  if W <= 0 or H <=0:
    raise ValueError("Width and height must both be positive")
  
  # matrix to use for all transforms
  M = markdims.get_perspective_matrix(W, H)

  abs_path = os.path.abspath(os.path.dirname(__file__))

  proto_path = os.path.join(abs_path, "mobilenet_ssd", "MobileNetSSD_deploy.prototxt")
  model_path = os.path.join(abs_path, "mobilenet_ssd", "MobileNetSSD_deploy.caffemodel")

  # load NN
  net = ObjectDetector(proto_path, model_path)

  # create video capture
  cap = cv2.VideoCapture(args.input)

  # instantiate our centroid tracker, then initialize a list to store
  # each of our dlib correlation trackers, followed by a dictionary to
  # map each unique object ID to a TrackableObject
  centroid_tracker = CentroidTracker(maxDisappeared=40, maxDistance=50)
  trackers = []
  trackableObjects = dict()

  totalFrames = 0

  # start the frames per second throughput estimator
  fps = FPS().start()

  res = True
  while res:
    res, frame = cap.read()
    
    if not res:
      break

    # time to re-detect!
    if totalFrames % args.skip_frames == 0:

      rects = net.step_detector(frame)
    else:
      rects = net.step_tracker(frame)

    if len(rects) == 0:
      continue

    # determine the IDs of objects being tracked
    objects = centroid_tracker.update(rects)

    for(objectID, centroid) in objects.items():
      
      # we may not be tracking this object yet
      tracked_object = trackableObjects.get(objectID, None)
      
      # we are interested in computations based on our perspective transform
      centroid_transformed = cv2.perspectiveTransform(np.array([centroid]))[0]

      if to is None:

        to = TrackableObject(objectID, centroid_transformed)
        trackableObjects[objectID] = to
      else:
        to.set_distance(centroid_transformed)
