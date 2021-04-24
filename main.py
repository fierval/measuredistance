from dimensions import markdims
from detections.detector import ObjectDetector
from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
import cv2
import sys, os, json, time
import numpy as np
from args.args import cmd_args
import logging
from imutils.video import FPS

logging.basicConfig(format='%(asctime)s  %(levelname)-10s %(message)s', datefmt="%Y-%m-%d-%H-%M-%S",
                    level=logging.INFO)

def try_parse_int(s):
  '''
  Try converting to int, if can't - return the string
  '''
  try:
    return int(s)
  except ValueError:
    return s

if __name__ == '__main__':
  args = cmd_args()

  W = args.width * args.scale
  H = args.depth * args.scale

  if W <= 0 or H <=0:
    raise ValueError("Width and height must both be positive")

  source = try_parse_int(args.input)  

  # TODO: debug only limit points
  limit_pts = np.array([[ 767,  603],
                        [1913,  435],
                        [1907,  887],
                        [478,  987]])

 # matrix to use for all transforms
  M = markdims.get_perspective_matrix(source, W, H, limit_pts=limit_pts)

  abs_path = os.path.abspath(os.path.dirname(__file__))

  proto_path = os.path.join(abs_path, "mobilenet_ssd", "MobileNetSSD_deploy.prototxt")
  model_path = os.path.join(abs_path, "mobilenet_ssd", "MobileNetSSD_deploy.caffemodel")

  # load NN
  net = ObjectDetector(proto_path, model_path, args.confidence)

  # create video capture
  cap = cv2.VideoCapture(source)

  # instantiate our centroid tracker, then initialize a list to store
  # each of our dlib correlation trackers, followed by a dictionary to
  # map each unique object ID to a TrackableObject
  centroid_tracker = CentroidTracker(maxDisappeared=40, maxDistance=50)
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

    if len(rects) != 0:
      # determine the IDs of objects being tracked
      objects = centroid_tracker.update(rects)

      # process & draw
      for(objectID, centroid) in objects.items():
        
        # we may not be tracking this object yet
        tracked_object = trackableObjects.get(objectID, None)

        # we are interested in computations based on our perspective transform
        expanded_centroid = np.array(centroid)[None, None, :].astype("float32")
        centroid_transformed = cv2.perspectiveTransform(expanded_centroid, M)[0].squeeze()

        if tracked_object is None:

          tracked_object = TrackableObject(objectID, centroid_transformed)
          trackableObjects[objectID] = tracked_object
        else:
          tracked_object.set_distance(centroid_transformed)

        # draw both the ID of the object and the centroid of the
        # object on the output frame
        text = f"{objectID}: {tracked_object.distance / args.scale :.2f}"
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

    totalFrames += 1
    fps.update()
    cv2.imshow("Video", frame)

    if cv2.waitKey(1) == 27:
      break

    time.sleep(0.1)

  cv2.destroyAllWindows()
  
  fps.stop()
  logging.info("Elapsed time: {:.2f}".format(fps.elapsed()))
  logging.info("FPS: {:.2f}".format(fps.fps()))

  res = []
  for (objectID, person) in trackableObjects.items():
    res += [{"person": objectID, "distance_travelled_meters": round(person.distance / args.scale, 2)}]

  print(json.dumps(res))    

