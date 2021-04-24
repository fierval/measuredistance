import os
import cv2
import dlib
import numpy as np

class ObjectDetector:
  '''
  Wraps our detection & tracking operations
  '''

  def __init__(self, proto_path, model_path, confidence = 0.5):

    self.net = cv2.dnn.readNetFromCaffe(proto_path, model_path)
    
    self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    
    self.orig_w = None
    self.orig_h = None
    self.width_height_mult = None
    self.trackers = []
    self.confidence = confidence

  def set_width_height(self, frame):
    if self.orig_h is None:
      self.orig_h, self.orig_w = frame.shape[:2]
      self.width_height_mult = np.array([self.orig_w, self.orig_h, self.orig_w, self.orig_h])

  def step_detector(self, frame):
    '''
    Detect people and initialize their correlation trackers
    '''

    self.set_width_height(frame)
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (self.orig_w, self.orig_h), 127.5)
    self.net.setInput(blob)
    detections = self.net.forward()

    # the frame from BGR to RGB for dlib
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # we are re-detecting, so abandon the old trackers
    self.trackers = []
    boxes = []

    # loop over the detections
    for i in np.arange(0, detections.shape[2]):
      # extract the confidence (i.e., probability) associated
      # with the prediction
      confidence = detections[0, 0, i, 2]
      # filter out weak detections by requiring a minimum
      # confidence
      if confidence > self.confidence:
        # extract the index of the class label from the
        # detections list
        idx = int(detections[0, 0, i, 1])

        # This is the index of class "person"
        if idx != 15:
          continue

        # compute the (x, y)-coordinates of the bounding box
        # for the object
        box = detections[0, 0, i, 3:7] * self.width_height_mult
        [startX, startY, endX, endY] = [int(v) for v in box]

        # construct a dlib rectangle object from the bounding
        # box coordinates and then start the dlib correlation
        # tracker
        tracker = dlib.correlation_tracker()
        rect = dlib.rectangle(startX, startY, endX, endY)
        tracker.start_track(rgb, rect)

        # add the tracker to our list of trackers so we can
        # utilize it during skip frames
        self.trackers.append(tracker)
        boxes.append([startX, startY, endX, endY])

    return boxes

  def step_tracker(self, frame):
    '''
    Just use the tracker to figure out positions of previously detected objects
    '''
    self.set_width_height(frame)    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes = []
    for tracker in self.trackers:

      tracker.update(rgb)
      pos = tracker.get_position()

      # unpack the position object
      startX = int(pos.left())
      startY = int(pos.top())
      endX = int(pos.right())
      endY = int(pos.bottom())

      boxes.append([startX, startY, endX, endY])

    return boxes
