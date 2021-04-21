import cv2
import numpy as np
from imutils.perspective import order_points
import logging

logging.basicConfig(format='%(asctime)s  %(levelname)-10s %(message)s', datefmt="%Y-%m-%d-%H-%M-%S",
                    level=logging.INFO)

ix,iy = -1,-1
img = None

def draw_points(point_collection):
  '''
  Draw the points we picked
  '''
  # the very last one is the "freshest" clone of the 
  # global image
  if(len(point_collection) < 1):
    return

  im = point_collection[-1]["img"]
  for p in point_collection:
    cv2.circle(im, p["pt"], 5, (0, 255, 0), -1)

# mouse callback function
def collect_points(event, x, y, flags, limits):
    global ix,iy, img

    if event == cv2.EVENT_LBUTTONDOWN:
      ix,iy = x,y
      img_clone = img.copy()
      limits += [{"pt": (x, y), "img": img_clone}]
      draw_points(limits)

    # undo
    elif event == cv2.EVENT_RBUTTONDOWN:
      # drop the last point
      if len(limits) > 0:
        limits.pop()


def get_limit_points(frame):
  '''
  Mark the limits of the floor with a mouse and return them in order (top, left), (top, right), (bottom, right), (bottom, left)
  '''
  global img

  img = frame.copy()  
  limits = []
  cv2.namedWindow("image")
  cv2.setMouseCallback("image", collect_points, limits)

  while True:
    im = img if len(limits) < 1 else limits[-1]["img"]

    cv2.imshow('image', im)
    if cv2.waitKey(1) == 27:
      break

  cv2.destroyAllWindows()

  if len(limits) != 4:
    raise ValueError("Need exactly 4 points deliniating the kitchen floor")
  
  # order the limits
  real_limits = np.array([p["pt"] for p in limits])

  # imutils order_points orders them the way we wanted clockwise starting at top left
  real_limits = order_points(real_limits)
  
  logging.info(f"result {real_limits}")

  return real_limits
  

