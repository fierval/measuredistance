from dimensions import markdims
import cv2

fn = "kitchen.mp4"
if __name__ == '__main__':
  cap = cv2.VideoCapture(fn)

  for i in range(0, 120):
    cap.read()

  _, frame = cap.read()

  pts = markdims.get_limit_points(frame)