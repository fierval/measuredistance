from dimensions import markdims
import cv2
import numpy as np
import time


fn = "kitchen.mp4"
w = 1400
h = 1600

if __name__ == '__main__':
  cap = cv2.VideoCapture(fn)

  for i in range(0, 120):
    cap.read()

  _, frame = cap.read()

  cap.release()
  pts = markdims.get_limit_points(frame)
  dst = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype="float32")
  M = cv2.getPerspectiveTransform(pts, dst)

  cap = cv2.VideoCapture(fn)
  res = True

  while res:
    res, frame = cap.read()
    if not res:
      break

    img = cv2.warpPerspective(frame, M, (w, h))

    cv2.imshow("warped", img)
    cv2.imshow("source", frame)

    if cv2.waitKey(1) == 27:
      break
    time.sleep(0.1)

  cv2.destroyAllWindows()