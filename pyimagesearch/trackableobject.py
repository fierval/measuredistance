import numpy as np

class TrackableObject:
  '''
  Object assigned to each person to track their positions
  '''
  def __init__(self, objectID, centroid, scale, jitter):
    # store the object ID, then initialize a list of centroids
    # using the current centroid
    self.objectID = objectID
    self.centroids = [centroid]

    # accumulate distance
    self.distance = 0

    self.scale = scale
    self.jitter = jitter
    
       
  def set_distance(self, pt):
    ''' 
    Get distance from point to latest centroid and set the point as the latest centroid
    '''
    (x1, y1) = pt
    (xc, yc) = self.centroids[-1]

    dist = np.linalg.norm(np.array(pt) - np.array(self.centroids[-1])) / self.scale

    if dist > self.jitter:
      self.distance += dist
      self.centroids.append(pt)
    
