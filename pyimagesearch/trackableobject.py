import numpy as np

class TrackableObject:
  '''
  Object assigned to each person to track their positions
  '''
  def __init__(self, objectID, centroid):
    # store the object ID, then initialize a list of centroids
    # using the current centroid
    self.objectID = objectID
    self.centroids = [centroid]

    # accumulate distance
    self.distance = 0
    
       
  def set_distance(self, pt):
    ''' 
    Get distance from point to latest centroid and set the point as the latest centroid
    '''
    (x1, y1) = pt
    (xc, yc) = self.centroids[-1]


    self.distance = np.linalg.norm(pt - self.centroids[-1])
    self.centroids.append(pt)
    
