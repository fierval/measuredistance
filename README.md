# Measuring Distance Traveled with Monocular Camera

The camera is watching a space where people are filmed perfroming different tasks, or a file recorded offline from such camera. This is the problem input. Video frames from the input are sent to an AI pipeline. The pipeline tracks people's movements and measures total distance each one of them travels overtime.

## Assumptioins

1. The project is a POC (proof of concept), and **not a production system**. This is a crucial assumption as it allows us to state certain parameters voluntaristically instead of spending time gathering requirements. It is motivated by the complexity of the problem statement vs time available to solve it.
1. The time during which employees are tracked and the result is produced, should be short as errors accumulate and we are not dealing with any possible problems here except "jitter" (see below)
1. The dimensions of the space are known in advance (or can be measured)

## Running the App

### Installation

Windows, Linux, or MacOS.

1. Install [Anaconda](https://www.anaconda.com/products/individual).
1. Change into the root of the solution using command/shell prompt

```sh
conda create -n ck python=3.6
conda activate ck
pip install -r requirements.txt
```
## Design & Implementation

### The Algorithm

1. Every time period (of 1 sec or so):
1. Detect people using object detector (MobileNet SSD in our case)
1. While time period not expried, track employees using correlation tracker (dlib).
1. On every tracking or detection, compute:  
  a. Centroid from the rectangle of the tracking object  
  b. Using centroid tracking, find the object in the set of known objects  
  c. Project the centroid on the plane parallel the "floor" and scaled based on the space dimensions  
  d. Compute distance travelled between the current and the last known centroid location. Account for "jitter". Accumulate the distance
1. Stop when user manually stops the app or when the video feed has ended

### Launching Test Feed

The following will launch the test feed:

```sh
python main.py --width 3.72 --depth 2.16 --jitter 0.1 --confidence 0.65 --input "videos/den3.mp4" --test
```

See detailed description of the arguments in [args\args.py](args/args.py)

This should produce the output:

```json
[{"person": 1, "distance_travelled_meters": 4.38}]
```
Click to launch the video:  

[![Test Distance Measure Video](https://img.youtube.com/vi/6fRGGj58IQo/0.jpg)](https://www.youtube.com/watch?v=6fRGGj58IQo)

### Launching a Real Feed

Before we run the algorithm we first need to establish the projection plane. Once the program is launched, the user is presented with a still image of the space where the plane that corresponds to the known dimensions needs to be marked by 4 points.

`LEFT` mouse button selects a point, `RIGHT` button cancels the current selection. Hit `<ENTER>` when 4 points are selected:


The video will play together with the projection, just for visual validation, hit `<ESC>`, to start the program execution.

### Implementation Notes

Initially, we run the object detector to detect objects we need to track (with parameterized confidence) and then the tracker is invoked every frame to track individual object movements. We correlate tracked object centroids from frame to frame and compute distances based on the centroid projections. We only accumulate distances and centroids if the current distance value is greater than the "jitter" parameter. This parameter compensates for detected centroid differences that may be due to tracking/detection artifacts while the object is really stationary. 0.1 meters appears to be sufficient. 

Detection step is repeated every so often (30 frames by default). We thus hope to increase performance by running correlation tracker  (using `dlib`) rather than a forward pass through a MobileNet SSD neural net.

The centroid tracker will drop objects it has not seen for a certain amount of time or that have moved too far from the previous centroid.

### Sources

Most of the code for centroid-tracking and object detection is re-used from [Pyimagesearch](https://www.pyimagesearch.com/2018/08/13/opencv-people-counter/)

### Proposed Metric

Given a video of certain length _t_, the MSE (Mean Squared Error) will be computed over a known travel distance for each person in the video. Given a set of such videos of fixed length, mean and standard deviation of these MSE values could be adopted as a possible metric.

### Testing

The above video is a short clip of me walking the known distance in my living room. Works very well for this short and very deterministic scenario.

### Performance

Current performance on the full HD stream is ~ 8 fps on a Ryzen 7/64 Gb/SSD desktop. This can easily be significantly improved on by:

- Using GPU for the object detector. Since I used the "stock" OpenCV instead of building my own for my own GPU, I could not take advantage of the GPU.
- Tracking every single video frame is unnecessary and hurts perfromance. Reducing the load, tracking based on the camera FPS, will improve performance for high-end cameras that stream at 30 or 60 fps, when only every second or every third frame is processed instead of every frame.