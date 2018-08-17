
# Advanced Lane Finding Project

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistortion_example.png "Original and Undistorted"
[image2]: ./examples/undistort_ex.png "Original and Undistorted"
[image3]: ./examples/warp_ex.png "Undistorted and Warp"
[image4]: ./examples/binarize_ex.png "Undistorted and Binarized"
[image5]: ./examples/binary_warped_ex.png "Binarized and Binary Warped"
[image6]: ./examples/line_detection_ex.png "Line detection"
[image7]: ./examples/find_lane_ex.png "Finding lane area"
[image8]: ./data_flow_binarize.png "Binarize the image"
[video1]: ./output_videos/project_video.mp4 "Video with detected lane"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in [calib_cam.py](calib_cam.py) which I refered https://github.com/udacity/CarND-Advanced-Lane-Lines.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

Souce code for all of the following points are implemented in [lane_lines.ipynb](lane_lines.ipynb).

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

This step is implemented in `binarize_for_line()` function. I used a combination of color and gradient thresholds to generate a binary image. The following diagram illustrates the data flow.

![alt text][image8]

Here's an example of my output for this step. 

![alt text][image4]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The perspective transform is performed by `class LaneWarper` which has the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image. In the following example, the source rect is in blue line and the expected position of it in wapred image is in red line.

![alt text][image3]

I confirmed how the binarized image is warped as well.

![alt text][image5]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

This step is implemented in `LineDetector.process()`.
To collect the pixels for line, the below 2 approach is used based on the status.

1. Histogram and window search approach is used at the first frame after the status is reset, when previous result cannot be used.
1. Search in a margin around the previous line position approach is used when there is confident result of line detection previously. But in case the result of this approach is not good for successive frames, the history of result is reset and go back to approach 1 to rediscover the confident position.

![alt text][image6]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

This step is implemented in `LineDetector.measure_real()`. In this step, we need to convert the coordinates from pixel to real-world. `np.polyfit()` is performed with the line points scaled to real-world. Then the radius of curvature and vehicle position are calculated based on the x value at the y bottom of the image.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

Here is an example of my result on a test image:

![alt text][image7]

The other result for test images are located in the directory:[output_images](./output_images/)

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_videos/project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

* Performance is not enough for real time processing
  * FPS was low on my desktop PC which has only Intel GPU integrated in CPU.
  * At least, line detection of left and right can be parallelized.
  * If OpenCV/numpy is accelerated with GPU, the performance might be much better.

* Right line detection might be more unstable if the car speed is slower
  * As the right side is "dashed line", the detection got unstable depending on the line position.
  * More advanced algorithm that utilizes the point that the line of lane is dashed may improve this. For example, the point that the "dashed" piece of line is moving toward the driver in accordance with the speed of the car.
