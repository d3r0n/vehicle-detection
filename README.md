[//]: # (Image References)
[image0]: ./images/header.gif "Header"
[image1]: ./images/car.png
[image2]: ./images/notcar.png
[image3]: ./images/hog.png
[image4]: ./images/sliding_window.png
[image5]: ./images/heatmap.png
[image6]: ./images/video1.gif
[image7]: ./images/video2.gif
[image8]: ./images/video3.gif
[video1]: ./project_ok.mp4

# Vehicle Detection Project

![alt text][image0]

>__TLDR;__ here is a link to my [main code.](https://github.com/d3r0n/vehicle-detection/blob/master/main.py)

---

### My achievements :rocket: in this rocket:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images
* Apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Normalize above feature vector
* Split data into train and test and train a Linear SVM classifier
* Implement a sliding-window technique with trained classifier to search for vehicles in images.
* Create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles on the video.
* Estimate a bounding box for vehicles detected.

---

### Histogram of Oriented Gradients (HOG)

#### 1. Data Load

The code for this step is contained in the first 20 lines of main.py

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1] ![alt text][image2]

In dataset there are 8792 images of cars 8968 images of not cars.

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image3]

#### 2. Final choice of HOG parameters.

I tried various combinations of parameters for extracting features and settled with following parameters which worked best with `SVC`.

| HOG parameter | Value         |
|:-------------:|:-------------:|
| Color space   | YCrCb         |
| Color channels | ALL |
| Orientation angles | 9 |
| Cell | 8x8 px |
| Block | 2x2 cells |

#### 3. Feature vector

This is how feature vector looks like:

| Feature:  | HOG           | Spatial binning  | Histogram bins |
|:---------:|:-------------:|:----------------:|:--------------:|
| Position: | 0-5291 (5292) | 5292-8363 (3072) | 8364-8459 (96) |

* HOG features are `1764` long per channel
* Spatial binning was done with `32x32` target size (`1024`) on each color channel (`3x1024`)
* Color histogram created with `32` bins on each color channel (`3x32`)

__Feature vector length - `8460`__

#### 4. Classifier

First, data has been scaled with scikit-learn StanderScaler
Then it has been split in proportion 80-20 between Train and Test.
Thus, __linear SVM__ has been trained using 14208 samples reaching accuracy on test set of __99.46 %__

#### 5. Sliding Window Search

Code for sliding window can be found in line `139` of `main.py`

Naïve approach for searching cars is to slice image into windows and run classifier on them.
This wont work well since we can actually split car in half between two adjacent boxes. Thus I have used overlapping windows.

What about cars which are further away and those close by? One approach would be to resize windows to match windows used in classifier training. I finally removed this from my code as it was ridiculously slow.

Another optimization is to narrow down the search space to the area of image where the road is.
There are not cars in the sky - just yet :wink:

![alt text][image4]

---

### Video Implementation

Here's a [link to my video result](./project_ok.mp4)

#### 1. Performance

Sliding window approach is very slow. Computing HOG features for each window separately and eventual resizing takes lot of time considering that pipeline has to do it hundreds of times per frame.

Different approach is to compute HOG in per whole frame and select only those which are in search window. Same goes for scaling. Instead of scaling each window it is better to scale whole picture frame.

Code for video pipeline can be found in `CarDetector` class in line `373`.

#### 2. Multiple detections

In order to reduce false positives I have used heatmap. Each car detection will make region of the image 'warmer'. When multiple detections overlap over image region it will become 'hot' and thus more probably detect a car. I used `scipy.ndimage.measurements.label()` to identify those areas in the heatmap. I constructed bounding boxes to cover the area of each blob detected.  

![alt text][image5]

Another optimization in order to remove false positives was to combine multiple heatmaps from consecutive frames. This smoothing makes car frames smoother and less wobbly. I had only to adapt the threshold a bit.

---

### Screens:

__Incoming car detection__

![alt text][image6]

__Bounding box merge__

![alt text][image7]

__Bounding box split__

![alt text][image8]

---

### Discussion

##### Individual car tracking

Current biggest problem of the above implementation are overlapping cars. Please see screen of bounding box merge. Suddenly one of the cars disappears from the scene. To solve this problem one could use some tracking algorithms to preserve information where the car has been before and where should be in near features.

Another potentially dangerous thing might be when bounding box stretches among two cars. Please have a look at bounding box split screen. This can be dangerous because the bounding box grows as if neighbouring car ware turning left and showed its longer left profile.

##### Thresholds

Another problem one can tackle is reducing thresholds in order to quicker detect a car. For instance have a look at screen with incoming car. Bounding box could appear quicker. Ideally when front lamp appears. But then one would have to return to the problem of false positives.

##### Model

Finally, one can think about changing SVC to more sophisticated detection classifier like CNN.
