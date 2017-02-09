# Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

This project implements a software pipeline to identify vehicles in a video.

## Feature Extraction and Classifer
This section discusses the features and classifier used in the vehicle/not vehicle classifier I used HOG/color/color histogram features, and the model was trained using SVMs.

### Histogram of Oriented Gradients HOG/Color/Spatial Bin Feature Extraction
In the notebook cells numbered: 24, 5, 6

First I experimented with HOG features with the default parameters suggested on the project spec. They looked really good, tuning down the number of pixels per cell from 8 to 6(and alternatively increasing cells per block) added a lot of noise to the feature extraction process and there was not a lot of activity of gradients along specific directions, i.e. I was losing information on directionlity of the edge. It would result in very bright points. I used the default parameters orientations=9, pixels_per_cell=8, cells_per_block=2. Setting number of pixels per cell too large seemed to effect signals from near vertical edges.

I chose the RGB space rather than HLS for HOG extraction, the L space returns really good features, but the H and S spaces were very noisy. HOG features are extracted on the R,G,B channels in the image.

The input window size over which features are extracted is a (64x64) section of the image, this window is downsampled to (32x32) and the raw RGB values from the image are used as color features.

And lastly histograms from the RGB channels are used as additional features in classification.

```
A feature vector(8480 size) = | color features (32x32) | hog features (1764x3) | color histograms (32x3) |
```

![alt tag](https://raw.githubusercontent.com/nalapati/sdc-vehicle-detection/master/car_non_car.jpg)
![alt tag](https://raw.githubusercontent.com/nalapati/sdc-vehicle-detection/master/hog_features_l.jpg)
![alt tag](https://raw.githubusercontent.com/nalapati/sdc-vehicle-detection/master/hog_features_r.jpg)
![alt tag](https://raw.githubusercontent.com/nalapati/sdc-vehicle-detection/master/hog_features_g.jpg)
![alt tag](https://raw.githubusercontent.com/nalapati/sdc-vehicle-detection/master/hog_features_b.jpg)
![alt tag](https://raw.githubusercontent.com/nalapati/sdc-vehicle-detection/master/n_hog_features_r.jpg)
![alt tag](https://raw.githubusercontent.com/nalapati/sdc-vehicle-detection/master/n_hog_features_g.jpg)
![alt tag](https://raw.githubusercontent.com/nalapati/sdc-vehicle-detection/master/n_hog_features_b.jpg)


### Training a classifier using HOG features
In the notebook cells numbered: 7

The features were then extracted over the vehicle/non-vehicles KITTI dataset. The features were shuffled and split 80/20 train and test. The dataset was augmented with non-car images from trees, roadside trees have vertical edges which worked well to reduce the false positive rate. An SVM linear classifier was used to fit the training data and was then evaluated on the test data. The resulting classifier has an accuracy of 99.6%. 
``NOTE: Features were normalized using a StandardScaler for both train, test and subsequently in the video pipeline.``

## Sliding Window Search
This section deals with the components to search an image for sections that contain cars.

### Scales and Overlap
In the notebook cells numbered: 22, 10, 11

First windows were drawn on an image manually to get a sense of size of the window wrt the entire image and cars at different points in the image. Next, intuitively made sense to use smaller windows close to the center of the image and larger windows further away from the center. The top half of the image is ignored and so are vehicles very far away(really small windows). The (1280, 720) sized images were reduced to (640, 360). Next, I decided overlap based on experimentation, too low overlap leads to not detecting cars and too high overlap increases computational overhead (In hindsight it might be better to simply translate test images and augment the dataset rather than control window overlap too much). Once the windows and search configuration was determined, search was conducted and features were extracted from bounding boxes and classified as containing cars and the bounding boxes coordinates were recorded.

```
windows = (
        slide_window(car, x_start_stop=[75, 550], y_start_stop=[190, 250], xy_window=(32, 32), xy_overlap=(0.5, 0.5))
        + slide_window(car, x_start_stop=[25, 625], y_start_stop=[200, 275], xy_window=(48, 48), xy_overlap=(0.65, 0.5))
        + slide_window(car, x_start_stop=[25, 625], y_start_stop=[190, 275], xy_window=(64, 64), xy_overlap=(0.7, 0.65))
        + slide_window(car, x_start_stop=[25, 625], y_start_stop=[190, 300], xy_window=(96, 96), xy_overlap=(0.7, 0.7))
    )
```

![alt tag](https://raw.githubusercontent.com/nalapati/sdc-vehicle-detection/master/test_bb.jpg)
![alt tag](https://raw.githubusercontent.com/nalapati/sdc-vehicle-detection/master/test_32_32.jpg)
![alt tag](https://raw.githubusercontent.com/nalapati/sdc-vehicle-detection/master/test_48_48.jpg)
![alt tag](https://raw.githubusercontent.com/nalapati/sdc-vehicle-detection/master/test_64_64.jpg)
![alt tag](https://raw.githubusercontent.com/nalapati/sdc-vehicle-detection/master/test_96_96.jpg)

### Sliding Window Search in Action
In the notebook cells numbered: 12, 13, 14, 15

HOG features alone had an accuracy of 98.3% augmented with color and histogram features bumped the accuracy to 99.6%. In addition to this, heatmaps were generated from the input image. Essentially increment a pixel in the red channel of a reference matrix if the pixel is within a bounding box that contains a car. Heatmaps are generated over bounding boxes over multiple frames of sliding window search. Thresholding is applied on pixel intensities, such that pixel values below the thresholds are zeroed out. The final bounding boxes are determined by labelling connected components in the reference matrix. The bounding boxes surrounding the connected components are then overlayed on original image. The test video at the end of this section best illustrates the search in action. 

Sliding Window samples:

![alt tag](https://raw.githubusercontent.com/nalapati/sdc-vehicle-detection/master/test_windows_with_car_result.jpg)
![alt tag](https://raw.githubusercontent.com/nalapati/sdc-vehicle-detection/master/heatmap_gray_thresholded.jpg)
![alt tag](https://raw.githubusercontent.com/nalapati/sdc-vehicle-detection/master/car_labeled.jpg)

Click on the image for the Youtube Video:

[![Poject Video Output](https://img.youtube.com/vi/mEcZCQf_txk/0.jpg)](https://www.youtube.com/watch?v=mEcZCQf_txk)

## Video Pipeline Implementation
In the notebook in cells numbered: 16, 17, 18, 19, 21

### Implementation Sequence

#### Iteration 1
Before implementing all the features the order in which this project was really implemented:
* Car/Non Car sample plots.
* Simple HOG feature extraction.
* Use the KITTI dataset to build a simple SVM model.
* Test model on sample images.
* Run a simple sliding window in a specific section of the image to retrieve bounding boxes.
* Detect bounding boxes in a video in a pipeline with a really low threshold (i.e. allow false positives)

#### Iteration 2
* Cracking down on accuracy, the first model had a test accuracy of 98.3%, downsampled the image and used the raw pixel values from the RGB image + color histograms in a window to augment HOG features leading to 99.6% accuracy.
* Tested the new SVM model on new images.

#### Iteration 3
* Explored different scale and sizes of windows and locations in the image and overlap in these windows.
* Tested the pipeline end to end allowing for false positives.

#### Iteration 4
* Heatmaps using consecutive frames and removing areas with low ``heat`` values. This cuts down false positives after a bit of tuning.
* Tested the video end to end to get reasonable results.

#### Iteration 5
* The pipeline was really slow, cut down evaluation to running the algo on every alternate frame. The code processes frames at 2.5fps, can be easily improved to a lot higher rate.

#### Iteration 6
* Focus on trees, the vertical edges and side backgrounds caused a lot of false positives causing the detector to think, trees were cars. I collected data on trees and augmented the KITTI dataset to drastically reduce false positive due to trees. 

### Method for handling false positives
The final pipeline evaluates the search on every alternate frame in the video, and applies a threshold of 8 on the heatmap generated from detections on 10 consecutive frames. Essentially a car needs to be found 8 out of 10 frames around the same area for it to be considered a detection. (The camera runs at ~20fps, so you have 20frames in a single second of very similar data.). This worked well in the project video output.

### Project Video Output
Click on the image for the Youtube Video:

[![Poject Video Output](https://img.youtube.com/vi/w8qakmP1gSM/0.jpg)](https://www.youtube.com/watch?v=w8qakmP1gSM)

## Discussion

### Problems
* Trees recognized as cars, addressed by adding more data. Needs semantics (multiscale features like in CNNs)
* Detection of vehicles at night or in rain, can be addressed with additional data, but to a limit. Needs semantics (multiscale features like in CNNs)
* Teaching the classifier to be car position/orientation invariant vs Tuning sliding overlap.
* Color features failing on some cars like grey colored/blue colored/green colored cars.
* Overfitting or class bias solved by adding even samples from both classes.
* False positive rate controlled by using heatmaps and high thresholds to consider presence of a car.
* Computationally expensive to run an exhaustive vehicle detection always, potentially consider a stateful detection, essentially detect cars every 10 frames but for a few more frames only search around high confidence windows, this could bring detection high up to 10+fps on a reasonably good laptop.

### Improvements
* Deep learning for better features for vehicle detection, we don't necessarily have to use a deep network, but a couple of convolutional layers could be used as encoders for the image to be used in detection. (Potentially GANs for robust car detection).
* Sequence models could work better on the time series data in a window, we could potentially do away with overlapping windows and have non overlapping windows activating if a car passes through it, using timeseries data in that window.
* A LOT more data, using mechanical turk to label car video data or the udacity dataset to make the classifier more robust.
