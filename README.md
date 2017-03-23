##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./Image1.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./SlidingWindow.png
[image4]: ./ExampleoutputLinearSVC.png
[image5]: ./Exampleoutput1.png
[image8]: ./Exampleoutput2.png
[image6]: ./IntegratedHeatmap.png
[image7]: ./LastImage.png


---
### Writeup / README



### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook `VehicleDetection_workspace.ipynb`.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Above image also shows how the exploration was done on different parameters. The `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`: Ipython widgets was used for ease of exploration.


#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters for `orientation`, `pixels_per_cell` and `hog_channel`. Channel 0 seems to have the maximum information. `'ALL'` was used eventually, so that there is no information loss. Lower `pixel_per_cell` seems to give better result than that of the higher ones. HOG visualization was used to see if the car features became prominent.
### Linear SVC

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using (see section Linear SVC in the ipython notebook). Around 3k samples randomly from the car and non-car images were used to train. A test accuracy of 0.99 was obtained. The model and other parameters used are stored as a pickle file in `svc_pickle.p`. In total, 11868 features were used including the color, spatial_feat and histogram features.

```
dct["svc"] = svc
dct["scaler"] = X_scaler
dct["orient"] = orient
dct["pix_per_cell"] = pix_per_cell
dct["cell_per_block"] = cell_per_block
dct["spatial_size"] = spatial_size
dct["hist_bins"] = hist_bins

pickle.dump(dct,open("svc_pickle.p","wb"))
```

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I used ipython widgets again to explore different scales and 1.5 seems to work good for the data. Below is the image of the widget with exploration capability. Also, explored 0.45 and 0.5 overlapping. 0.5 works good. More overlapping mean more calculation and hence slow processing.

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation


Here's a the video

[![Vehicle Detection](https://img.youtube.com/vi/0Fqs79vQjaM/0.jpg)](https://www.youtube.com/watch?v=0Fqs79vQjaM "Vehicle Detection")


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

False positives are removed using two tricks.
1. Looking at the lower half of the pixels in the image.
2. Heatmaps are colleted and keeping a running average of the heatmaps. Currently, slightly wobbly implementation is looking at 5 past heatmaps and averaging the values.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

![alt text][image8]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:

![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:

![alt text][image7]



---

###Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

1. Some of the parameters are hardcoded (like the area to look at), which sometimes becomes problematic as it might miss some vehicles that are far apart.
2. R-CNN based methods might be more robust. Recurrent networks can be used to pass on the information between consecutive frames.
3. Detection of the volume of the vehicle, not just 2D bounding box , but 3D.
4. Identifying the  car (make of the vehicle) and using that as input to consecutive frames.
5. Also, identifying the area of interest using Machine learning model before searching for vehicles could be helpful in reducing the processing time.

### References

1. https://www.youtube.com/watch?v=7S5qXET179I
2. Slack channel
