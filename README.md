[AKAZE_00]: ./output/AKAZE_00.png "AKAZE_00"
[BRISK_00]: ./output/BRISK_00.png "BRISK_00"
[FAST_00]: ./output/FAST_00.png "FAST_00"
[HARRIS_00]: ./output/HARRIS_00.png "HARRIS_00"
[ORB_00]: ./output/ORB_00.png "ORB_00"
[SHITOMASI_00]: ./output/SHITOMASI_00.png "SHITOMASI_00"
[SIFT_00]: ./output/SIFT_00.png "SIFT_00"

# SFND 2D Feature Tracking

<img src="images/keypoints.png" width="820" height="248" />

The idea of the camera course is to build a collision detection system - that's the overall goal for the Final Project. As a preparation for this, you will now build the feature tracking part and test various detector / descriptor combinations to see which ones perform best. This mid-term project consists of four parts:

* First, you will focus on loading images, setting up data structures and putting everything into a ring buffer to optimize memory load. 
* Then, you will integrate several keypoint detectors such as HARRIS, FAST, BRISK and SIFT and compare them with regard to number of keypoints and speed. 
* In the next part, you will then focus on descriptor extraction and matching using brute force and also the FLANN approach we discussed in the previous lesson. 
* In the last part, once the code framework is complete, you will test the various algorithms in different combinations and compare them with regard to some performance measures. 

See the classroom instruction and code comments for more details on each of these parts. Once you are finished with this project, the keypoint matching part will be set up and you can proceed to the next lesson, where the focus is on integrating Lidar points and on object detection using deep-learning. 

## Dependencies for Running Locally
* cmake >= 2.8
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1 (Linux, Mac), 3.81 (Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* OpenCV >= 4.5
  * This must be compiled from source using the `-D OPENCV_ENABLE_NONFREE=ON` cmake flag for testing the SIFT and SURF detectors.
  * The OpenCV 4.5.0 source code can be found [here](https://github.com/opencv/opencv/tree/4.5.0)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory in the top level directory: `mkdir build || cd build`
3. Compile: `cmake .. || make`
4. Run it: `./2D_feature_tracking`.
---

## Descriptions

### MP.1 Data Buffer Optimization
*Implement a vector for dataBuffer objects whose size does not exceed a limit (e.g. 2 elements). This can be achieved by pushing in new elements on one end and removing elements on the other end.*

I use `std::list` to achieve this data buffer.
```c++
std::list<DataFrame> dataBuffer;
if (dataBuffer.size() == dataBufferSize) {
  dataBuffer.pop_front();
}
```

### MP.2 Keypoint Detection
*Implement detectors HARRIS, FAST, BRISK, ORB, AKAZE, and SIFT and make them selectable by setting a string accordingly.*

For **HARRIS detector**, I modify `void detKeypointsShiTomasi()` to add one more boolean parameter to switch between `HARRIS` and `ShiTomasi`.
```c++
void detKeypointsShiTomasi(std::vector<cv::KeyPoint> |keypoints, cv::Mat |img,
                           bool useHarris = false, bool bVis = false) {
  // ...
  // Using useHarris to choose HARRIS or ShiTomasi descriptor.
  cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance,
                          cv::Mat(), blockSize, useHarris, k);
  // ...
}
```

For **FAST**, **BRISK**, **ORB**, **AKAZE**, and **SIFT**, I implement in the function `void detKeypointsModern()`.
```c++
void detKeypointsModern(std::vector<cv::KeyPoint> |keypoints, cv::Mat |img,
                        std::string detectorType, bool bVis) {
  cv::Ptr<cv::FeatureDetector> detector;
  if (detectorType.compare("FAST") == 0) {
    // ...
    detector = cv::FastFeatureDetector::create(threshold, bNMS, type);
  } else if (detectorType.compare("BRISK") == 0) {
    // ...
    detector = cv::BRISK::create(threshold, octaves, patternScale);
  } else if (detectorType.compare("ORB") == 0) {
    // ...
    detector =
        cv::ORB::create(nfeatures, scaleFactor, nlevels, edgeThreshold,
                        firstLevel, WTA_K, scoreType, patchSize, fastThreshold);
  } else if (detectorType.compare("AKAZE") == 0) {
    // ...
    detector =
        cv::AKAZE::create(descriptor_type, descriptor_size, descriptor_channels,
                          threshold, nOctaves, nOctaveLayers, diffusivity);
  } else if (detectorType.compare("SIFT") == 0) {
    // ...
    detector = cv::SIFT::create(nfeatures, nOctaveLayers, contrastThreshold,
                                edgeThreshold, sigma);
  }
  // ...
}
```

### MP.3 Keypoint Removal
*Remove all keypoints outside of a pre-defined rectangle and only use the keypoints within the rectangle for further processing.*

Below is the way I use to remove the keypoints out of the rectangle box.
```c++
bool bFocusOnVehicle = true;
cv::Rect vehicleRect(535, 180, 180, 150);
if (bFocusOnVehicle) {
  vector<cv::KeyPoint> precedingKeypoints;
  for (int i = 0; i < keypoints.size(); i++) {
    if (vehicleRect.contains(keypoints[i].pt)) {
      precedingKeypoints.push_back(keypoints[i]);
    }
  }
  keypoints.swap(precedingKeypoints);
}
```

### MP.4 Keypoint Descriptors
*Implement descriptors BRIEF, ORB, FREAK, AKAZE and SIFT and make them selectable by setting a string accordingly.*

I implement **BRIEF**, **ORB**, **FREAK**, **AKAZE** and **SIFT** descriptor in the function `void descKeypoints`, using the parameter `descriptorMethod` to choose from one of them.
```c++
void descKeypoints(vector<cv::KeyPoint> |keypoints, cv::Mat |img,
                   cv::Mat |descriptors, string descriptorMethod) {
  // select appropriate descriptor
  cv::Ptr<cv::DescriptorExtractor> extractor;
  if (descriptorMethod.compare("BRISK") == 0) {
    // ...
    extractor = cv::BRISK::create(threshold, octaves, patternScale);
  } else if (descriptorMethod.compare("BRIEF") == 0) {
    extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
  } else if (descriptorMethod.compare("ORB") == 0) {
    // ...
    extractor =
        cv::ORB::create(nfeatures, scaleFactor, nlevels, edgeThreshold,
                        firstLevel, WTA_K, scoreType, patchSize, fastThreshold);
  } else if (descriptorMethod.compare("FREAK") == 0) {
    // ...
    extractor = cv::xfeatures2d::FREAK::create(
        orientationNormalized, scaleNormalized, patternScale, nOctaves);
  } else if (descriptorMethod.compare("AKAZE") == 0) {
    // ...
    extractor =
        cv::AKAZE::create(descriptor_type, descriptor_size, descriptor_channels,
                          threshold, nOctaves, nOctaveLayers, diffusivity);
  } else if (descriptorMethod.compare("SIFT") == 0) {
    // ...
    extractor = cv::SIFT::create(nfeatures, nOctaveLayers, contrastThreshold,
                                 edgeThreshold, sigma);
  }
  // ...
}
```

### MP.5 Descriptor Matching and MP.6 Descriptor Distance Ratio
*Implement FLANN matching as well as k-nearest neighbor selection. Both methods must be selectable using the respective strings in the main function.*

*Use the K-Nearest-Neighbor matching to implement the descriptor distance ratio test, which looks at the ratio of best vs. second-best match to decide whether to keep an associated pair of keypoints.*

I implement **FLANN matching** and **k-nearest neighbor selection** in the function `void matchDescriptors`.
```c++
void matchDescriptors(std::vector<cv::KeyPoint> |kPtsSource,
                      std::vector<cv::KeyPoint> |kPtsRef, cv::Mat |descSource,
                      cv::Mat |descRef, std::vector<cv::DMatch> |matches,
                      std::string descriptorType, std::string matcherType,
                      std::string selectorType) {
  // configure matcher
  bool crossCheck = false;
  cv::Ptr<cv::DescriptorMatcher> matcher;

  if (matcherType.compare("MAT_BF") == 0) {
    int normType = descriptorType.compare("DES_BINARY") == 0 ? cv::NORM_HAMMING
                                                             : cv::NORM_L2;
    matcher = cv::BFMatcher::create(normType, crossCheck);
    cout << "BF matching cross-check=" << crossCheck;
  } else if (matcherType.compare("MAT_FLANN") == 0) {
    if (descSource.type() != CV_32F) {
      // OpenCV bug workaround : convert binary descriptors to floating point
      // due to a bug in current OpenCV implementation
      descSource.convertTo(descSource, CV_32F);
      descRef.convertTo(descRef, CV_32F);
    }

    matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    cout << "FLANN matching";
  }

  // perform matching task
  if (selectorType.compare("SEL_NN") == 0) {  // nearest neighbor (best match)
    double t = (double)cv::getTickCount();
    // Finds the best match for each descriptor in desc1
    matcher->match(descSource, descRef, matches);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << " (NN) with n=" << matches.size() << " matches in "
         << 1000 * t / 1.0 << " ms" << endl;
  } else if (selectorType.compare("SEL_KNN") == 0) {
    // k nearest neighbors (k=2)
    vector<vector<cv::DMatch>> knn_matches;
    double t = (double)cv::getTickCount();
    // finds the 2 best matches
    matcher->knnMatch(descSource, descRef, knn_matches, 2);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << " (KNN) with n=" << knn_matches.size() << " matches in "
         << 1000 * t / 1.0 << " ms" << endl;

    // filter matches using descriptor distance ratio test
    constexpr double minDescDistRatio = 0.8;
    for (auto it = knn_matches.begin(); it != knn_matches.end(); ++it) {
      if ((*it)[0].distance < minDescDistRatio * (*it)[1].distance) {
        matches.push_back((*it)[0]);
      }
    }
    cout << "# keypoints removed = " << knn_matches.size() - matches.size()
         << endl;
  }
}
```

### MP.7 Performance Evaluation 1
*Count the number of keypoints on the preceding vehicle for all 10 images and take note of the distribution of their neighborhood size. Do this for all the detectors you have implemented.*

**The number of keypoints**
| Descriptor | Img.0 | Img.1 | Img.2 | Img.3 | Img.4 | Img.5 | Img.6 | Img.7 | Img.8 | Img.9 |
| ---------- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| Harris     | 50    | 54    | 53    | 55    | 56    | 58    | 57    | 61    | 59    | 57    |
| Shi-Tomasi | 125   | 118   | 123   | 120   | 120   | 113   | 114   | 123   | 111   | 112   |
| FAST       | 149   | 152   | 150   | 155   | 149   | 149   | 156   | 150   | 138   | 143   |
| BRISK      | 264   | 282   | 282   | 277   | 297   | 279   | 289   | 272   | 266   | 254   |
| ORB        | 92    | 102   | 106   | 113   | 109   | 125   | 130   | 129   | 127   | 128   |
| AKAZE      | 166   | 157   | 161   | 155   | 163   | 164   | 173   | 175   | 177   | 179   |
| SIFT       | 138   | 132   | 124   | 137   | 134   | 140   | 137   | 148   | 159   | 137   |

**The distribution of keypoint's neighborhood size**

**Harris** and **Shi-Tomasi** have similarly small neighborhood size. And they are uniformly distributed.

**Harris keypoints**
![HARRIS_00]

**Shi-Tomasi keypoints**
![SHITOMASI_00]

**FAST** keypoint's neighborhood size is a little larger than **Harris** and **Shi-Tomasi**. And they are a little overlapped.

**FAST keypoints**
![FAST_00]

**BRISK** and **ORB** have very large neighborhood size and they are overlapped with each other.

**BRISK keypoints**
![BRISK_00]

**ORB keypoints**
![ORB_00]

**AKAZE** and **SIFT** have some small and some large neighborhood size. And their distribution looks more uniform than other detectors.

**AKAZE keypoints**
![AKAZE_00]

**SIFT keypoints**
![SIFT_00]

### MP.8 Performance Evaluation 2
*Count the number of matched keypoints for all 10 images using all possible combinations of detectors and descriptors. In the matching step, the BF approach is used with the descriptor distance ratio set to 0.8.*

According to [this discution](https://github.com/kyamagu/mexopencv/issues/351#issuecomment-319528154), **KAZE/AKAZE** descriptors will only work with **KAZE/AKAZE** keypoints. And **SIFT(detector) + ORB(descriptor)** will encounter out of memory probloem in my case.


| detector/descriptor | Img.0-1 | Img.1-2 | Img.2-3 | Img.3-4 | Img.4-5 | Img.5-6 | Img.6-7 | Img.7-8 | Img.8-9 |
| ------------------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- |
| Harris + BRISK      | 50      | 54      | 53      | 55      | 56      | 58      | 57      | 61      | 59      |
| Harris + BRIEF      | 50      | 54      | 53      | 55      | 56      | 58      | 57      | 61      | 59      |
| Harris + ORB        | 50      | 54      | 53      | 55      | 56      | 58      | 57      | 61      | 59      |
| Harris + FREAK      | 50      | 54      | 53      | 55      | 56      | 58      | 57      | 61      | 59      |
| Harris + AKAZE      | N/A     | N/A     | N/A     | N/A     | N/A     | N/A     | N/A     | N/A     | N/A     |
| Harris + SIFT       | 50      | 54      | 53      | 55      | 56      | 58      | 57      | 61      | 59      |
|                     |         |         |         |         |         |         |         |         |         |
| SHITOMASI + BRISK   | 125     | 118     | 123     | 120     | 120     | 113     | 114     | 123     | 111     |
| SHITOMASI + BRIEF   | 125     | 118     | 123     | 120     | 120     | 113     | 114     | 123     | 111     |
| SHITOMASI + ORB     | 125     | 118     | 123     | 120     | 120     | 113     | 114     | 123     | 111     |
| SHITOMASI + FREAK   | 125     | 118     | 123     | 120     | 120     | 113     | 114     | 123     | 111     |
| SHITOMASI + AKAZE   | N/A     | N/A     | N/A     | N/A     | N/A     | N/A     | N/A     | N/A     | N/A     |
| SHITOMASI + SIFT    | 125     | 118     | 123     | 120     | 120     | 113     | 114     | 123     | 111     |
|                     |         |         |         |         |         |         |         |         |         |
| FAST + BRISK        | 149     | 152     | 150     | 155     | 149     | 149     | 156     | 150     | 138     |
| FAST + BRIEF        | 149     | 152     | 150     | 155     | 149     | 149     | 156     | 150     | 138     |
| FAST + ORB          | 149     | 152     | 150     | 155     | 149     | 149     | 156     | 150     | 138     |
| FAST + FREAK        | 149     | 152     | 150     | 155     | 149     | 149     | 156     | 150     | 138     |
| FAST + AKAZE        | N/A     | N/A     | N/A     | N/A     | N/A     | N/A     | N/A     | N/A     | N/A     |
| FAST + SIFT         | 149     | 152     | 150     | 155     | 149     | 149     | 156     | 150     | 138     |
|                     |         |         |         |         |         |         |         |         |         |
| BRISK + BRISK       | 264     | 282     | 282     | 277     | 297     | 279     | 289     | 272     | 266     |
| BRISK + BRIEF       | 264     | 282     | 282     | 277     | 297     | 279     | 289     | 272     | 266     |
| BRISK + ORB         | 264     | 282     | 282     | 277     | 297     | 279     | 289     | 272     | 266     |
| BRISK + FREAK       | 242     | 260     | 263     | 264     | 274     | 256     | 269     | 255     | 243     |
| BRISK + AKAZE       | N/A     | N/A     | N/A     | N/A     | N/A     | N/A     | N/A     | N/A     | N/A     |
| BRISK + SIFT        | 264     | 282     | 282     | 277     | 297     | 279     | 289     | 272     | 266     |
|                     |         |         |         |         |         |         |         |         |         |
| ORB + BRISK         | 83      | 93      | 95      | 103     | 101     | 116     | 120     | 120     | 119     |
| ORB + BRIEF         | 92      | 102     | 106     | 113     | 109     | 125     | 130     | 129     | 127     |
| ORB + ORB           | 92      | 102     | 106     | 113     | 109     | 125     | 130     | 129     | 127     |
| ORB + FREAK         | 46      | 53      | 56      | 65      | 55      | 64      | 66      | 71      | 73      |
| ORB + AKAZE         | N/A     | N/A     | N/A     | N/A     | N/A     | N/A     | N/A     | N/A     | N/A     |
| ORB + SIFT          | 92      | 102     | 106     | 113     | 109     | 125     | 130     | 129     | 127     |
|                     |         |         |         |         |         |         |         |         |         |
| AKAZE + BRISK       | 166     | 157     | 161     | 155     | 163     | 164     | 173     | 175     | 177     |
| AKAZE + BRIEF       | 166     | 157     | 161     | 155     | 163     | 164     | 173     | 175     | 177     |
| AKAZE + ORB         | 166     | 157     | 161     | 155     | 163     | 164     | 173     | 175     | 177     |
| AKAZE + FREAK       | 166     | 157     | 161     | 155     | 163     | 164     | 173     | 175     | 177     |
| AKAZE + AKAZE       | 166     | 157     | 161     | 155     | 163     | 164     | 173     | 175     | 177     |
| AKAZE + SIFT        | 166     | 157     | 161     | 155     | 163     | 164     | 173     | 175     | 177     |
|                     |         |         |         |         |         |         |         |         |         |
| SIFT + BRISK        | 137     | 132     | 124     | 137     | 134     | 140     | 137     | 148     | 159     |
| SIFT + BRIEF        | 138     | 132     | 124     | 137     | 134     | 140     | 137     | 148     | 159     |
| SIFT + ORB          | N/A     | N/A     | N/A     | N/A     | N/A     | N/A     | N/A     | N/A     | N/A     |
| SIFT + FREAK        | 137     | 131     | 123     | 136     | 133     | 139     | 135     | 147     | 158     |
| SIFT + AKAZE        | N/A     | N/A     | N/A     | N/A     | N/A     | N/A     | N/A     | N/A     | N/A     |
| SIFT + SIFT         | 138     | 132     | 124     | 137     | 134     | 140     | 137     | 148     | 159     |

### MP.9 Performance Evaluation 3
*Log the time it takes for keypoint detection and descriptor extraction. The results must be entered into a spreadsheet and based on this data, the TOP3 detector / descriptor combinations must be recommended as the best choice for our purpose of detecting keypoints on vehicles.*



| Detectors & Descriptors | BRISK   | BRIEF   | ORB     | FREAK   | AKAZE    | SIFT     |
| ----------------------- | ------- | ------- | ------- | ------- | -------- | -------- |
| Harris                  | 12.89ms | 13.27ms | 15.05ms | 31.97ms | N/A      | 29.33ms  |
| Shi-Tomasi              | 14.51ms | 13.96ms | 16.25   | 32.88ms | N/A      | 29.46ms  |
| FAST                    | 2.85ms  | 1.86ms  | 3.93ms  | 25.08ms | N/A      | 19.32ms  |
| BRISK                   | 35.36ms | 34.32ms | 43.15ms | 56.28ms | N/A      | 57.23ms  |
| ORB                     | 7.88ms  | 6.89ms  | 16.44ms | 30.08ms | N/A      | 33.57ms  |
| AKAZE                   | 57.96ms | 55.61ms | 62.67ms | 78.50ms | 103.22ms | 72.81ms  |
| SIFT                    | 84.70ms | 84.63ms | N/A     | 108.17  | N/A      | 148.09ms |


Consider to the balance of keypoint's distribution, and time efficiency, my top 3 detector + descriptor conbinations would be:
1. FAST + BRIEF
2. FAST + BRISK
3. FAST + ORB