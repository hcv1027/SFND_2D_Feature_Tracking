#include <numeric>
#include "matching2D.hpp"

using namespace std;

// Find best matches for keypoints in two camera images based on several
// matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource,
                      std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource,
                      cv::Mat &descRef, std::vector<cv::DMatch> &matches,
                      std::string descriptorType, std::string matcherType,
                      std::string selectorType) {
  // configure matcher
  bool crossCheck = false;
  cv::Ptr<cv::DescriptorMatcher> matcher;

  if (matcherType.compare("MAT_BF") == 0) {
    int normType = cv::NORM_HAMMING;
    matcher = cv::BFMatcher::create(normType, crossCheck);
  } else if (matcherType.compare("MAT_FLANN") == 0) {
    // ...
  }

  // perform matching task
  if (selectorType.compare("SEL_NN") == 0) {  // nearest neighbor (best match)

    matcher->match(
        descSource, descRef,
        matches);  // Finds the best match for each descriptor in desc1
  } else if (selectorType.compare("SEL_KNN") ==
             0) {  // k nearest neighbors (k=2)

    // ...
  }
}

// Use one of several types of state-of-art descriptors to uniquely identify
// keypoints
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img,
                   cv::Mat &descriptors, string descriptorType) {
  // select appropriate descriptor
  cv::Ptr<cv::DescriptorExtractor> extractor;
  if (descriptorType.compare("BRISK") == 0) {
    int threshold = 30;         // FAST/AGAST detection threshold score.
    int octaves = 3;            // detection octaves (use 0 to do single scale)
    float patternScale = 1.0f;  // apply this scale to the pattern used for
                                // sampling the neighbourhood of a keypoint.

    extractor = cv::BRISK::create(threshold, octaves, patternScale);
  } else {
    //...
  }

  // perform feature description
  double t = (double)cv::getTickCount();
  extractor->compute(img, keypoints, descriptors);
  t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
  cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0
       << " ms" << endl;
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img,
                           bool useHarris, bool bVis) {
  // compute detector parameters based on image size
  int blockSize = 4;  //  size of an average block for computing a derivative
                      //  covariation matrix over each pixel neighborhood
  double maxOverlap = 0.0;  // max. permissible overlap
                            // between two features in %
  double minDistance = (1.0 - maxOverlap) * blockSize;
  int maxCorners =
      img.rows * img.cols / max(1.0, minDistance);  // max. num. of keypoints

  double qualityLevel = 0.01;  // minimal accepted quality of image corners
  double k = 0.04;

  // Apply corner detection
  double t = (double)cv::getTickCount();
  vector<cv::Point2f> corners;
  cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance,
                          cv::Mat(), blockSize, useHarris, k);

  // add corners to result vector
  for (auto it = corners.begin(); it != corners.end(); ++it) {
    cv::KeyPoint newKeyPoint;
    newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
    newKeyPoint.size = blockSize;
    keypoints.push_back(newKeyPoint);
  }
  t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
  string name = useHarris ? "Harris " : "Shi-Tomasi ";
  cout << name << "detection with n=" << keypoints.size() << " keypoints in "
       << 1000 * t / 1.0 << " ms" << endl;

  // visualize results
  if (bVis) {
    showImage(name, keypoints, img);
  }
}

void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img,
                        std::string detectorType, bool bVis) {
  cv::Ptr<cv::FeatureDetector> detector;
  if (detectorType.compare("FAST") == 0) {
    int threshold = 30;  // difference between intensity of the central pixel
                         // and pixels of a circle around this pixel
    bool bNMS = true;    // perform non-maxima suppression on keypoints
    cv::FastFeatureDetector::DetectorType type =
        cv::FastFeatureDetector::TYPE_9_16;  // TYPE_9_16, TYPE_7_12, TYPE_5_8
    detector = cv::FastFeatureDetector::create(threshold, bNMS, type);
  } else if (detectorType.compare("BRISK") == 0) {
    int threshold = 30;         // FAST/AGAST detection threshold score.
    int octaves = 3;            // detection octaves (use 0 to do single scale)
    float patternScale = 1.0f;  // apply this scale to the pattern used for
                                // sampling the neighbourhood of a keypoint.

    detector = cv::BRISK::create(threshold, octaves, patternScale);
  } else if (detectorType.compare("ORB") == 0) {
    int nfeatures = 500;
    float scaleFactor = 1.2f;
    int nlevels = 8;
    int edgeThreshold = 31;
    int firstLevel = 0;
    int WTA_K = 2;
    cv::ORB::ScoreType scoreType = cv::ORB::HARRIS_SCORE;
    int patchSize = 31;
    int fastThreshold = 20;
    detector =
        cv::ORB::create(nfeatures, scaleFactor, nlevels, edgeThreshold,
                        firstLevel, WTA_K, scoreType, patchSize, fastThreshold);
  } else if (detectorType.compare("AKAZE") == 0) {
    cv::AKAZE::DescriptorType descriptor_type = cv::AKAZE::DESCRIPTOR_MLDB;
    int descriptor_size = 0;
    int descriptor_channels = 3;
    float threshold = 0.001f;
    int nOctaves = 4;
    int nOctaveLayers = 4;
    cv::KAZE::DiffusivityType diffusivity = cv::KAZE::DIFF_PM_G2;
    detector =
        cv::AKAZE::create(descriptor_type, descriptor_size, descriptor_channels,
                          threshold, nOctaves, nOctaveLayers, diffusivity);
  } else if (detectorType.compare("SIFT") == 0) {
    int nfeatures = 0;
    int nOctaveLayers = 3;
    double contrastThreshold = 0.04;
    double edgeThreshold = 10;
    double sigma = 1.6;
    detector = cv::SIFT::create(nfeatures, nOctaveLayers, contrastThreshold,
                                edgeThreshold, sigma);
  }

  // Apply corner detection
  double t = (double)cv::getTickCount();
  detector->detect(img, keypoints);
  t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
  cout << detectorType << " detection with n=" << keypoints.size()
       << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

  // visualize results
  if (bVis) {
    showImage(detectorType, keypoints, img);
  }
}

void showImage(const std::string &type, std::vector<cv::KeyPoint> &keypoints,
               cv::Mat &img) {
  cv::Mat visImage = img.clone();
  cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1),
                    cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
  string windowName = type + " Corner Detector Results";
  cv::namedWindow(windowName, 6);
  imshow(windowName, visImage);
  cv::waitKey(0);
}

void temp(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img,
          bool bVis = false) {
  // Apply corner detection
  double t = (double)cv::getTickCount();
  vector<cv::Point2f> corners;

  // add corners to result vector
  for (auto it = corners.begin(); it != corners.end(); ++it) {
    cv::KeyPoint newKeyPoint;
    // newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
    // newKeyPoint.size = blockSize;
    keypoints.push_back(newKeyPoint);
  }
  t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
  cout << "Harris detection with n=" << keypoints.size() << " keypoints in "
       << 1000 * t / 1.0 << " ms" << endl;

  // visualize results
  if (bVis) {
    showImage("Harris", keypoints, img);
  }
}