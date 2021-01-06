/* INCLUDES FOR THIS PROJECT */
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <list>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <sstream>
#include <vector>
#include "json.hpp"

#include "dataStructures.h"
#include "matching2D.hpp"

using namespace std;

struct Params {
  string detectorType;
  string descriptorMethod;
  string matcherType;
  string descriptorType;
  string selectorType;

  Params() = default;
  ~Params() = default;
};

Params params;

void readConfig() {
  using json = nlohmann::json;

  std::ifstream json_stream("../src/config/params.json");
  json params_json;
  json_stream >> params_json;
  json_stream.close();

  params.detectorType = params_json["detectorType"];
  params.descriptorMethod = params_json["descriptorMethod"];
  params.matcherType = params_json["matcherType"];
  params.descriptorType = params_json["descriptorType"];
  params.selectorType = params_json["selectorType"];

  /* params.filterRes = params_json["filterRes"];
  params.minPoint =
      Eigen::Vector4f(params_json["minPoint"][0], params_json["minPoint"][1],
                      params_json["minPoint"][2], 1);
  params.maxPoint =
      Eigen::Vector4f(params_json["maxPoint"][0], params_json["maxPoint"][1],
                      params_json["maxPoint"][2], 1);
  params.clusterTol = params_json["clusterTol"];
  params.clusterMinSize = params_json["clusterMinSize"];
  params.clusterMaxSize = params_json["clusterMaxSize"]; */
}

/* MAIN PROGRAM */
int main(int argc, const char *argv[]) {
  /* INIT VARIABLES AND DATA STRUCTURES */

  // Read config
  readConfig();

  // data location
  string dataPath = "../";

  // camera
  string imgBasePath = dataPath + "images/";
  string imgPrefix =
      "KITTI/2011_09_26/image_00/data/000000";  // left camera, color
  string imgFileType = ".png";
  int imgStartIndex = 0;  // first file index to load (assumes Lidar and camera
                          // names have identical naming convention)
  int imgEndIndex = 9;    // last file index to load
  int imgFillWidth =
      4;  // no. of digits which make up the file index (e.g. img-0001.png)

  // misc
  int dataBufferSize = 2;      // no. of images which are held in memory (ring
                               // buffer) at the same time
  list<DataFrame> dataBuffer;  // list of data frames which are held in memory
                               // at the same time
  bool bVis = false;           // visualize results

  /* MAIN LOOP OVER ALL IMAGES */
  int counter = 0;
  double keypointTime = 0.0;
  double descriptorTime = 0.0;
  double matchTime = 0.0;
  for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex;
       imgIndex++, counter++) {
    /* LOAD IMAGE INTO BUFFER */

    // assemble filenames for current index
    ostringstream imgNumber;
    imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
    string imgFullFilename =
        imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

    // load image from file and convert to grayscale
    cv::Mat img, imgGray;
    img = cv::imread(imgFullFilename);
    cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

    //// STUDENT ASSIGNMENT
    //// TASK MP.1 -> replace the following code with ring buffer of size
    /// dataBufferSize
    if (dataBuffer.size() == dataBufferSize) {
      dataBuffer.pop_front();
    }

    // push image into data frame buffer
    DataFrame frame;
    frame.cameraImg = imgGray;
    dataBuffer.push_back(frame);

    //// EOF STUDENT ASSIGNMENT
    cout << "#1 : LOAD IMAGE INTO BUFFER done" << endl;

    /* DETECT IMAGE KEYPOINTS */

    /*  extract 2D keypoints from current image */
    // create empty feature list for current image
    vector<cv::KeyPoint> keypoints;
    string detectorType = params.detectorType;

    //// STUDENT ASSIGNMENT
    //// TASK MP.2 -> add the following keypoint detectors in file
    /// matching2D.cpp and enable string-based selection based on detectorType /
    ///-> HARRIS, FAST, BRISK, ORB, AKAZE, SIFT

    if (detectorType.compare("SHITOMASI") == 0) {
      detKeypointsShiTomasi(keypoints, imgGray, keypointTime, false, false);
    } else if (detectorType.compare("HARRIS") == 0) {
      detKeypointsShiTomasi(keypoints, imgGray, keypointTime, true, false);
    } else if (detectorType.compare("FAST") == 0 ||
               detectorType.compare("BRISK") == 0 ||
               detectorType.compare("ORB") == 0 ||
               detectorType.compare("AKAZE") == 0 ||
               detectorType.compare("SIFT") == 0) {
      detKeypointsModern(keypoints, imgGray, detectorType, keypointTime, false);
    }
    //// EOF STUDENT ASSIGNMENT

    //// STUDENT ASSIGNMENT
    //// TASK MP.3 -> only keep keypoints on the preceding vehicle

    // only keep keypoints on the preceding vehicle
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
    /* if (imgIndex == 0) {
      cv::Mat visImage = img.clone();

      cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1),
                        cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
      cv::rectangle(visImage, vehicleRect, cv::Scalar(0, 0, 255), 1);
      imwrite("../output/" + detectorType + "_00.png", visImage);
    }
    cout << detectorType << " detects " << keypoints.size() << " keypoints"
         << endl; */

    //// EOF STUDENT ASSIGNMENT

    // optional : limit number of keypoints (helpful for debugging and learning)
    bool bLimitKpts = false;
    if (bLimitKpts) {
      int maxKeypoints = 50;

      if (detectorType.compare("SHITOMASI") == 0 ||
          detectorType.compare("HARRIS") == 0) {
        // there is no response info, so keep the first 50 as they are
        // sorted in descending quality order
        keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
      }
      cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
      cout << " NOTE: Keypoints have been limited!" << endl;
    }

    // push keypoints and descriptor for current frame to end of data buffer
    dataBuffer.rbegin()->keypoints = keypoints;
    cout << "#2 : DETECT KEYPOINTS done" << endl;

    /* EXTRACT KEYPOINT DESCRIPTORS */

    //// STUDENT ASSIGNMENT
    //// TASK MP.4 -> add the following descriptors in file matching2D.cpp and
    /// enable string-based selection based on descriptorMethod / -> BRIEF, ORB,
    /// FREAK, AKAZE, SIFT

    cv::Mat descriptors;
    string descriptorMethod =
        params.descriptorMethod;  // BRIEF, ORB, FREAK, AKAZE, SIFT
    descKeypoints(dataBuffer.rbegin()->keypoints,
                  dataBuffer.rbegin()->cameraImg, descriptors, descriptorMethod,
                  descriptorTime);
    //// EOF STUDENT ASSIGNMENT

    // push descriptors for current frame to end of data buffer
    dataBuffer.rbegin()->descriptors = descriptors;

    cout << "#3 : EXTRACT DESCRIPTORS done" << endl;

    // wait until at least two images have been processed
    if (dataBuffer.size() > 1) {
      /* MATCH KEYPOINT DESCRIPTORS */

      vector<cv::DMatch> matches;
      string matcherType = params.matcherType;        // MAT_BF, MAT_FLANN
      string descriptorType = params.descriptorType;  // DES_BINARY, DES_HOG
      string selectorType = params.selectorType;      // SEL_NN, SEL_KNN

      //// STUDENT ASSIGNMENT
      //// TASK MP.5 -> add FLANN matching in file matching2D.cpp
      //// TASK MP.6 -> add KNN match selection and perform descriptor distance
      /// ratio filtering with t=0.8 in file matching2D.cpp

      matchDescriptors(next(dataBuffer.rbegin())->keypoints,
                       dataBuffer.rbegin()->keypoints,
                       next(dataBuffer.rbegin())->descriptors,
                       dataBuffer.rbegin()->descriptors, matches,
                       descriptorType, matcherType, selectorType, matchTime);

      //// EOF STUDENT ASSIGNMENT

      // store matches in current data frame
      dataBuffer.rbegin()->kptMatches = matches;

      cout << "#4 : MATCH KEYPOINT DESCRIPTORS done" << endl << endl;

      // visualize matches between current and previous image
      bVis = false;
      if (bVis) {
        cv::Mat matchImg = (dataBuffer.rbegin()->cameraImg).clone();
        cv::drawMatches(
            next(dataBuffer.rbegin())->cameraImg,
            next(dataBuffer.rbegin())->keypoints,
            dataBuffer.rbegin()->cameraImg, dataBuffer.rbegin()->keypoints,
            matches, matchImg, cv::Scalar::all(-1), cv::Scalar::all(-1),
            vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

        string windowName = "Matching keypoints between two camera images";
        cv::namedWindow(windowName, 7);
        cv::imshow(windowName, matchImg);
        cout << "Press key to continue to next image" << endl;
        cv::waitKey(0);  // wait for key to be pressed
      }
      bVis = false;
    }

  }  // eof loop over all images

  // Log for analysis
  cout << "Average Keypoint time: " << keypointTime / counter << endl;
  cout << "Average Descriptor time: " << descriptorTime / counter << endl;
  cout << "Average Match time: " << matchTime / (counter - 1) << endl << endl;

  return 0;
}
