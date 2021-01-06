#ifndef matching2D_hpp
#define matching2D_hpp

#include <stdio.h>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"

void detKeypointsShiTomasi(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img,
                           double &time, bool useHarris = false,
                           bool bVis = false);

void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img,
                        std::string detectorType, double &time,
                        bool bVis = false);
void descKeypoints(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img,
                   cv::Mat &descriptors, std::string descriptorMethod,
                   double &time);
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource,
                      std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource,
                      cv::Mat &descRef, std::vector<cv::DMatch> &matches,
                      std::string descriptorType, std::string matcherType,
                      std::string selectorType, double &time);

void showImage(const std::string &type, std::vector<cv::KeyPoint> &keypoints,
               cv::Mat &img);

#endif /* matching2D_hpp */
