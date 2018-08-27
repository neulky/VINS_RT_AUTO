#pragma once

#include <ros/ros.h>
#include <vector>
#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <fstream>
#include "utility/utility.h"

const double FOCAL_LENGTH = 460.0;
const int WINDOW_SIZE = 10;
const int NUM_OF_CAM = 1;
const int NUM_OF_F = 1000;

extern double INIT_DEPTH;
extern double MIN_PARALLAX;
extern int ESTIMATE_EXTRINSIC;

extern std::string EX_CALIB_RESULT_PATH;

extern std::vector<Eigen::Matrix3d> RIC;
extern std::vector<Eigen::Vector3d> TIC;
extern Eigen::Vector3d G;

extern double SOLVER_TIME;

extern double TD;
extern double TR;
extern int ESTIMATE_TD;
extern double ROW, COL;

//优化
extern int NUM_ITERATIONS;


void readParameters(ros::NodeHandle &n);
enum SIZE_PARAMETERIZATION
{
  SIZE_POSE = 7,
  SIZE_SPEEDBIAS = 9,
  SIZE_FEATURE = 1
};

enum StateOrder
{
  O_P = 0,
  O_R = 3,
  O_V = 6,
  O_BA = 9,
  O_BG = 12
};

enum NoiseOrder
{
  O_AN = 0,
  O_GN = 3,
  O_AW = 6,
  O_GW = 9
};
