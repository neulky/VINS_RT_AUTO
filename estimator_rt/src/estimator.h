#pragma once

#include "parameters.h"
#include "feature_manager.h"
#include "utility/utility.h"
#include "utility/tic_toc.h"
#include "initial/initial_ex_rotation.h"
#include "initial/initial_sfm.h"
#include "initial/initial_alignment.h"
#include "initial/solve_5pts.h"

#include <std_msgs/Header.h>
#include <std_msgs/Float32.h>

#include <ceres/ceres.h>
#include "factor/projection_factor.h"
#include "factor/pose_local_parameterization.h"

#include <opencv2/core/eigen.hpp>
#include <queue>
#include <unordered_map>

class Estimator
{
public:
  Estimator();
  
  void setParameter();
  
  void processImage(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, const std_msgs::Header &header);
  
  void processPR(Eigen::Matrix3d _R, Vector3d _P);
  
  void clearState();
  
  void solverOdometry();
  
  void optimization();
  
  void vector2double();
  
  void double2vector();
  
  void slideWindow();
  
  void slideWindowNew();
  
  void slideWindowOld();
  
  bool initialStructure();
  
  bool visualInitialAlign();
  
  bool relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l);
  
  enum MarginalizationFlag
  {
    MARGIN_OLD = 0,
    MARGIN_SECOND_NEW = 1
  };
  
  enum SolverFlag
  {
    INITIAL,
    NON_LINEAR
  };
  
  MarginalizationFlag marginalization_flag;
  SolverFlag solver_flag;
  Vector3d g;
  
  Matrix3d ric[NUM_OF_CAM];
  Vector3d tic[NUM_OF_CAM];
  
  Matrix3d relative_R[(WINDOW_SIZE + 1)];
  Vector3d relative_T[(WINDOW_SIZE + 1)];
  
  Matrix3d Rs[(WINDOW_SIZE + 1)];
  Vector3d Ps[(WINDOW_SIZE + 1)];
  
  Matrix3d R0;
  Vector3d P0;
  
  bool first_frame;
  
  double para_Pose[WINDOW_SIZE + 1][SIZE_POSE];
  double para_Ex_Pose[NUM_OF_CAM][SIZE_POSE];
  double para_Feature[NUM_OF_F][SIZE_FEATURE];
  
  Matrix3d back_R0, last_R, last_R0;
  Vector3d back_P0, last_P, last_P0;
  std_msgs::Header Headers[(WINDOW_SIZE + 1)];
  map<double, ImageFrame> all_image_frame;
  
  FeatureManager f_manager;
  InitialEXRotation initial_ex_rotation;
  MotionEstimator m_estimator;
  
  int frame_count;
  
  int sum_of_front, sum_of_back;
  double td;

};
