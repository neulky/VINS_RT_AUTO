#pragma once
#include <eigen3/Eigen/Dense>
#include <iostream>
#include "../utility/utility.h"
#include <ros/ros.h>
#include <map>
#include "../feature_manager.h"
#include <opencv2/core/core.hpp>

using namespace Eigen;
using namespace std;

class ImageFrame
{
    public:
        ImageFrame(){};
        ImageFrame(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>>& _points, double _t):t{_t},is_key_frame{false}
        {
            points = _points;
        };
        map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>> > > points;
        double t;
        Matrix3d R;
        Vector3d T;
	Matrix3d imu_R;
	Vector3d imu_T;
        bool is_key_frame;
};

bool VisualIMUAlignment(map<double, ImageFrame> &all_image_frame, double &s, Vector3d &tic);