#include <stdio.h>
#include <queue>
#include <map>
#include <thread>
#include <mutex>
#include <condition_variable>

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include "utility/visualization.h"
#include "estimator.h"
#include "parameters.h"
#include "fstream"

using namespace std;

Estimator estimator;

int sum_of_wait = 0;
std::condition_variable con;

bool init_feature = 0;
bool init_imu = 1;

queue<geometry_msgs::PoseStampedConstPtr> pose_buf;
queue<sensor_msgs::PointCloudConstPtr> feature_buf;

std::mutex m_buf;

void feature_callback(const sensor_msgs::PointCloudConstPtr &feature_msg)
{
  
    if (!init_feature)
    {
        //skip the first detected feature, which doesn't contain optical flow speed
        init_feature = 1;
        return;
    }
    m_buf.lock();
    feature_buf.push(feature_msg);
    m_buf.unlock();
    con.notify_one();
    
    ROS_INFO_STREAM("stamp = " << setprecision(19) << feature_msg->header.stamp);
    ROS_INFO_STREAM("stamp.tosec() = " << setprecision(19) << feature_msg->header.stamp);
}

void pose_callback(const geometry_msgs::PoseStampedConstPtr &pose_msg)
{
  m_buf.lock();
  pose_buf.push(pose_msg);
  m_buf.unlock();
  con.notify_one();
}

std::vector<std::pair<geometry_msgs::PoseStampedConstPtr, sensor_msgs::PointCloudConstPtr>> getMeasurements()
{
  std::vector<std::pair<geometry_msgs::PoseStampedConstPtr, sensor_msgs::PointCloudConstPtr>> measurements;
  
  while(true)
  {
    if(pose_buf.empty() || feature_buf.empty())
      return measurements;
    if(!(pose_buf.back()->header.stamp > feature_buf.front()->header.stamp))
    {
      ROS_WARN("wait for pose");
      sum_of_wait++;
      return measurements;
    }
    if(!(pose_buf.front()->header.stamp < feature_buf.front()->header.stamp))
    {
      ROS_WARN("throw img");
      feature_buf.pop();
      continue;
    }
    sensor_msgs::PointCloudConstPtr img_msg = feature_buf.front();
    feature_buf.pop();
    geometry_msgs::PoseStampedConstPtr pose_msg = pose_buf.front();
    pose_buf.pop();
    measurements.emplace_back(pose_msg, img_msg);
  }
  return measurements;
}

void process()
{
  while(true)
  {
    std::vector<std::pair<geometry_msgs::PoseStampedConstPtr, sensor_msgs::PointCloudConstPtr>> measurements;
    std::unique_lock<std::mutex> lk(m_buf);
    con.wait(lk, [&]
	{
	  return (measurements = getMeasurements()).size() != 0;
	});
    lk.unlock();
    
    for(auto &measurement : measurements)
    {
      auto img_msg = measurement.second;
      auto pose_msg = measurement.first;
      
      double rx = 0, ry = 0, rz = 0, rw = 0, px = 0, py = 0, pz = 0;
      rw = pose_msg->pose.orientation.w;
      rx = pose_msg->pose.orientation.x;
      ry = pose_msg->pose.orientation.y;
      rz = pose_msg->pose.orientation.z;
      px = pose_msg->pose.position.x;
      py = pose_msg->pose.position.y;
      pz = pose_msg->pose.position.z;
      
      Eigen::Quaterniond R_temp(rw, rx, ry, rz);
      Eigen::Vector3d P_temp(px, py, pz);
      
      estimator.processPR(R_temp.toRotationMatrix(), P_temp);
      
      map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> image;
      for(unsigned int i = 0;i < img_msg->points.size();i++)
      {
	int v = img_msg->channels[0].values[i] + 0.5;
	int feature_id = v / NUM_OF_CAM;
	int camera_id = v % NUM_OF_CAM;
	double x = img_msg->points[i].x;
	double y = img_msg->points[i].y;
	double z = img_msg->points[i].z;
	double p_u = img_msg->channels[1].values[i];
	double p_v = img_msg->channels[2].values[i];
	double velocity_x = img_msg->channels[3].values[i];
	double velocity_y = img_msg->channels[4].values[i];
	ROS_ASSERT(z == 1);
	Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
	xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
	image[feature_id].emplace_back(camera_id, xyz_uv_velocity);
      }
      estimator.processImage(image, img_msg->header);
    }   
  }
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "estimator_rt");
  ros::NodeHandle n("~");
  ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
  readParameters(n);
  
  estimator.setParameter();
  
  ros::Subscriber sub_pr = n.subscribe("/location_pr", 2000, pose_callback);
  ros::Subscriber sub_image = n.subscribe("/feature_tracker/feature", 2000, feature_callback);
  //ros::Subscriber sub_restart = n.Subscriber("/feature_tracker/restart", 2000, restart_callback);

  std::thread measurement_process{process};
  
  ros::spin();
  
  return 0;
}