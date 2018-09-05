#include "estimator.h"

Estimator::Estimator():f_manager{Rs}
{
  ROS_INFO("init begins");
  clearState();
}

void Estimator::setParameter()
{
  for(int i = 0;i < NUM_OF_CAM;i++)
  {
    ric[i] = RIC[i];
    tic[i] = TIC[i];
  }
  f_manager.setRic(ric);
  
}


void Estimator::clearState()
{
  for(int i = 0;i < WINDOW_SIZE + 1;i++)
  {
    Rs[i].setIdentity();
    Ps[i].setZero();
    
    relative_R[i].setIdentity();
    relative_T[i].setZero();
  }
  
  for(int i = 0;i < NUM_OF_CAM;i++)
  {
    ric[i].setIdentity();
    tic[i].setZero();
  }
  
  sum_of_back = 0;
  sum_of_front = 0;
  frame_count = 0;
  solver_flag = INITIAL;
  first_frame = true;
  td = TD;
}

void Estimator::processPR(Eigen::Matrix3d _R, Vector3d _P)
{
  if(first_frame == true)
  {
    R0 = _R;
    P0 = _P;
    first_frame = false;
  }
  Rs[frame_count] = R0.inverse() * _R;
  Ps[frame_count] = _P - P0;
  
  if(frame_count != 0)
  {
    relative_R[frame_count] = Rs[frame_count-1].inverse() * _R;
    relative_T[frame_count] = _P - Ps[frame_count - 1];
  }
}

void Estimator::processImage(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, const std_msgs::Header &header)
{
  ROS_DEBUG("new image coming -------------------------------------");
  ROS_DEBUG("add feature points %lu", image.size());
  if(f_manager.addFeatureCheckParallax(frame_count, image, td))
    marginalization_flag = MARGIN_OLD;
  else
    marginalization_flag = MARGIN_SECOND_NEW;
  
  ImageFrame imageframe(image, header.stamp.toSec());
  imageframe.imu_R = Rs[frame_count];
  imageframe.imu_T = Ps[frame_count];
  all_image_frame.insert(make_pair(header.stamp.toSec(), imageframe));
  Headers[frame_count] = header;
  if(ESTIMATE_EXTRINSIC == 2)
  {
    if(frame_count !=0)
    {
      vector<pair<Vector3d, Vector3d>> corres = f_manager.getCorresponding(frame_count - 1, frame_count);
      Matrix3d calib_ric;
      Quaterniond relative_Q{relative_R[frame_count]}; 
      if(initial_ex_rotation.CalibrationExRotation(corres, relative_Q, calib_ric))
      {
	ROS_WARN("initial extrinsic rotation calib success");
        ROS_WARN_STREAM("initial extrinsic rotation: " << endl << calib_ric);
	ric[0] = calib_ric;
	RIC[0] = calib_ric;
	ESTIMATE_EXTRINSIC = 1;
      }
    }
  }
  
  if(solver_flag == INITIAL)
  {
    if(frame_count == WINDOW_SIZE)
    {
      bool result = false;
      if(ESTIMATE_EXTRINSIC != 2)
      {
	result = initialStructure();
      }
      if(result)
      {
	solver_flag = NON_LINEAR;
	solverOdometry();
	slideWindow();
	f_manager.removeFailures();
	ROS_INFO("Initialization finish!");
	last_R = Rs[WINDOW_SIZE];
	last_P = Ps[WINDOW_SIZE];
	last_R0 = Rs[0];
	last_P0 = Ps[0];
      }
      else
	slideWindow();
    }
    else
      frame_count++;
  }
  else
  {
    solverOdometry();
    slideWindow();
    f_manager.removeFailures();
    
    last_R = Rs[WINDOW_SIZE];
    last_P = Ps[WINDOW_SIZE];
    last_R0 = Rs[0];
    last_P0 = Ps[0];
  }
     
}
bool Estimator::initialStructure()
{
  Quaterniond Q[frame_count + 1];
  Vector3d T[frame_count + 1];
  map<int, Vector3d> sfm_tracked_points;
  vector<SFMFeature> sfm_f;
  for(auto &it_per_id : f_manager.feature)
  {
    int imu_j = it_per_id.start_frame - 1;
    SFMFeature tmp_feature;
    tmp_feature.state = false;
    tmp_feature.id = it_per_id.feature_id;
    for (auto &it_per_frame : it_per_id.feature_per_frame)
    {
      imu_j++;
      Vector3d pts_j = it_per_frame.point;
      tmp_feature.observation.push_back(make_pair(imu_j, Eigen::Vector2d{pts_j.x(), pts_j.y()}));
    }
    sfm_f.push_back(tmp_feature);
  }
  Matrix3d relative_R;
  Vector3d relative_T;
  int l;
  if (!relativePose(relative_R, relative_T, l))
  {
      ROS_INFO("Not enough features or parallax; Move device around");
      return false;
  }
  GlobalSFM sfm;
  if(!sfm.construct(frame_count + 1, Q, T, l,
	    relative_R, relative_T,
	    sfm_f, sfm_tracked_points))
  {
      ROS_DEBUG("global SFM failed!");
      marginalization_flag = MARGIN_OLD;
      return false;
  }

  //solve pnp for all frame
  map<double, ImageFrame>::iterator frame_it;
  map<int, Vector3d>::iterator it;
  frame_it = all_image_frame.begin();
  for (int i = 0; frame_it != all_image_frame.end(); frame_it++)
  {
      // provide initial guess
      cv::Mat r, rvec, t, D, tmp_r;
      if((frame_it->first) == Headers[i].stamp.toSec())
      {
	  frame_it->second.is_key_frame = true;
	  frame_it->second.R = Q[i].toRotationMatrix() * RIC[0].transpose();
	  frame_it->second.T = T[i];
	  i++;
	  continue;
      }
      if((frame_it->first) > Headers[i].stamp.toSec())
      {
	  i++;
      }
      Matrix3d R_inital = (Q[i].inverse()).toRotationMatrix();
      Vector3d P_inital = - R_inital * T[i];
      cv::eigen2cv(R_inital, tmp_r);
      cv::Rodrigues(tmp_r, rvec);
      cv::eigen2cv(P_inital, t);

      frame_it->second.is_key_frame = false;
      vector<cv::Point3f> pts_3_vector;
      vector<cv::Point2f> pts_2_vector;
      for (auto &id_pts : frame_it->second.points)
      {
	  int feature_id = id_pts.first;
	  for (auto &i_p : id_pts.second)
	  {
	      it = sfm_tracked_points.find(feature_id);
	      if(it != sfm_tracked_points.end())
	      {
		  Vector3d world_pts = it->second;
		  cv::Point3f pts_3(world_pts(0), world_pts(1), world_pts(2));
		  pts_3_vector.push_back(pts_3);
		  Vector2d img_pts = i_p.second.head<2>();
		  cv::Point2f pts_2(img_pts(0), img_pts(1));
		  pts_2_vector.push_back(pts_2);
	      }
	  }
      }
      cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);     
      if(pts_3_vector.size() < 6)
      {
	  cout << "pts_3_vector size " << pts_3_vector.size() << endl;
	  ROS_DEBUG("Not enough points for solve pnp !");
	  return false;
      }
      if (! cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1))
      {
	  ROS_DEBUG("solve pnp fail!");
	  return false;
      }
      cv::Rodrigues(rvec, r);
      MatrixXd R_pnp,tmp_R_pnp;
      cv::cv2eigen(r, tmp_R_pnp);
      R_pnp = tmp_R_pnp.transpose();
      MatrixXd T_pnp;
      cv::cv2eigen(t, T_pnp);
      T_pnp = R_pnp * (-T_pnp);
      frame_it->second.R = R_pnp * RIC[0].transpose();
      frame_it->second.T = T_pnp;
  }
  if (visualInitialAlign())
      return true;
  else
  {
      ROS_INFO("misalign visual structure with IMU");
      return false;
  }  
}

bool Estimator::visualInitialAlign()
{
  double s;
  Vector3d tic1;
  //solve scale
  bool result = VisualIMUAlignment(all_image_frame, s, tic1);
  if(!result)
  {
    ROS_DEBUG("solve g failed!");
    return false;
  }
  
  //change state
  Vector3d Ps_tmp[WINDOW_SIZE + 1];
  for(int i = 0; i <= frame_count; i++)
  {
    //no scale
    Ps_tmp[i] = all_image_frame[Headers[i].stamp.toSec()].T;
    all_image_frame[Headers[i].stamp.toSec()].is_key_frame = true;
  }
  //ESTIMATE_EXTRINSIC_TIC == 0 tic have value
  if(ESTIMATE_EXTRINSIC_TIC == 2)
    tic[0] = tic1;
  
  VectorXd dep = f_manager.getDepthVector();
  for(int i = 0; i < dep.size(); i++)
    dep[i] = -1;
  f_manager.clearDepth(dep);
  
  //triangulate on cam pose , no tic
  Vector3d TIC_TMP[NUM_OF_CAM];
  for(int i = 0; i < NUM_OF_CAM; i++)
    TIC_TMP[i].setZero();
  ric[0] = RIC[0];
  f_manager.setRic(ric);
  f_manager.triangulate(Ps_tmp, &(TIC_TMP[0]), &(RIC[0]));
  
  //restore the depth of 3d points in space according to the scale
  for(auto &it_per_id : f_manager.feature)
  {
    it_per_id.used_num = it_per_id.feature_per_frame.size();
    if(!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
      continue;
    it_per_id.estimated_depth *= s;
  }
  
  return true;
}

bool Estimator::relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l)
{
    // find previous frame which contians enough correspondance and parallex with newest frame
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        vector<pair<Vector3d, Vector3d>> corres;
        corres = f_manager.getCorresponding(i, WINDOW_SIZE);
        if (corres.size() > 20)
        {
            double sum_parallax = 0;
            double average_parallax;
            for (int j = 0; j < int(corres.size()); j++)
            {
                Vector2d pts_0(corres[j].first(0), corres[j].first(1));
                Vector2d pts_1(corres[j].second(0), corres[j].second(1));
                double parallax = (pts_0 - pts_1).norm();
                sum_parallax = sum_parallax + parallax;

            }
            average_parallax = 1.0 * sum_parallax / int(corres.size());
            if(average_parallax * 460 > 30 && m_estimator.solveRelativeRT(corres, relative_R, relative_T))
            {
                l = i;
                ROS_DEBUG("average_parallax %f choose l %d and newest frame to triangulate the whole structure", average_parallax * 460, l);
                return true;
            }
        }
    }
    return false;
}


void Estimator::solverOdometry()
{
  if(frame_count < WINDOW_SIZE)
    return;
  //三角化求三维点的深度 estimated_depth
  f_manager.triangulate(Ps, tic, ric);
  optimization();
}

void Estimator::optimization()
{
  ceres::Problem problem;
  ceres::LossFunction *loss_function;
  //loss_function = new ceres::HuberLoss(1.0)
  loss_function = new ceres::CauchyLoss(1.0);
  for(int i = 0; i < NUM_OF_CAM; i++)
  {
    ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
    problem.AddParameterBlock(para_Ex_Pose[i], SIZE_POSE, local_parameterization);
  }
  
  vector2double();
  
  int f_m_cnt = 0;
  int feature_index = -1;
  for (auto &it_per_id : f_manager.feature)
  {
      it_per_id.used_num = it_per_id.feature_per_frame.size();
      if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
	  continue;

      ++feature_index;

      int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
      
      Vector3d pts_i = it_per_id.feature_per_frame[0].point;

      for (auto &it_per_frame : it_per_id.feature_per_frame)
      {
	  imu_j++;
	  if (imu_i == imu_j)
	  {
	      continue;
	  }
	  Vector3d pts_j = it_per_frame.point;
// 	  if (ESTIMATE_TD)
// 	  {
// 		  ProjectionTdFactor *f_td = new ProjectionTdFactor(pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
// 								    it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td,
// 								    it_per_id.feature_per_frame[0].uv.y(), it_per_frame.uv.y());
// 		  problem.AddResidualBlock(f_td, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index], para_Td[0]);
// 		  /*
// 		  double **para = new double *[5];
// 		  para[0] = para_Pose[imu_i];
// 		  para[1] = para_Pose[imu_j];
// 		  para[2] = para_Ex_Pose[0];
// 		  para[3] = para_Feature[feature_index];
// 		  para[4] = para_Td[0];
// 		  f_td->check(para);
// 		  */
// 	  }
//	  else
//	  {
	      ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j);
	      problem.AddResidualBlock(f, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index]);
//	  }
	  f_m_cnt++;
      }
  }
  
  ROS_DEBUG("visual measurement count: %d", f_m_cnt);
  
  ceres::Solver::Options options;
  
  options.linear_solver_type = ceres::DENSE_SCHUR;
  //options.num_threads = 2;
  options.trust_region_strategy_type = ceres::DOGLEG;
  options.max_num_iterations = NUM_ITERATIONS;
  
  if(marginalization_flag == MARGIN_OLD)
    options.max_solver_time_in_seconds = SOLVER_TIME * 4.0 / 5.0;
  else
    options.max_solver_time_in_seconds = SOLVER_TIME;
  
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  
  double2vector();
 
}

void Estimator::vector2double()
{
  for (int i = 0; i <= WINDOW_SIZE; i++)
  {
      para_Pose[i][0] = Ps[i].x();
      para_Pose[i][1] = Ps[i].y();
      para_Pose[i][2] = Ps[i].z();
      Quaterniond q{Rs[i]};
      para_Pose[i][3] = q.x();
      para_Pose[i][4] = q.y();
      para_Pose[i][5] = q.z();
      para_Pose[i][6] = q.w();
  }
    
  for(int i = 0; i < NUM_OF_CAM; i++)
  {
    para_Ex_Pose[i][0] = tic[i].x();
    para_Ex_Pose[i][1] = tic[i].y();
    para_Ex_Pose[i][2] = tic[i].z();
    Quaterniond q{ric[i]};
    para_Ex_Pose[i][3] = q.x();
    para_Ex_Pose[i][4] = q.y();
    para_Ex_Pose[i][5] = q.z();
    para_Ex_Pose[i][6] = q.w();
  }
  
  VectorXd dep = f_manager.getDepthVector();
  for(int i = 0;i < f_manager.getFeatureCount();i++)
  {
    para_Feature[i][0] = dep[i];
  }
}

void Estimator::double2vector()
{
  for (int i = 0; i < NUM_OF_CAM; i++)
  {
      tic[i] = Vector3d(para_Ex_Pose[i][0],
			para_Ex_Pose[i][1],
			para_Ex_Pose[i][2]);
      ric[i] = Quaterniond(para_Ex_Pose[i][6],
			    para_Ex_Pose[i][3],
			    para_Ex_Pose[i][4],
			    para_Ex_Pose[i][5]).toRotationMatrix();
  }

  VectorXd dep = f_manager.getDepthVector();
  for (int i = 0; i < f_manager.getFeatureCount(); i++)
      dep(i) = para_Feature[i][0];
  f_manager.setDepth(dep);
}

void Estimator::slideWindow()
{
  if(marginalization_flag == MARGIN_OLD)
  {
    back_R0 = Rs[0];
    back_P0 = Ps[0];
    if(frame_count ==WINDOW_SIZE)
    {
      for(int i = 0;i < WINDOW_SIZE;i++)
      {
	Rs[i].swap(Rs[i+1]);
	Ps[i].swap(Ps[i+1]);
      }
      
      Ps[WINDOW_SIZE] = Ps[WINDOW_SIZE - 1];
      Rs[WINDOW_SIZE] = Rs[WINDOW_SIZE - 1];
      
      //???
//       if (true || solver_flag == INITIAL)
//       {
// 	  double t_0 = Headers[0].stamp.toSec();
// 	  map<double, ImageFrame>::iterator it_0;
// 	  it_0 = all_image_frame.find(t_0);
// 	  delete it_0->second.pre_integration;
// 	  all_image_frame.erase(all_image_frame.begin(), it_0);
//       }
      slideWindowOld();
    }   
  }
  else
  {
    if(frame_count == WINDOW_SIZE)
    {
      Ps[frame_count - 1] = Ps[frame_count];
      Rs[frame_count - 1] = Rs[frame_count];
      
      slideWindowNew();
    }
  }
}

void Estimator::slideWindowNew()
{
  sum_of_front++;
  f_manager.removeFront(frame_count);
}

void Estimator::slideWindowOld()
{
  sum_of_back++;
  
  bool shift_depth = solver_flag == NON_LINEAR ? true : false;
  if(shift_depth)
  {
    Matrix3d R0, R1;
    Vector3d P0, P1;
    R0 = back_R0 * ric[0];
    R1 = Rs[0] * ric[0];
    P0 = back_P0 + back_R0 * tic[0];
    P1 = Ps[0] + Rs[0] * tic[0];
    f_manager.removeBackShiftDepth(R0, P0, R1, P1);
  }
  else
    f_manager.removeBack();
}










