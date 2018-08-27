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
  solver_flag = NON_LINEAR;
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
  
  if(frame_count == WINDOW_SIZE)
  {
    if(ESTIMATE_EXTRINSIC != 2)
    {
      solverOdometry(); 
      slideWindow();
      f_manager.removeFailures();
      
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










