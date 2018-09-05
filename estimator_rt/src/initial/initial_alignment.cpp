#include "initial_alignment.h"

bool LinearAlignment(map<double, ImageFrame> &all_image_frame, double &s, Vector3d &tic)
{
    int all_frame_count = all_image_frame.size();

    cv::Mat A = cv::Mat::zeros(all_frame_count * 3, 4, CV_64F);
    cv::Mat B = cv::Mat::zeros(all_frame_count * 3, 1, CV_64F);

    map<double, ImageFrame>::iterator frame_i;
    int i = 0;
    for (frame_i = all_image_frame.begin(); frame_i != all_image_frame.end(); frame_i++, i++)
    {
        MatrixXd tmp_A(3, 4);
        tmp_A.setZero();
        Vector3d tmp_b;
        tmp_b.setZero();

        tmp_A.block<3, 1>(0, 0) = frame_i->second.T;
        tmp_A.block<3, 3>(0, 1) = frame_i->second.imu_R;  
        tmp_b = frame_i->second.imu_T;
	
	cv::Mat lambda = cv::Mat::zeros(4, 4, CV_64F);
	for(int i = 0;i < 4;i++)
	  for(int j = 0;j < 4;j++)
	    lambda.at<double>(i,j) = tmp_A(i,j);
	  
	cv::Mat gamma = cv::Mat::zeros(3, 1, CV_64F);
	for(int i = 0;i < 3;i++)
	  gamma.at<double>(i) = tmp_b(i);
	
	lambda.copyTo(A.rowRange(3 * i, 3 * i + 3).colRange(0, 4));
	gamma.copyTo(B.rowRange(3 *i, 3 * i + 3).col(0));
    }
    
    cv::Mat w, u, vt;
    
    //Note w is 4x1 vector by SVDecomp
    cv::SVDecomp(A, w, u, vt, cv::SVD::MODIFY_A);
    
    //Compute winv
    cv::Mat winv = cv::Mat::eye(4, 4, CV_64F);
    for(int i = 0;i < 4;i++)
    {
      if(fabs(w.at<double>(i)) < 1e-10)
      {
	w.at<double>(i) += 1e-10;
      }
      winv.at<double>(i, i) = 1./w.at<double>(i);
    }
    //Then x = vt'*winv*u'*B
    cv::Mat x = vt.t() * winv * u.t() * B;
    
    //x = [s, tic]
    s = x.at<double>(0);
    
    cv::Mat tic_tmp = x.rowRange(1,4);
    for(int i = 0;i < 3;i++)
      tic(i) = tic_tmp.at<double>(i);
    
    if(s < 0.0 )
        return false;   
    else
        return true;
}

bool VisualIMUAlignment(map<double, ImageFrame> &all_image_frame, double &s, Vector3d &tic)
{
    if(LinearAlignment(all_image_frame, s, tic))
        return true;
    else 
        return false;
}
