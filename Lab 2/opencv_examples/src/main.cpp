#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

void drawPoints( const std::vector<cv::Point2f> &img_pts, cv::Mat &img, 
                 const cv::Scalar &color = cv::Scalar(255,255,255) )
{
  for( auto &p : img_pts )
  {
    cv::line(img, p, p, color, 5);
  }
}

float frand()
{
  return static_cast <float> (rand()) / static_cast <float> (RAND_MAX) - 0.5f;
}

void perturbatePoints( std::vector<cv::Point2f> &img_pts, float max_noise = 1.0 )
{
  for( auto &p : img_pts )
  {
    p.x += max_noise*frand();
    p.y += max_noise*frand();
  }
}

void showImage( const cv::Mat &img )
{
  cv::imshow("Display image", img);
  std::cout<<"Type \"ESC\" to exit\n";
  while(cv::waitKey() != 27);
}

int main(int argc, char** argv)
{
  std::vector<cv::Point3f> scene_pts;

  // Prepare an "object" defined by a set of co-planar points   
  scene_pts.emplace_back(-0.5, -0.5,  0);
  scene_pts.emplace_back( 0,   -0.5,  0);
  scene_pts.emplace_back( 0.5, -0.5,  0);
  scene_pts.emplace_back(-0.5,  0,    0);
  scene_pts.emplace_back( 0,    0,    0);
  scene_pts.emplace_back( 0.5,  0,    0);
  scene_pts.emplace_back(-0.5,  0.5,  0);
  scene_pts.emplace_back( 0,    0.5,  0);
  scene_pts.emplace_back( 0.5,  0.5,  0);
  
  cv::Size img_size(1024, 768);
  cv::Mat_<cv::Vec3b> display_img = cv::Mat_<cv::Vec3b>::zeros(img_size);
  
  // Some camera parameters (no distortion params, we assume an undistorted camera)
  double fx = 9.6785787400000004e+02, fy = 9.6344770300000005e+02,
         cx = 5.5825267699999995e+02, cy = 4.1089457800000002e+02;
         
  // Linear intrinsic paramters matrix
  cv::Mat_<double> K = (cv::Mat_<double>(3,3) << fx,   0,   cx, 
                                                 0,    fy,  cy,
                                                 0,    0,   1);
 
  // Define some object->camera transfomration (rotation in axis-angle representation)
  cv::Mat_<double> r_vec_0 = (cv::Mat_<double>(3,1) << M_PI/6, M_PI/6, M_PI/6),
                   t_vec_0 = (cv::Mat_<double>(3,1) << 0,0, 3.0), 
                   r_mat_0;
  
  // Convert the rotation in axis-angle representation to a 3x3 rotation matrix
  cv::Rodrigues (r_vec_0, r_mat_0);
  
  // Project points into the image plane
  std::vector<cv::Point2f> img_pts_0, img_pts_1;
  cv::projectPoints (scene_pts, r_mat_0, t_vec_0, K, cv::Mat(), img_pts_0);
  
  // Display the projected points in green
  drawPoints( img_pts_0, display_img, cv::Scalar(0,255,0) );
  showImage( display_img );
  
  // Add some gaussian noise and redraw in red
  perturbatePoints( img_pts_0, 1.0 );
  drawPoints( img_pts_0, display_img, cv::Scalar(0,0,255) );
  showImage( display_img );  
  

  cv::Mat inlier_mask_H;
  // Recovere with RANSAC the homography matrix (3.0 is the RANSAC threshold to define an inlier)
  // inlier mask H is a vector with dimension equal to the number of correspondences: 
  // each element is set to a value greater than zero only if the corresponding correspondence represents an inlier
  cv::Mat H = cv::findHomography(scene_pts, img_pts_0, cv::RANSAC, 3.0, inlier_mask_H);
  
  // Count the inliers for the homography model
  int n_inliers_H = 0;
  for(int r = 0; r < inlier_mask_H.rows; r++)
    if(inlier_mask_H.at<unsigned char>(r))
      n_inliers_H++;
  
  
  std::cout<<"Num inliers H : "<<n_inliers_H<<std::endl;
  
  // Define e relative camera motion
  cv::Mat_<double> rel_r_vec = (cv::Mat_<double>(3,1) << -M_PI/24, -M_PI/24, -M_PI/24),
                   rel_t_vec = (cv::Mat_<double>(3,1) << 0.2, 0, 0), rel_r_mat, r_mat_1, t_vec_1;
  
  // Convert the rotation in axis-angle representation to a 3x3 rotation matrix
  cv::Rodrigues (rel_r_vec, rel_r_mat);
  
  // Compute the object->camera transfomration fro the second view
  r_mat_1 = rel_r_mat*r_mat_0;
  t_vec_1 = rel_r_mat*t_vec_0 + rel_t_vec;

  // Project the 3D ponts in the second view
  cv::projectPoints (scene_pts, r_mat_1, t_vec_1, K, /*No dist coeff*/ cv::Mat(), img_pts_1);

  // Display the projected points in blue
  drawPoints( img_pts_1, display_img, cv::Scalar(255,0,0) );
  showImage( display_img );
  
  // Add some gaussian noise also in the second view
  perturbatePoints( img_pts_1, 1.0 );

  // Recovere with RANSAC the homography matrix between the two views
  H = cv::findHomography(img_pts_0, img_pts_1, cv::RANSAC, 3.0, inlier_mask_H);
  
  std::vector<cv::Mat> rotations, translations, normals;

  // Extracts the relative camera motion between the two views
  // The decomposeHomographyMat() function may return up to four mathematical solution sets. 
  cv::decomposeHomographyMat(H, K, rotations, translations, normals);

  // Print the "true" relative transfomration (i.e., the ground truth)
  std::cout<<"gt_r : "<<std::endl<<rel_r_vec<<std::endl;
  std::cout<<"gt_t : "<<std::endl<<rel_t_vec<<std::endl;

  // Print all the extracted solution: one should be quite close to the ground truth, up to a scalar factor
  for( int i = 0; i < rotations.size(); i++ )
  {
    cv::Mat rvec_decomp;
    cv::Rodrigues(rotations[i], rvec_decomp);
    std::cout<<"r["<<i<<"] : "<<std::endl<<rvec_decomp<<std::endl;
    std::cout<<"t["<<i<<"] : "<<std::endl<<translations[i]<<std::endl;
  }
  
  // Reset the display
  display_img = cv::Mat_<cv::Vec3b>::zeros(img_size);

  scene_pts.clear();

  // Prepare a new "object" defined by a set of non co-planar points
  // (the 8 corners of a cube)
  scene_pts.emplace_back(-0.5, -0.5,  -0.5);
  scene_pts.emplace_back(-0.5, -0.5,   0.5);
  scene_pts.emplace_back(-0.5,  0.5,  -0.5);
  scene_pts.emplace_back(-0.5,  0.5,  0.5);
  scene_pts.emplace_back( 0.5, -0.5,  -0.5);
  scene_pts.emplace_back( 0.5, -0.5,   0.5);
  scene_pts.emplace_back( 0.5,  0.5,  -0.5);
  scene_pts.emplace_back( 0.5,  0.5,  0.5);
  
  // Project the points into the image plane of the first view
  cv::projectPoints (scene_pts, r_mat_0, t_vec_0, K, /*No dist coeff*/ cv::Mat(), img_pts_0);
  
  // Display the projected points in green
  drawPoints( img_pts_0, display_img, cv::Scalar(0,255,0) );
  showImage( display_img );
  
  // Add some gaussian noise and redraw in red
  perturbatePoints( img_pts_0, 1.0 );
  drawPoints( img_pts_0, display_img, cv::Scalar(0,0,255) );
  showImage( display_img );
  
  // Use the PnP + RANSAC algorithms to estimate the transformation given the set of 3D-2D correspondences
  cv::Mat est_r_vec, est_r_mat, est_t_vec;
  cv::solvePnPRansac(scene_pts, img_pts_0, K, cv::Mat(), est_r_vec, est_t_vec);
  
  // Print both the ground truth and the estimated transformations: they should be quite similar
  std::cout<<"gt_r : "<<std::endl<<r_vec_0<<std::endl;
  std::cout<<"gt_t : "<<std::endl<<t_vec_0<<std::endl;
  std::cout<<"est_r_vec : "<<std::endl<<est_r_vec<<std::endl;
  std::cout<<"est_t_vec : "<<std::endl<<est_t_vec<<std::endl; 
  
  // Project the points into the image plane of the second view
  cv::projectPoints (scene_pts, r_mat_1, t_vec_1, K, /*No dist coeff*/ cv::Mat(), img_pts_1);
  
  // Display the projected points in blue
  drawPoints( img_pts_1, display_img, cv::Scalar(255,0,0) );
  showImage( display_img );
  
  // Add some gaussian noise
  perturbatePoints( img_pts_1, 1.0 );
  
  // "Normalize" the points (i.e., represents their coordinates in the canonical camera)
  std::vector<cv::Point2f> und_img_pts_0, und_img_pts_1;
  cv::undistortPoints(img_pts_0, und_img_pts_0, K, cv::Mat());
  cv::undistortPoints(img_pts_1, und_img_pts_1, K, cv::Mat());
  
  // Estimate the essential matrix between the two views, along with the corresponding inlier mask
  cv::Mat inlier_mask_E;
  cv::Mat E = cv::findEssentialMat(und_img_pts_0, und_img_pts_1, cv::Mat_<double>::eye(3,3), cv::RANSAC, 0.995, 0.01, inlier_mask_E);

  // Count the inliers for the essential model
  int n_inliers_E = 0;
  for(int r = 0; r < inlier_mask_E.rows; r++)
    if(inlier_mask_E.at<unsigned char>(r))
      n_inliers_E++;  
  
  std::cout<<"Num inliers E : "<<n_inliers_E<<std::endl;
  
  // Recover relative camera rotation and translation from an estimated essential matrix, by usinf the
  // inlier mask as input/output parameter
  cv::recoverPose(E, und_img_pts_0, und_img_pts_1, K, est_r_mat, est_t_vec, inlier_mask_E);
  
  // Print both the ground truth and the estimated transformations: they should be quite similar, up to a scalar factor
  // although often the estimate is not so good due to poins noise and the small number of points
  cv::Rodrigues(est_r_mat, est_r_vec);
  std::cout<<"gt_r : "<<std::endl<<rel_r_vec<<std::endl;
  std::cout<<"gt_t : "<<std::endl<<rel_t_vec<<std::endl;
  std::cout<<"est_r_vec : "<<std::endl<<est_r_vec<<std::endl;
  std::cout<<"est_t_vec : "<<std::endl<<est_t_vec<<std::endl; 
  
  return 0;
}
