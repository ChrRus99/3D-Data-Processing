#include "Registration.h"


struct PointDistance
{ 
  ////////////////////////////////////////////////////////////////////////////////////////////////////
  // This class should include an auto-differentiable cost function. 
  // To rotate a point given an axis-angle rotation, use
  // the Ceres function:
  // AngleAxisRotatePoint(...) (see ceres/rotation.h)
  // Similarly to the Bundle Adjustment case initialize the struct variables with the source and  the target point.
  // You have to optimize only the 6-dimensional array (rx, ry, rz, tx ,ty, tz).
  // WARNING: When dealing with the AutoDiffCostFunction template parameters,
  // pay attention to the order of the template parameters
  ////////////////////////////////////////////////////////////////////////////////////////////////////

  PointDistance(const Eigen::Vector3d& source_point, const Eigen::Vector3d& target_point)
    : source_point{source_point}, target_point(target_point) {}

  template <typename T>
  bool operator()(const T* const transformation, T* residuals) const {
    // Extract the source (data) point
    Eigen::Matrix<T, 3, 1> d;
    d << T(source_point[0]), T(source_point[1]), T(source_point[2]);

    // Extract the target (model) point
    Eigen::Matrix<T, 3, 1> m;
    m << T(target_point[0]), T(target_point[1]), T(target_point[2]);

    // Extract the transformation parameters
    const T* const R = transformation;
    const T* const t = transformation + 3;

    // Apply the R,t transformation to the source point
    Eigen::Matrix<T, 3, 1> d_transf;
    ceres::AngleAxisRotatePoint(R, d.data(), d_transf.data());
    d_transf[0] += t[0];
    d_transf[1] += t[1];
    d_transf[2] += t[2];

    // Compute the residual
    residuals[0] = m.x() - d_transf.x();
    residuals[1] = m.y() - d_transf.y();
    residuals[2] = m.z() - d_transf.z();

    return true;
  }

  static ceres::CostFunction* Create(const Eigen::Vector3d& source_point, const Eigen::Vector3d& target_point) {
    return new ceres::AutoDiffCostFunction<PointDistance, 3, 6>(
      new PointDistance(source_point, target_point)
    );  
  }

  Eigen::Vector3d source_point;
  Eigen::Vector3d target_point;
};


Registration::Registration(std::string cloud_source_filename, std::string cloud_target_filename)
{
  open3d::io::ReadPointCloud(cloud_source_filename, source_ );
  open3d::io::ReadPointCloud(cloud_target_filename, target_ );
  Eigen::Vector3d gray_color;
  source_for_icp_ = source_;
}


Registration::Registration(open3d::geometry::PointCloud cloud_source, open3d::geometry::PointCloud cloud_target)
{
  source_ = cloud_source;
  target_ = cloud_target;
  source_for_icp_ = source_;
}


void Registration::draw_registration_result()
{
  //clone input
  open3d::geometry::PointCloud source_clone = source_;
  open3d::geometry::PointCloud target_clone = target_;

  //different color
  Eigen::Vector3d color_s;
  Eigen::Vector3d color_t;
  color_s<<1, 0.706, 0;
  color_t<<0, 0.651, 0.929;

  target_clone.PaintUniformColor(color_t);
  source_clone.PaintUniformColor(color_s);
  source_clone.Transform(transformation_);

  auto src_pointer =  std::make_shared<open3d::geometry::PointCloud>(source_clone);
  auto target_pointer =  std::make_shared<open3d::geometry::PointCloud>(target_clone);
  open3d::visualization::DrawGeometries({src_pointer, target_pointer});
  return;
}



void Registration::execute_icp_registration(double threshold, int max_iteration, double relative_rmse, std::string mode)
{ 
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //ICP main loop
  //Check convergence criteria and the current iteration.
  //If mode=="svd" use get_svd_icp_transformation if mode=="lm" use get_lm_icp_transformation.
  //Remember to update transformation_ class variable, you can use source_for_icp_ to store transformed 3d points.
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  // RMSE of the previous iteration
  double prev_rmse = 0.0;

  // Iterative Closest Point main loop
  for(int i=0; i<max_iteration; i++) {
    /* Search new correspondences */
    // Associate the points of the two point clouds using the nearest neighbor criteria
    std::tuple<std::vector<size_t>, std::vector<size_t>, double> associated_points_tuple = find_closest_point(threshold);


    /* Check convergence */
    // Extract the current RMSE
    double curr_rmse = std::get<2>(associated_points_tuple);

    // Compute the relative RMSE
    //double curr_relative_rmse = std::abs(curr_rmse - prev_rmse) / prev_rmse;
    double curr_relative_rmse = std::abs(curr_rmse - prev_rmse);

    std::printf("Iteration [%d]: RMSE = %0.8f, relative RMSE = %0.8f\n", i, curr_rmse, curr_relative_rmse);

    // Convergence: Stop
    if(curr_relative_rmse < relative_rmse) {
      std::printf("Convergence at iteration %d!\n", i);
      return;
    }

    // Update the previous RMSE with the current RMSE
    prev_rmse = curr_rmse;


    /* Estimate R,t */
    // Retrieve the cumulated transformation
    Eigen::Matrix4d cumulative_transformation = get_transformation();

    // Estimate the current transformation
    Eigen::Matrix4d curr_transformation;

    if(mode == "svd") {
      curr_transformation = get_svd_icp_transformation(
        std::get<0>(associated_points_tuple),
        std::get<1>(associated_points_tuple)
      );
    } else if(mode == "lm") {
      curr_transformation = get_lm_icp_registration(
        std::get<0>(associated_points_tuple),
        std::get<1>(associated_points_tuple)
      );
    }

    // Update the cumulative transformation with the current transformation (for the next iteration)
    // Note: pre-multiplication because the transformation is defined w.r.t. the (world/fixed) reference frame
    cumulative_transformation = curr_transformation * cumulative_transformation;
    set_transformation(cumulative_transformation);
    

    /* Apply R,t */
    // Transform the points using the new estimated transformation parameters 
    source_for_icp_.Transform(curr_transformation);


    // Repeat: re-associate the points ...
  }

  std::printf("ICP failed to converge!\n");

  return;
}


std::tuple<std::vector<size_t>, std::vector<size_t>, double> Registration::find_closest_point(double threshold)
{ ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //Find source and target indices: for each source point find the closest one in the target and discard if their 
  //distance is bigger than threshold
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  
  std::vector<size_t> target_indices;
  std::vector<size_t> source_indices;
  Eigen::Vector3d source_point;
  double rmse = 0.0;

  // Build a K-D tree from the target point cloud
  open3d::geometry::KDTreeFlann target_kd_tree(target_);

  // Clone the source point cloud
  open3d::geometry::PointCloud source_clone = source_for_icp_;

  // Get the number of point in the source point cloud
  int num_source_points = source_clone.points_.size();
  
  std::vector<int> idx(1);        // index of the closest target point
  std::vector<double> dist2(1);   // distance of the closest target point
  double mse = 0.0;
  
  // For each source point find the closest one in the target
  // and discard if their distance is bigger than threshold
  for(size_t i=0; i < num_source_points; ++i) {
    // Get the i-th point from the source point cloud.
    source_point = source_clone.points_[i];

    // Find the nearest neighbor in the target point cloud using K-D tree.
    target_kd_tree.SearchKNN(source_point, 1, idx, dist2);

    // Discard if the distance is bigger than the threshold
    if(sqrt(dist2[0]) <= threshold) {
      // Store source and target indices
      source_indices.push_back(i);
      target_indices.push_back(idx[0]);

      // Update the MSE
      mse = mse * i/(i+1) + dist2[0]/(i+1);
    }
  }

  // Compute the RMSE
  rmse = sqrt(mse);

  return {source_indices, target_indices, rmse};
}

Eigen::Matrix4d Registration::get_svd_icp_transformation(std::vector<size_t> source_indices, std::vector<size_t> target_indices){
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //Find point clouds centroids and subtract them. 
  //Use SVD (Eigen::JacobiSVD<Eigen::MatrixXd>) to find best rotation and translation matrix.
  //Use source_indices and target_indices to extract point to compute the 3x3 matrix to be decomposed.
  //Remember to manage the special reflection case.
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  Eigen::Matrix4d transformation = Eigen::Matrix4d::Identity(4,4);

  // Clone the source point cloud
  open3d::geometry::PointCloud source_clone = source_for_icp_;

  // Find point cloud centroids
  Eigen::Vector3d source_centroid = std::get<0>(source_clone.ComputeMeanAndCovariance());
  Eigen::Vector3d target_centroid = std::get<0>(target_.ComputeMeanAndCovariance());

  // Compute the 3x3 matrix W
  Eigen::Matrix3d W;

  for(int i=0; i<source_indices.size(); i++) {
    // Subtract the point clouds centroids from the two point clouds
    Eigen::Vector3d d_first = source_clone.points_[source_indices[i]] - source_centroid;
    Eigen::Vector3d m_first = target_.points_[target_indices[i]] - target_centroid;

    // Update the matrix W
    W += m_first * d_first.transpose(); 
  }

  // Compute the SVD of the matrix W
  Eigen::JacobiSVD<Eigen::MatrixXd> W_svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);

  // Compute an optimal solution for the rotation matrix R
  Eigen::Matrix3d R_opt = W_svd.matrixU() * W_svd.matrixV().transpose();

  // Handle the special reflection case (due to corrupted data)
  if(R_opt.determinant() == -1) {
    // Fix the rotation matrix R
    Eigen::DiagonalMatrix<double, 3> diagonal_matrix;
    diagonal_matrix.diagonal() << 1.0, 1.0, -1.0;
    R_opt = W_svd.matrixU() * diagonal_matrix * W_svd.matrixV().transpose();
  }

  // Compute the optimal tranlsation vector t (in closed form)
  Eigen::Vector3d t_opt = target_centroid - R_opt * source_centroid; 

  // Build the optimal transformation matrix [R t]
  transformation.block<3, 3>(0, 0) = R_opt; 
  transformation.block<3, 1>(0, 3) = t_opt; 

  return transformation;
}

Eigen::Matrix4d Registration::get_lm_icp_registration(std::vector<size_t> source_indices, std::vector<size_t> target_indices)
{
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //Use LM (Ceres) to find best rotation and translation matrix. 
  //Remember to convert the euler angles in a rotation matrix, store it coupled with the final translation on:
  //Eigen::Matrix4d transformation.
  //The first three elements of std::vector<double> transformation_arr represent the euler angles, the last ones
  //the translation.
  //use source_indices and target_indices to extract point to compute the matrix to be decomposed.
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  Eigen::Matrix4d transformation = Eigen::Matrix4d::Identity(4,4);
  ceres::Solver::Options options;
  options.minimizer_progress_to_stdout = false;
  options.num_threads = 4;
  options.max_num_iterations = 100;

  std::vector<double> transformation_arr(6, 0.0);
  int num_points = source_indices.size();

  // Clone the source point cloud
  open3d::geometry::PointCloud source_clone = source_for_icp_;

  // Define a Ceres problem
  ceres::Problem problem;
  ceres::Solver::Summary summary;
  
  // For each point....
  for( int i = 0; i < num_points; i++ )
  {
    // Extract the source (data) and the corresponding target (model) point
    Eigen::Vector3d source_point = source_clone.points_[source_indices[i]];
    Eigen::Vector3d target_point = target_.points_[target_indices[i]];

    // Add a residuals block to the Ceres problem
    ceres::CostFunction* cost_function = PointDistance::Create(source_point, target_point);
    problem.AddResidualBlock(
      cost_function,
      nullptr,
      transformation_arr.data()
    );
  }

  // Solve the Ceres problem to compute the optimal rigid body transformation R,t: {rx, ry, rx, tx, ty, tz}
  Solve(options, &problem, &summary);

  // Extract the optimal rotation angles [rx, ry, rx]
  Eigen::AngleAxisd roll(transformation_arr[0], Eigen::Vector3d::UnitX());
  Eigen::AngleAxisd pitch(transformation_arr[1], Eigen::Vector3d::UnitY());
  Eigen::AngleAxisd yaw(transformation_arr[2], Eigen::Vector3d::UnitZ());

  // Compute the optimal rotation matrix R
  Eigen::Matrix3d R_opt;
  R_opt = yaw * pitch * roll;

  // Extract the optimal translation vector t = [tx, ty, tz]
  Eigen::Vector3d t_opt(transformation_arr[3], transformation_arr[4], transformation_arr[5]);

  // Build the optimal transformation matrix [R t]
  transformation.block<3, 3>(0, 0) = R_opt; 
  transformation.block<3, 1>(0, 3) = t_opt; 

  return transformation;
}


void Registration::set_transformation(Eigen::Matrix4d init_transformation)
{
  transformation_=init_transformation;
}


Eigen::Matrix4d  Registration::get_transformation()
{
  return transformation_;
}

double Registration::compute_rmse()
{
  open3d::geometry::KDTreeFlann target_kd_tree(target_);
  open3d::geometry::PointCloud source_clone = source_;
  source_clone.Transform(transformation_);
  int num_source_points  = source_clone.points_.size();
  Eigen::Vector3d source_point;
  std::vector<int> idx(1);
  std::vector<double> dist2(1);
  double mse;
  for(size_t i=0; i < num_source_points; ++i) {
    source_point = source_clone.points_[i];
    target_kd_tree.SearchKNN(source_point, 1, idx, dist2);
    mse = mse * i/(i+1) + dist2[0]/(i+1);
  }
  return sqrt(mse);
}

void Registration::write_tranformation_matrix(std::string filename)
{
  std::ofstream outfile (filename);
  if (outfile.is_open())
  {
    outfile << transformation_;
    outfile.close();
  }
}

void Registration::save_merged_cloud(std::string filename)
{
  //clone input
  open3d::geometry::PointCloud source_clone = source_;
  open3d::geometry::PointCloud target_clone = target_;

  source_clone.Transform(transformation_);
  open3d::geometry::PointCloud merged = target_clone+source_clone;
  open3d::io::WritePointCloud(filename, merged );
}


