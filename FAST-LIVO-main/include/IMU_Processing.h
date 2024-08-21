
#ifndef IMU_PROCESSING_H
#define IMU_PROCESSING_H
#include <cmath>
#include <math.h>
#include <deque>
#include <mutex>
#include <thread>
#include <fstream>
#include <csignal>
#include <ros/ros.h>
#include <so3_math.h>
#include <Eigen/Eigen>
#include <common_lib.h>
#include <pcl/common/io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <condition_variable>
#include <nav_msgs/Odometry.h>
#include <pcl/common/transforms.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <tf/transform_broadcaster.h>
#include <eigen_conversions/eigen_msg.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <fast_livo/States.h>
#include <geometry_msgs/Vector3.h>

#ifdef USE_IKFOM
#include "use-ikfom.hpp"
#endif

/// *************Preconfiguration

#define MAX_INI_COUNT (200)   //最大初始化次数
//排序
const bool time_list(PointType &x, PointType &y); //{return (x.curvature < y.curvature);};

/// *************IMU Process and undistortion
class ImuProcess
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ImuProcess();
  ~ImuProcess();
  
  void Reset();//重置IMU处理器的状态。
  void Reset(double start_timestamp, const sensor_msgs::ImuConstPtr &lastimu);
  void push_update_state(double offs_t, StatesGroup state); //更新状态
  void set_extrinsic(const V3D &transl, const M3D &rot);//设置外参（激光雷达相对于IMU的位置和姿态）
  void set_gyr_cov_scale(const V3D &scaler);  //设置陀螺仪和加速度计的协方差缩放因子
  void set_acc_cov_scale(const V3D &scaler);  
  void set_gyr_bias_cov(const V3D &b_g);      ////设置陀螺仪和加速度计偏置的协方差
  void set_acc_bias_cov(const V3D &b_a);
  #ifdef USE_IKFOM
  Eigen::Matrix<double, 12, 12> Q;
  //void Process(const MeasureGroup &meas,  esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, PointCloudXYZI::Ptr pcl_un_);
  void Process(const LidarMeasureGroup &lidar_meas,  esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, PointCloudXYZI::Ptr cur_pcl_un_)
  #else
  void Process(const LidarMeasureGroup &lidar_meas, StatesGroup &stat, PointCloudXYZI::Ptr cur_pcl_un_);
  void Process2(LidarMeasureGroup &lidar_meas, StatesGroup &stat, PointCloudXYZI::Ptr cur_pcl_un_);
  void UndistortPcl(LidarMeasureGroup &lidar_meas, StatesGroup &state_inout, PointCloudXYZI &pcl_out);
  #endif

  ros::NodeHandle nh;
  ofstream fout_imu;
  V3D cov_acc;  //加速度计和陀螺仪的协方差
  V3D cov_gyr;
  V3D cov_acc_scale;  //加速度计和陀螺仪的协方差缩放因子
  V3D cov_gyr_scale;
  V3D cov_bias_gyr; //陀螺仪和加速度计偏置的协方差
  V3D cov_bias_acc;
  double first_lidar_time;  //第一帧激光雷达数据的时间戳

 private:
 #ifdef USE_IKFOM
  void IMU_init(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, int &N);  //初始化IMU
  void UndistortPcl(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, PointCloudXYZI &pcl_in_out);
  #else
  void IMU_init(const MeasureGroup &meas, StatesGroup &state, int &N);
  void Forward(const MeasureGroup &meas, StatesGroup &state_inout, double pcl_beg_time, double end_time);//前向处理（状态预测）
  void Backward(const LidarMeasureGroup &lidar_meas, StatesGroup &state_inout, PointCloudXYZI &pcl_out);//后向处理（点云校正）
  #endif

  PointCloudXYZI::Ptr cur_pcl_un_;    // 当前未去畸变的点云
  sensor_msgs::ImuConstPtr last_imu_; // 上一个IMU消息
  deque<sensor_msgs::ImuConstPtr> v_imu_;//IMU消息队列
  vector<Pose6D> IMUpose;             //IMU的姿态序列
  vector<M3D>    v_rot_pcl_;          //点云旋转矩阵的向量
  M3D Lid_rot_to_IMU;                 //LiDAR到IMU的旋转矩阵
  V3D Lid_offset_to_IMU;              //LiDAR到IMU的偏移向量
  V3D mean_acc;                       //平均加速度和角速度
  V3D mean_gyr;
  V3D angvel_last;                    //上一时刻的角速度、加速度等数据
  V3D acc_s_last;
  V3D last_acc;
  V3D last_ang;
  double start_timestamp_;            // 起始时间戳
  double last_lidar_end_time_;        //上一次LiDAR扫描结束的时间
  int    init_iter_num = 1;           //初始化迭代次数
  bool   b_first_frame_ = true;       //标记是否为第一帧数据
  bool   imu_need_init_ = true;       //标记IMU是否需要初始化
};
#endif
