// This file is part of SVO - Semi-direct Visual Odometry.
//
// Copyright (C) 2014 Christian Forster <forster at ifi dot uzh dot ch>
// (Robotics and Perception Group, University of Zurich, Switzerland).
//
// SVO is free software: you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or any later version.
//
// SVO is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include <stdexcept>
#include <frame.h>
#include <feature.h>
#include <point.h>
#include <boost/bind.hpp>
#include <vikit/math_utils.h>
#include <vikit/vision.h>
#include <vikit/performance_monitor.h>
// #include <fast/fast.h>

namespace lidar_selection {

int Frame::frame_counter_ = 0; 

Frame::Frame(vk::AbstractCamera* cam, const cv::Mat& img) :
    id_(frame_counter_++), 
    cam_(cam), 
    key_pts_(5), 
    is_keyframe_(false)
{
  initFrame(img);
}

Frame::~Frame()
{
  std::for_each(fts_.begin(), fts_.end(), [&](FeaturePtr i){i.reset();});
}

void Frame::initFrame(const cv::Mat& img)
{
  // check image
  if(img.empty() || img.type() != CV_8UC1 || img.cols != cam_->width() || img.rows != cam_->height())
    throw std::runtime_error("Frame: provided image has not the same size as the camera model or image is not grayscale");

  // Set keypoints to nullptr
  std::for_each(key_pts_.begin(), key_pts_.end(), [&](FeaturePtr ftr){ ftr=nullptr; });
  
  ImgPyr ().swap(img_pyr_);
  img_pyr_.push_back(img);
  // Build Image Pyramid
  // frame_utils::createImgPyramid(img, max(Config::nPyrLevels(), Config::kltMaxLevel()+1), img_pyr_);
  // frame_utils::createImgPyramid(img, 5, img_pyr_); 
}

void Frame::setKeyframe()
{
  is_keyframe_ = true;  //将当前帧标记为关键帧
  setKeyPoints();
}

void Frame::addFeature(FeaturePtr ftr)
{
  fts_.push_back(ftr);
}

void Frame::setKeyPoints()
{
  for(size_t i = 0; i < 5; ++i)//清理无效的关键点（前5个）
    if(key_pts_[i] != nullptr)
      if(key_pts_[i]->point == nullptr)
        key_pts_[i] = nullptr;
  std::for_each(fts_.begin(), fts_.end(), [&](FeaturePtr ftr){ if(ftr->point != nullptr) checkKeyPoints(ftr); }); //对每个特征点
}

void Frame::checkKeyPoints(FeaturePtr ftr)
{
  const int cu = cam_->width()/2;     //图像中心
  const int cv = cam_->height()/2;

  // center pixel
  if(key_pts_[0] == nullptr)
    key_pts_[0] = ftr;

  else if(std::max(std::fabs(ftr->px[0]-cu), std::fabs(ftr->px[1]-cv))
        < std::max(std::fabs(key_pts_[0]->px[0]-cu), std::fabs(key_pts_[0]->px[1]-cv)))
    key_pts_[0] = ftr;

  // 五个区域
  // key_pts_[0]: 最接近图像中心的点
  // key_pts_[1]: 右下区域
  // key_pts_[2]: 右上区域
  // key_pts_[3]: 左上区域
  // key_pts_[4]: 左下区域

  // (ftr->px[0]-cu) 计算特征点 x 坐标到中心的距离
  // (ftr->px[1]-cv) 计算特征点 y 坐标到中心的距离
  // 这两个距离的乘积代表了点到中心的"极端程度"
  // 比较新旧特征点这个乘积的大小
  // 如果新特征点的乘积更大，说明它更接近角落，因此更新关键点

  if(ftr->px[0] >= cu && ftr->px[1] >= cv)
  {
    if(key_pts_[1] == nullptr)    //如果没有（即 key_pts_[1] 为空），就直接将当前特征点 ftr 设为该区域的关键点。
      key_pts_[1] = ftr;
    else if((ftr->px[0]-cu) * (ftr->px[1]-cv)
          > (key_pts_[1]->px[0]-cu) * (key_pts_[1]->px[1]-cv))        
      key_pts_[1] = ftr;
  } 

  if(ftr->px[0] >= cu && ftr->px[1] < cv)
  {
    if(key_pts_[2] == nullptr)
      key_pts_[2] = ftr;
    // else if((ftr->px[0]-cu) * (ftr->px[1]-cv)
    else if((ftr->px[0]-cu) * (cv-ftr->px[1])
          // > (key_pts_[2]->px[0]-cu) * (key_pts_[2]->px[1]-cv))
          > (key_pts_[2]->px[0]-cu) * (cv-key_pts_[2]->px[1]))
      key_pts_[2] = ftr;
  }

  if(ftr->px[0] < cu && ftr->px[1] < cv)
  {
    if(key_pts_[3] == nullptr)
      key_pts_[3] = ftr;
    else if((ftr->px[0]-cu) * (ftr->px[1]-cv)
          > (key_pts_[3]->px[0]-cu) * (key_pts_[3]->px[1]-cv))
      key_pts_[3] = ftr;
  }

  if(ftr->px[0] < cu && ftr->px[1] >= cv)  
  // if(ftr->px[0] < cv && ftr->px[1] >= cv)
  {
    if(key_pts_[4] == nullptr)
      key_pts_[4] = ftr;

    else if(cu-(ftr->px[0]) * (ftr->px[1]-cv) 
          > (cu-key_pts_[4]->px[0]) * (key_pts_[4]->px[1]-cv))      
      key_pts_[4] = ftr;
  }
}

void Frame::removeKeyPoint(FeaturePtr ftr)
{
  bool found = false;
  std::for_each(key_pts_.begin(), key_pts_.end(), [&](FeaturePtr& i){
    if(i == ftr) {
      i = nullptr;
      found = true;
    }
  });
  if(found)
    setKeyPoints();
}

bool Frame::isVisible(const Vector3d& xyz_w) const
{
  Vector3d xyz_f = T_f_w_*xyz_w;

  if(xyz_f.z() < 0.0)
    return false; // point is behind the camera
  Vector2d px = f2c(xyz_f);

  if(px[0] >= 0.0 && px[1] >= 0.0 && px[0] < cam_->width() && px[1] < cam_->height())
    return true;
  return false;
}

/// Utility functions for the Frame class
namespace frame_utils {

void createImgPyramid(const cv::Mat& img_level_0, int n_levels, ImgPyr& pyr)
{
  pyr.resize(n_levels);
  pyr[0] = img_level_0;

  for(int i=1; i<n_levels; ++i)
  {
    pyr[i] = cv::Mat(pyr[i-1].rows/2, pyr[i-1].cols/2, CV_8U);
    vk::halfSample(pyr[i-1], pyr[i]);
  }
}


bool getSceneDepth(const Frame& frame, double& depth_mean, double& depth_min)
{
  vector<double> depth_vec;
  depth_vec.reserve(frame.fts_.size());
  depth_min = std::numeric_limits<double>::max(); 
  for(auto it=frame.fts_.begin(), ite=frame.fts_.end(); it!=ite; ++it)
  {
    if((*it)->point != nullptr) 
    {
      const double z = frame.w2f((*it)->point->pos_).z();
      depth_vec.push_back(z);
      depth_min = fmin(z, depth_min);
    }
  }
  if(depth_vec.empty())
  {
    cout<<"Cannot set scene depth. Frame has no point-observations!"<<endl;
    return false;
  }
  depth_mean = vk::getMedian(depth_vec);
  return true;
}

} // namespace frame_utils
} // namespace lidar_selection
