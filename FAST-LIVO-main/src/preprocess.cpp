#include "preprocess.h"

#define RETURN0     0x00
#define RETURN0AND1 0x10

Preprocess::Preprocess()
  :feature_enabled(0), lidar_type(AVIA), blind(0.01), point_filter_num(1)
{
  inf_bound = 10;
  N_SCANS   = 6;
  group_size = 8;
  disA = 0.01;
  disB = 0.1; // B?
  p2l_ratio = 225;
  limit_maxmid =6.25;
  limit_midmin =6.25;
  limit_maxmin = 3.24;
  jump_up_limit = 170.0;
  jump_down_limit = 8.0;
  cos160 = 160.0;
  edgea = 2;
  edgeb = 0.1;
  smallp_intersect = 172.5;
  smallp_ratio = 1.2;
  given_offset_time = false;

  jump_up_limit = cos(jump_up_limit/180*M_PI);
  jump_down_limit = cos(jump_down_limit/180*M_PI);
  cos160 = cos(cos160/180*M_PI);
  smallp_intersect = cos(smallp_intersect/180*M_PI);
}

Preprocess::~Preprocess() {}

void Preprocess::set(bool feat_en, int lid_type, double bld, int pfilt_num)
{
  feature_enabled = feat_en;//开启特征提取
  lidar_type = lid_type;    //雷达信号
  blind = bld;              //盲区
  point_filter_num = pfilt_num;
}

void Preprocess::process(const livox_ros_driver::CustomMsg::ConstPtr &msg, PointCloudXYZI::Ptr &pcl_out)  
{  
  avia_handler(msg);  //livox avia 处理函数 CustomMsg -> PointCloudXYZI
  *pcl_out = pl_surf;
}

void Preprocess::process(const sensor_msgs::PointCloud2::ConstPtr &msg, PointCloudXYZI::Ptr &pcl_out)
{
  //根据雷达类型来处理点云
  switch (lidar_type)
  {
  case OUST64:
    oust64_handler(msg);
    break;

  case VELO16:
    velodyne_handler(msg);
    break;

  case XT32:
    xt32_handler(msg);
    break;
  
  default:
    printf("Error LiDAR Type");
    break;
  }
  *pcl_out = pl_surf;
}


void Preprocess::avia_handler(const livox_ros_driver::CustomMsg::ConstPtr &msg)
{
  pl_surf.clear();
  pl_corn.clear();
  pl_full.clear();
  double t1 = omp_get_wtime();
  uint plsize = msg->point_num;
  uint effect_ind = 0;

  //内存预分配
  pl_corn.reserve(plsize);
  pl_surf.reserve(plsize);
  pl_full.resize(plsize);

  for(int i=0; i<N_SCANS; i++)
  {
    pl_buff[i].clear();
    pl_buff[i].reserve(plsize);
  }

  // 当功能启用时执行以下操作
  if (feature_enabled)
  {
      // 遍历点云数据，处理每个点
      for(uint i=1; i<plsize; i++)
      {
          // 如果当前点与前一个点的x、y、z坐标差值小于指定阈值，
          // 或者点的x和y坐标平方和小于盲区阈值，
          // 或者点的line属性大于最大扫描线数，
          // 或者点的tag属性与指定值不匹配，则跳过该点
          if((abs(msg->points[i].x - msg->points[i-1].x) < 1e-8) 
              || (abs(msg->points[i].y - msg->points[i-1].y) < 1e-8)
              || (abs(msg->points[i].z - msg->points[i-1].z) < 1e-8)
              || (msg->points[i].x * msg->points[i].x + msg->points[i].y * msg->points[i].y < blind)
              || (msg->points[i].line > N_SCANS)
              || ((msg->points[i].tag & 0x30) != RETURN0AND1))
          {
              continue;
          }

          // 否则，将点的x、y、z和intensity值复制到pl_full数组中
          pl_full[i].x = msg->points[i].x;
          pl_full[i].y = msg->points[i].y;
          pl_full[i].z = msg->points[i].z;
          pl_full[i].intensity = msg->points[i].reflectivity;
          pl_full[i].curvature = msg->points[i].offset_time / float(1000000); //使用曲率作为每个激光点的时间 将时间转换为秒
          pl_buff[msg->points[i].line].push_back(pl_full[i]);
      }

      // 对每个扫描线的点云数据进行处理
      for(int j=0; j<N_SCANS; j++)
      {
        // 如果当前扫描线的点数量小于等于5，则跳过该扫描线
        if(pl_buff[j].size() <= 5) continue;
        // 获取当前扫描线的点云数据引用
        pcl::PointCloud<PointType> &pl = pl_buff[j];
        // 获取当前点云数据的大小
        plsize = pl.size();
        // 获取当前扫描线的类型向量引用，并清空和重新设置其大小
        vector<orgtype> &types = typess[j];
        types.clear();
        types.resize(plsize);
        // 减少一次循环次数，因为最后一个点不需要计算
        plsize--;
        // 遍历点云数据，计算每个点的range和dista值
        for(uint i=0; i<plsize; i++)
        {
          types[i].range = pl[i].x * pl[i].x + pl[i].y * pl[i].y;
          vx = pl[i].x - pl[i + 1].x;
          vy = pl[i].y - pl[i + 1].y;
          vz = pl[i].z - pl[i + 1].z;
          types[i].dista = vx * vx + vy * vy + vz * vz;
        }
        // 计算最后一个点的range值
        types[plsize].range = pl[plsize].x * pl[plsize].x + pl[plsize].y * pl[plsize].y;
        // 将计算好的类型数据传递给give_feature函数进行进一步处理
        give_feature(pl, types);
      }
  }
  else
  {
    // 遍历除第一个点之外的所有点
    for(uint i=1; i<plsize; i++)
    {
        // 检查各种条件以过滤掉不需要的点
        if((abs(msg->points[i].x - msg->points[i-1].x) < 1e-8) 
            || (abs(msg->points[i].y - msg->points[i-1].y) < 1e-8)
            || (abs(msg->points[i].z - msg->points[i-1].z) < 1e-8)// 过滤掉与前一个点距离太近的点（差异小于1e-8）
            || (msg->points[i].x * msg->points[i].x + msg->points[i].y * msg->points[i].y < blind)// 过滤掉盲区里的点（在'blind'半径内）
            || (msg->points[i].line > N_SCANS)// 过滤掉超出N_SCANS的扫描线上的点
            || ((msg->points[i].tag & 0x30) != RETURN0AND1)) // 过滤掉不是第一次或第二次回波的点
        {
            continue;
        }

        effect_ind ++;// 增加有效点计数器
        // 每隔'point_filter_num'个点处理一次 相当于降采样
        if(effect_ind % point_filter_num == 0)
        {
            // 将点数据复制到pl_full
            pl_full[i].x = msg->points[i].x;
            pl_full[i].y = msg->points[i].y;
            pl_full[i].z = msg->points[i].z;
            pl_full[i].intensity = msg->points[i].reflectivity;
            pl_full[i].curvature = msg->points[i].offset_time / float(1000000); // 将时间转换为秒 并存储在curvature中
            pl_surf.push_back(pl_full[i]);// 将点添加到pl_surf中
        }
    }
  }
  // printf("feature extraction time: %lf \n", omp_get_wtime()-t1);
}

void Preprocess::oust64_handler(const sensor_msgs::PointCloud2::ConstPtr &msg)
{
  pl_surf.clear();
  pl_corn.clear();
  pl_full.clear();
  pcl::PointCloud<ouster_ros::Point> pl_orig;
  pcl::fromROSMsg(*msg, pl_orig);
  int plsize = pl_orig.size();
  pl_corn.reserve(plsize);
  pl_surf.reserve(plsize);
  if (feature_enabled)
  {
    for (int i = 0; i < N_SCANS; i++)
    {
      pl_buff[i].clear();
      pl_buff[i].reserve(plsize);
    }

    for (uint i = 0; i < plsize; i++)
    {
      double range = pl_orig.points[i].x * pl_orig.points[i].x + pl_orig.points[i].y * pl_orig.points[i].y + pl_orig.points[i].z * pl_orig.points[i].z;
      if (range < (blind * blind)) continue;
      Eigen::Vector3d pt_vec;
      PointType added_pt;
      added_pt.x = pl_orig.points[i].x;
      added_pt.y = pl_orig.points[i].y;
      added_pt.z = pl_orig.points[i].z;
      added_pt.intensity = pl_orig.points[i].intensity;
      added_pt.normal_x = 0;
      added_pt.normal_y = 0;
      added_pt.normal_z = 0;
      double yaw_angle = atan2(added_pt.y, added_pt.x) * 57.3;
      if (yaw_angle >= 180.0)
        yaw_angle -= 360.0;
      if (yaw_angle <= -180.0)
        yaw_angle += 360.0;

      added_pt.curvature = pl_orig.points[i].t * 1.e-6f;
      if(pl_orig.points[i].ring < N_SCANS)
      {
        pl_buff[pl_orig.points[i].ring].push_back(added_pt);
      }
    }

    for (int j = 0; j < N_SCANS; j++)
    {
      PointCloudXYZI &pl = pl_buff[j];
      int linesize = pl.size();
      vector<orgtype> &types = typess[j];
      types.clear();
      types.resize(linesize);
      linesize--;
      for (uint i = 0; i < linesize; i++)
      {
        types[i].range = sqrt(pl[i].x * pl[i].x + pl[i].y * pl[i].y);
        vx = pl[i].x - pl[i + 1].x;
        vy = pl[i].y - pl[i + 1].y;
        vz = pl[i].z - pl[i + 1].z;
        types[i].dista = vx * vx + vy * vy + vz * vz;
      }
      types[linesize].range = sqrt(pl[linesize].x * pl[linesize].x + pl[linesize].y * pl[linesize].y);
      give_feature(pl, types);
    }
  }
  else
  {
    double time_stamp = msg->header.stamp.toSec();  //换算成秒
    // 遍历原始点云中的所有点
    for (int i = 0; i < pl_orig.points.size(); i++)
    {
      // 根据 point_filter_num 进行点云降采样，只处理每 point_filter_num 个点中的一个
      if (i % point_filter_num != 0) continue;
      // 计算点到原点的距离（范围）
      double range = pl_orig.points[i].x * pl_orig.points[i].x + pl_orig.points[i].y * pl_orig.points[i].y + pl_orig.points[i].z * pl_orig.points[i].z;
      
      // 如果点距离小于 blind 阈值，则跳过该点（去除盲区的点）
      if (range < (blind * blind)) continue;

      // 创建一个新的点，准备添加到 pl_surf 点云中
      Eigen::Vector3d pt_vec;
      PointType added_pt;

      // 复制原始点的 x, y, z 坐标和强度值
      added_pt.x = pl_orig.points[i].x;
      added_pt.y = pl_orig.points[i].y;
      added_pt.z = pl_orig.points[i].z;
      added_pt.intensity = pl_orig.points[i].intensity;
      // 初始化法向量为
      added_pt.normal_x = 0;
      added_pt.normal_y = 0;
      added_pt.normal_z = 0;
      // 将原始点的时间信息转换为毫秒单位，存储在曲率字段中
      added_pt.curvature = pl_orig.points[i].t * 1.e-6f; // curvature unit: ms
      // 将处理后的点添加到 pl_surf 点云中
      pl_surf.points.push_back(added_pt);
    }
  }
  // pub_func(pl_surf, pub_full, msg->header.stamp);
  // pub_func(pl_surf, pub_corn, msg->header.stamp);
}

void Preprocess::velodyne_handler(const sensor_msgs::PointCloud2::ConstPtr &msg)
{
    pl_surf.clear();
    pl_corn.clear();
    pl_full.clear();

    pcl::PointCloud<velodyne_ros::Point> pl_orig;
    pcl::fromROSMsg(*msg, pl_orig);
    int plsize = pl_orig.points.size();
    if (plsize == 0) return;
    pl_surf.reserve(plsize);

    /*** These variables only works when no point timestamps given ***/
    double omega_l = 0.361 * 10;       // scan angular velocity 10Hz
    std::vector<bool> is_first(N_SCANS,true);
    std::vector<double> yaw_fp(N_SCANS, 0.0);      // yaw of first scan point
    std::vector<float> yaw_last(N_SCANS, 0.0);   // yaw of last scan point
    std::vector<float> time_last(N_SCANS, 0.0);  // last offset time
    /*****************************************************************/
    //通过计算同一层第一个点和最后一个点的偏航角，可以估算出一次扫描的角度范围
    // 检查最后一个点的时间是否大于0
    if (pl_orig.points[plsize - 1].time > 0)
    {
      given_offset_time = true;// 如果是，说明已经给出了偏移时间
    }
    else
    {
      given_offset_time = false; // 如果不是，说明没有给出偏移时间

      // 计算第一个点的偏航角（yaw），并转换为度数
      double yaw_first = atan2(pl_orig.points[0].y, pl_orig.points[0].x) * 57.29578;
      double yaw_end  = yaw_first;// 初始化结束偏航角为起始偏航角

      int layer_first = pl_orig.points[0].ring; // 获取第一个点的扫描层
      // 从最后一个点开始向前遍历
      for (uint i = plsize - 1; i > 0; i--)
      {
        // 找到与第一个点在同一扫描层的最后一个点
        if (pl_orig.points[i].ring == layer_first)
        {
          // 计算该点的偏航角，并转换为度数
          yaw_end = atan2(pl_orig.points[i].y, pl_orig.points[i].x) * 57.29578;
          break;
        }
      }
    }

    if(feature_enabled)
    {
      for (int i = 0; i < N_SCANS; i++)
      {
        pl_buff[i].clear();
        pl_buff[i].reserve(plsize);
      }
      
      for (int i = 0; i < plsize; i++)
      {
        PointType added_pt;
        added_pt.normal_x = 0;
        added_pt.normal_y = 0;
        added_pt.normal_z = 0;
        int layer  = pl_orig.points[i].ring;
        if (layer >= N_SCANS) continue;
        added_pt.x = pl_orig.points[i].x;
        added_pt.y = pl_orig.points[i].y;
        added_pt.z = pl_orig.points[i].z;
        added_pt.intensity = pl_orig.points[i].intensity;
        added_pt.curvature = pl_orig.points[i].time * 1.e-3f; // units: ms

        if (!given_offset_time)
        {
          double yaw_angle = atan2(added_pt.y, added_pt.x) * 57.2957;
          if (is_first[layer])
          {
            // printf("layer: %d; is first: %d", layer, is_first[layer]);
              yaw_fp[layer]=yaw_angle;
              is_first[layer]=false;
              added_pt.curvature = 0.0;
              yaw_last[layer]=yaw_angle;
              time_last[layer]=added_pt.curvature;
              continue;
          }

          if (yaw_angle <= yaw_fp[layer])
          {
            added_pt.curvature = (yaw_fp[layer]-yaw_angle) / omega_l;
          }
          else
          {
            added_pt.curvature = (yaw_fp[layer]-yaw_angle+360.0) / omega_l;
          }

          if (added_pt.curvature < time_last[layer])  added_pt.curvature+=360.0/omega_l;

          yaw_last[layer] = yaw_angle;
          time_last[layer]=added_pt.curvature;
        }

        pl_buff[layer].points.push_back(added_pt);
      }

      for (int j = 0; j < N_SCANS; j++)
      {
        PointCloudXYZI &pl = pl_buff[j];
        int linesize = pl.size();
        if (linesize < 2) continue;
        vector<orgtype> &types = typess[j];
        types.clear();
        types.resize(linesize);
        linesize--;
        for (uint i = 0; i < linesize; i++)
        {
          types[i].range = sqrt(pl[i].x * pl[i].x + pl[i].y * pl[i].y);
          vx = pl[i].x - pl[i + 1].x;
          vy = pl[i].y - pl[i + 1].y;
          vz = pl[i].z - pl[i + 1].z;
          types[i].dista = vx * vx + vy * vy + vz * vz;
        }
        types[linesize].range = sqrt(pl[linesize].x * pl[linesize].x + pl[linesize].y * pl[linesize].y);
        give_feature(pl, types);
      }
    }
    else
    {
      for (int i = 0; i < plsize; i++)
      {
        PointType added_pt;
        // 初始化法向量为0
        added_pt.normal_x = 0;
        added_pt.normal_y = 0;
        added_pt.normal_z = 0;
        // 复制原始点的坐标和强度
        added_pt.x = pl_orig.points[i].x;
        added_pt.y = pl_orig.points[i].y;
        added_pt.z = pl_orig.points[i].z;
        added_pt.intensity = pl_orig.points[i].intensity;
        // 将时间信息转换为毫秒并存储在曲率字段中
        added_pt.curvature = pl_orig.points[i].time * 1.e-3f;  // curvature unit: ms // 

        if (!given_offset_time)
        {
          int layer = pl_orig.points[i].ring;
          // 计算点的偏航角（度数）
          double yaw_angle = atan2(added_pt.y, added_pt.x) * 57.2957;

          if (is_first[layer]) // 如果是该层的第一个点
          {
            // printf("layer: %d; is first: %d", layer, is_first[layer]);
            
              yaw_fp[layer]=yaw_angle;
              is_first[layer]=false;
              added_pt.curvature = 0.0;
              yaw_last[layer]=yaw_angle;
              time_last[layer]=added_pt.curvature;
              continue;
          }

          // 计算偏移时间
          if (yaw_angle <= yaw_fp[layer])
          {
            added_pt.curvature = (yaw_fp[layer]-yaw_angle) / omega_l;
          }
          else
          {
            added_pt.curvature = (yaw_fp[layer]-yaw_angle+360.0) / omega_l;
          }
          // 处理跨越360度边界的情况
          if (added_pt.curvature < time_last[layer])  added_pt.curvature+=360.0/omega_l;

          yaw_last[layer] = yaw_angle;
          time_last[layer]=added_pt.curvature;
        }
        // 对点进行降采样和距离过滤
        if (i % point_filter_num == 0)
        {
          if(added_pt.x*added_pt.x+added_pt.y*added_pt.y+added_pt.z*added_pt.z > (blind * blind))//盲区
          {
            pl_surf.points.push_back(added_pt);
          }
        }
      }
    }
}

void Preprocess::xt32_handler(const sensor_msgs::PointCloud2::ConstPtr &msg)
{
  pl_surf.clear();

  pcl::PointCloud<xt32_ros::Point> pl_orig;
  pcl::fromROSMsg(*msg, pl_orig);
  int plsize = pl_orig.points.size();
  pl_surf.reserve(plsize);// 预分配内存以提高效率

  double time_head = pl_orig.points[0].timestamp;// 获取第一个点的时间戳作为基准时间

  for (int i = 0; i < plsize; i++)
  {
    PointType added_pt;

    added_pt.normal_x = 0;
    added_pt.normal_y = 0;
    added_pt.normal_z = 0;
    // 复制原始点的坐标和强度
    added_pt.x = pl_orig.points[i].x;
    added_pt.y = pl_orig.points[i].y;
    added_pt.z = pl_orig.points[i].z;
    added_pt.intensity = pl_orig.points[i].intensity;
    // 计算相对于第一个点的时间偏移（转换为毫秒）
    added_pt.curvature = (pl_orig.points[i].timestamp - time_head) * 1000.f;

    //降采样
    if (i % point_filter_num == 0)
    {
      //盲区
      if (added_pt.x * added_pt.x + added_pt.y * added_pt.y + added_pt.z * added_pt.z > blind)
      {
        pl_surf.points.push_back(added_pt);
      }
    }
  }
}

// give_feature函数，用于提取点云特征
void Preprocess::give_feature(pcl::PointCloud<PointType> &pl, vector<orgtype> &types)
{
  // 获取点云大小
  uint plsize = pl.size();
  // 定义变量plsize2，用于后续循环中调整点云大小
  uint plsize2;
  // 若点云为空，则打印错误信息并返回
  if(plsize == 0)
  {
    printf("something wrong\n");
    return;
  }
  // 初始化head变量，用于从点云的特定位置开始处理
  uint head = 0;

  // 循环直到找到第一个距离大于blind的点
  while(types[head].range < blind)
  {
    head++;
  }

  // 处理点云中的每个点，判断并标记平面类型
  plsize2 = (plsize > group_size) ? (plsize - group_size) : 0;  //group_size = 8

  // 定义当前方向和上一个方向的向量
  Eigen::Vector3d curr_direct(Eigen::Vector3d::Zero());
  Eigen::Vector3d last_direct(Eigen::Vector3d::Zero());

  // 定义索引变量i_nex和i2，用于追踪点云中的下一个点和临时存储当前点索引
  uint i_nex = 0, i2;
  // 定义上一个点和上一个点的下一个点的索引
  uint last_i = 0; uint last_i_nex = 0;
  // 定义上一个点的状态和当前平面类型
  int last_state = 0;
  int plane_type;

  // 遍历点云，从head开始，到plsize2结束
  for(uint i=head; i<plsize2; i++) 
  {
    // 跳过距离小于blind的点
    if(types[i].range < blind)
    {
      continue;
    }

    // 保存当前点索引到i2
    i2 = i;

    // 判断当前点的平面类型，并更新i_nex和curr_direct
    plane_type = plane_judge(pl, types, i, i_nex, curr_direct);
    
    // 如果点属于平面类型1（真实平面）
    if(plane_type == 1)
    {
      // 标记该平面上的点为真实平面或可能平面
      for(uint j=i; j<=i_nex; j++)  //遍历从当前点 i 到下一个点 i_nex 的所有点
      { 
        if(j!=i && j!=i_nex)        //如果不是起始点和终止点，标记为真实的平面，因为中间的点通常更稳定，边缘的点不一定属于这个平面
        {
          types[j].ftype = Real_Plane;  
        }
        else
        {
          types[j].ftype = Poss_Plane;
        }
      }
      
      // 根据上一个点的方向和当前点的方向判断是否为边缘平面
      if(last_state==1 && last_direct.norm()>0.1)
      {
        double mod = last_direct.transpose() * curr_direct; //计算上一个点和当前点的方向向量的点积 mod = |v1|*|v2|*cos(theta) 
        if(mod>-0.707 && mod<0.707)     //这一条件是基于点积的值来判断两个向量之间的夹角。这里的 -0.707 和 0.707 对应的是余弦值 
        {
          types[i].ftype = Edge_Plane;  //夹角在45度到135度之间：这意味着当前点的方向与上一个点的方向可能存在一个显著的转折,认为是边缘平面
        }
        else
        {
          types[i].ftype = Real_Plane;  //当两个向量的夹角小于45度或大于135度时，认为是真实平面
        }
      }
      
      // 更新循环索引和上一个点的状态
      i = i_nex - 1;
      last_state = 1;
    }
    // 如果点属于平面类型2（非平面）
    else // if(plane_type == 2)
    {
      // 更新循环索引和上一个点的状态
      i = i_nex;
      last_state = 0;
    }

    // 更新上一个点的相关变量
    last_i = i2;
    last_i_nex = i_nex;
    last_direct = curr_direct;
  }

  for(uint i=head+3; i<plsize2; i++)
  {
    // 跳过距离小于blind或已经被标记为真实平面的点
    if(types[i].range<blind || types[i].ftype>=Real_Plane)
    {
      continue;
    }

    // 跳过距离过近的点
    if(types[i-1].dista<1e-16 || types[i].dista<1e-16)
    {
      continue;
    }

    // 计算当前点的向量表示
    Eigen::Vector3d vec_a(pl[i].x, pl[i].y, pl[i].z);
    // 定义两个向量，分别表示当前点与前一个和后一个点的方向
    Eigen::Vector3d vecs[2];

    // 遍历计算两个方向的向量，并更新类型信息
    for(int j=0; j<2; j++)
    {
      int m = -1;
      if(j == 1)
      {
        m = 1;
      }

      // 如果相邻点距离小于blind，更新边缘类型并跳过
      if(types[i+m].range < blind)
      {
        if(types[i].range > inf_bound)
        {
          types[i].edj[j] = Nr_inf;
        }
        else
        {
          types[i].edj[j] = Nr_blind;
        }
        continue;
      }

      // 计算方向向量，并更新角度信息
      vecs[j] = Eigen::Vector3d(pl[i+m].x, pl[i+m].y, pl[i+m].z);
      vecs[j] = vecs[j] - vec_a;
      //vec_a为前一个点的方向向量，vecs[j]为后一个点的方向向量
      types[i].angle[j] = vec_a.dot(vecs[j]) / vec_a.norm() / vecs[j].norm(); //计算点积

      if(types[i].angle[j] < jump_up_limit)
      {
        types[i].edj[j] = Nr_180;
      }
        types[i].edj[j] = Nr_zero;
      }
    }

    // 计算两个方向向量的夹角，并判断是否为边缘跳跃
    types[i].intersect = vecs[Prev].dot(vecs[Next]) / vecs[Prev].norm() / vecs[Next].norm();
    if(types[i].edj[Prev]==Nr_nor && types[i].edj[Next]==Nr_zero && types[i].dista>0.0225 && types[i].dista>4*types[i-1].dista)
    {
      if(types[i].intersect > cos160)
      {
        if(edge_jump_judge(pl, types, i, Prev))
        {
          types[i].ftype = Edge_Jump;
        }
      }
    }
    else if(types[i].edj[Prev]==Nr_zero && types[i].edj[Next]== Nr_nor && types[i-1].dista>0.0225 && types[i-1].dista>4*types[i].dista)
    {
      if(types[i].intersect > cos160)
      {
        if(edge_jump_judge(pl, types, i, Next))
        {
          types[i].ftype = Edge_Jump;
        }
      }
    }
    else if(types[i].edj[Prev]==Nr_nor && types[i].edj[Next]==Nr_inf)
    {
      if(edge_jump_judge(pl, types, i, Prev))
      {
        types[i].ftype = Edge_Jump;
      }
    }
    else if(types[i].edj[Prev]==Nr_inf && types[i].edj[Next]==Nr_nor)
    {
      if(edge_jump_judge(pl, types, i, Next))
      {
        types[i].ftype = Edge_Jump;
      }
     
    }
    else if(types[i].edj[Prev]>Nr_nor && types[i].edj[Next]>Nr_nor)
    {
      if(types[i].ftype == Nor)
      {
        types[i].ftype = Wire;
      }
    }
  }

  // 进一步处理点云，识别并标记小平面
  plsize2 = plsize-1;
  double ratio;
  for(uint i=head+1; i<plsize2; i++)
  {
    // 跳过距离小于blind的点
    if(types[i].range<blind || types[i-1].range<blind || types[i+1].range<blind)
    {
      continue;
    }
    
    // 跳过距离过近的点
    if(types[i-1].dista<1e-8 || types[i].dista<1e-8)
    {
      continue;
    }

    // 如果点为普通类型，根据相邻点的距离比和夹角判断是否为真实平面
    if(types[i].ftype == Nor)
    {
      // 计算相邻点的距离比
      if(types[i-1].dista > types[i].dista)
      {
        ratio = types[i-1].dista / types[i].dista;
      }
      else
      {
        ratio = types[i].dista / types[i-1].dista;
      }

      // 如果夹角和距离比满足条件，将点和相邻点标记为真实平面
      if(types[i].intersect<smallp_intersect && ratio < smallp_ratio)
      {
        if(types[i-1].ftype == Nor)
        {
          types[i-1].ftype = Real_Plane;
        }
        if(types[i+1].ftype == Nor)
        {
          types[i+1].ftype = Real_Plane;
        }
        types[i].ftype = Real_Plane;
      }
    }
  }

  // 收集处理后的点云数据，将点根据类型添加到不同的容器中
  int last_surface = -1;
  for(uint j=head; j<plsize; j++)
  {
    // 如果点为真实平面或可能平面，添加到表面点云容器
    if(types[j].ftype==Poss_Plane || types[j].ftype==Real_Plane)
    {
      if(last_surface == -1)
      {
        last_surface = j;
      }
    
      if(j == uint(last_surface+point_filter_num-1))
      {
        PointType ap;
        ap.x = pl[j].x;
        ap.y = pl[j].y;
        ap.z = pl[j].z;
        ap.curvature = pl[j].curvature;
        pl_surf.push_back(ap);

        last_surface = -1;
      }
    }
    // 如果点为边缘跳跃或边缘平面，添加到角点容器
    else if(types[j].ftype==Edge_Jump || types[j].ftype==Edge_Plane)
    {
      pl_corn.push_back(pl[j]);
    }
    // 对于其他类型的点，计算平均值并添加到表面点云容器
    else if(last_surface != -1)
    {
      PointType ap;
      for(uint k=last_surface; k<j; k++)
      {
        ap.x += pl[k].x;
        ap.y += pl[k].y;
        ap.z += pl[k].z;
        ap.curvature += pl[k].curvature;
      }
      ap.x /= (j-last_surface);
      ap.y /= (j-last_surface);
      ap.z /= (j-last_surface);
      ap.curvature /= (j-last_surface);
      pl_surf.push_back(ap);
    }
    // 重置上一个表面点的索引
    last_surface = -1;
  }

//==================================================================
// 函 数 名：Preprocess::pub_func
// 功能描述：发布预处理后的点云数据。
//           将PointCloudXYZI类型的点云数据转换为适合ROS使用的格式(sensor_msgs::PointCloud2)，并设置其高度、宽度、坐标系ID及时间戳。
// 输入参数：
//   pl: PointCloudXYZI类型的引用，表示待发布的点云数据。
//   ct: ros::Time类型的常量引用，表示点云数据的捕获时间。
// 返 回 值：无
//==================================================================

void Preprocess::pub_func(PointCloudXYZI &pl, const ros::Time &ct)
{
  pl.height = 1;
  pl.width = pl.size();

  // 创建一个空的sensor_msgs::PointCloud2消息来存储点云数据。
  sensor_msgs::PointCloud2 output;

  pcl::toROSMsg(pl, output);
  // 设置点云消息的坐标系ID为"livox"。
  output.header.frame_id = "livox";
  // 设置点云消息的时间戳为给定的时间。
  output.header.stamp = ct;
}

// 根据输入的点云数据和当前点索引，判断下一个处理点的索引，并计算当前方向向量
// 参数:
// - pl: 输入的点云数据
// - types: 点云中每个点的类型信息
// - i_cur: 当前处理点的索引
// - i_nex: 输出的下一个处理点的索引
// - curr_direct: 输出的当前方向向量
// 返回值:
// - 1: 表示成功找到下一个处理点和方向向量
// - 0: 表示不满足条件，重置方向向量
// - 2: 表示遇到盲区，重置方向向量
int Preprocess::plane_judge(const PointCloudXYZI &pl, vector<orgtype> &types, uint i_cur, uint &i_nex, Eigen::Vector3d &curr_direct)
{
  // 计算并更新当前组的距离阈值
  double group_dis = disA*types[i_cur].range + disB;
  group_dis = group_dis * group_dis;

  // 定义变量存储距离数组
  vector<double> disarr;
  disarr.reserve(20);

  // 遍历当前组内的点，排除盲区并存储点到起始点的距离
  for(i_nex=i_cur; i_nex<i_cur+group_size; i_nex++)
  {
    if(types[i_nex].range < blind)
    {
      curr_direct.setZero();
      return 2;
    }
    disarr.push_back(types[i_nex].dista);
  }

  // 进一步搜索下一个处理点，确保不超出点云范围，不在盲区内，并计算距离阈值
  double two_dis;
  for(;;)
  {
    if((i_cur >= pl.size()) || (i_nex >= pl.size())) break;

    if(types[i_nex].range < blind)
    {
      curr_direct.setZero();
      return 2;
    }

    // 计算当前点与起始点的距离平方
    vx = pl[i_nex].x - pl[i_cur].x;
    vy = pl[i_nex].y - pl[i_cur].y;
    vz = pl[i_nex].z - pl[i_cur].z;
    two_dis = vx*vx + vy*vy + vz*vz;

    if(two_dis >= group_dis)
    {
      break;
    }

    disarr.push_back(types[i_nex].dista);
    i_nex++;
  }

  // 计算并处理当前组内的点的宽度和长度信息
  double leng_wid = 0;
  for(uint j=i_cur+1; j<i_nex; j++)
  {
    v1[0] = pl[j].x - pl[i_cur].x;
    v1[1] = pl[j].y - pl[i_cur].y;
    v1[2] = pl[j].z - pl[i_cur].z;

    // 计算向量叉乘以评估点的分布
    v2[0] = v1[1]*vz - vy*v1[2];
    v2[1] = v1[2]*vx - v1[0]*vz;
    v2[2] = v1[0]*vy - vx*v1[1];

    double lw = v2[0]*v2[0] + v2[1]*v2[1] + v2[2]*v2[2];
    if(lw > leng_wid)
    {
      leng_wid = lw;
    }
  }

  // 判断当前组是否满足平面条件 长度和宽度比值如果小于一个阈值，则认为不满足平面条件，重置方向向量并返回
  if((two_dis*two_dis/leng_wid) < p2l_ratio)
  {
    curr_direct.setZero();
    return 0;
  }

  // 对距离数组进行排序，用于后续的条件判断
  uint disarrsize = disarr.size();
  for(uint j=0; j<disarrsize-1; j++)
  {
    for(uint k=j+1; k<disarrsize; k++)
    {
      if(disarr[j] < disarr[k])
      {
        leng_wid = disarr[j];
        disarr[j] = disarr[k];
        disarr[k] = leng_wid;
      }
    }
  }

  // 检查第二小的距离值是否过小，若是，则重置方向向量并返回
  if(disarr[disarr.size()-2] < 1e-16)
  {
    curr_direct.setZero();
    return 0;
  }

  // 根据不同的激光雷达类型进行额外的条件判断
  if(lidar_type==AVIA)
  {
    double dismax_mid = disarr[0]/disarr[disarrsize/2];
    double dismid_min = disarr[disarrsize/2]/disarr[disarrsize-2];

    if(dismax_mid>=limit_maxmid || dismid_min>=limit_midmin)
    {
      curr_direct.setZero();
      return 0;
    }
  }
  else
  {
    double dismax_min = disarr[0] / disarr[disarrsize-2];
    if(dismax_min >= limit_maxmin)
    {
      curr_direct.setZero();
      return 0;
    }
  }

  // 更新当前方向向量并归一化
  curr_direct << vx, vy, vz;
  curr_direct.normalize();
  return 1;
}

/**
 * @brief 判断是否为边缘跳跃点
 * 
 * 该函数用于在预处理阶段判断某点是否可能是由于边缘跳跃引起的异常点。通过比较相邻点的距离和角度信息来判断。
 * 
 * @param pl 输入的点云数据
 * @param types 组织类型向量，包含了点的有关信息
 * @param i 当前判断点的索引
 * @param nor_dir 周围环境的方向，0表示前向，1表示后向
 * @return true 表示不是边缘跳跃点，false 表示可能是边缘跳跃点
 */
bool Preprocess::edge_jump_judge(const PointCloudXYZI &pl, vector<orgtype> &types, uint i, Surround nor_dir)
{
  // 根据周围环境方向判断盲区内的点
  if(nor_dir == 0)
  {
    // 判断前向盲区内的点是否满足条件
    if(types[i-1].range<blind || types[i-2].range<blind)
    {
      // 如果盲区内的点不满足条件，则认为是边缘跳跃点
      return false;
    }
  }
  else if(nor_dir == 1)
  {
    // 判断后向盲区内的点是否满足条件
    if(types[i+1].range<blind || types[i+2].range<blind)
    {
      // 如果盲区内的点不满足条件，则认为是边缘跳跃点
      return false;
    }
  }
  
  // 获取相邻点的距离信息
  double d1 = types[i+nor_dir-1].dista;
  double d2 = types[i+3*nor_dir-2].dista;
  
  // 交换距离，确保d1 >= d2
  if(d1<d2)
  {
    // 交换d1和d2的值
    d = d1;
    d1 = d2;
    d2 = d;
  }
  
  // 对距离进行开方处理
  d1 = sqrt(d1);
  d2 = sqrt(d2);
  
  // 根据设定的阈值判断是否为边缘跳跃点
  if(d1>edgea*d2 || (d1-d2)>edgeb)
  {
    return false;
  }
  
  return true;
}