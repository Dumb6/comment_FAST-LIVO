#include "lidar_selection.h"

namespace lidar_selection {

LidarSelector::LidarSelector(const int gridsize, SparseMap* sparsemap ): grid_size(gridsize), sparse_map(sparsemap)
{
    downSizeFilter.setLeafSize(0.2, 0.2, 0.2);
    G = Matrix<double, DIM_STATE, DIM_STATE>::Zero();
    H_T_H = Matrix<double, DIM_STATE, DIM_STATE>::Zero();
    Rli = M3D::Identity();
    Rci = M3D::Identity();
    Rcw = M3D::Identity();
    Jdphi_dR = M3D::Identity();
    Jdp_dt = M3D::Identity();
    Jdp_dR = M3D::Identity();
    Pli = V3D::Zero();
    Pci = V3D::Zero();
    Pcw = V3D::Zero();
    width = 800;
    height = 600;
}

LidarSelector::~LidarSelector() 
{
    delete sparse_map;
    delete sub_sparse_map;
    delete[] grid_num;
    delete[] map_index;
    delete[] map_value;
    delete[] align_flag;
    delete[] patch_cache;
    unordered_map<int, Warp*>().swap(Warp_map);
    unordered_map<VOXEL_KEY, float>().swap(sub_feat_map);
    unordered_map<VOXEL_KEY, VOXEL_POINTS*>().swap(feat_map);  
}

void LidarSelector::set_extrinsic(const V3D &transl, const M3D &rot)
{
    Pli = -rot.transpose() * transl;
    Rli = rot.transpose();
}

void LidarSelector::init()  
{
    sub_sparse_map = new SubSparseMap;
    Rci = sparse_map->Rcl * Rli;                    //相机坐标系到IMU坐标系的旋转矩阵和平移向量
    Pci= sparse_map->Rcl*Pli + sparse_map->Pcl;
    M3D Ric;                                    
    V3D Pic;
    Jdphi_dR = Rci;                                 //雅可比矩阵旋转部分
    Pic = -Rci.transpose() * Pci;
    M3D tmp;
    tmp << SKEW_SYM_MATRX(Pic);                     //反对称矩阵-->中间对旋转R求导会用到
    Jdp_dR = -Rci * tmp;
    width = cam->width();                           //相机图像的宽度和高度
    height = cam->height();
    grid_n_width = static_cast<int>(width/grid_size);//网格的宽度和高度数量
    grid_n_height = static_cast<int>(height/grid_size);
    length = grid_n_width * grid_n_height;
    fx = cam->errorMultiplier2();                   //相机内参
    fy = cam->errorMultiplier() / (4. * fx);
    grid_num = new int[length];                     //网格的大小
    map_index = new int[length];                    //索引
    map_value = new float[length];
    align_flag = new int[length];
    map_dist = (float*)malloc(sizeof(float)*length);
    memset(grid_num, TYPE_UNKNOWN, sizeof(int)*length);
    memset(map_index, 0, sizeof(int)*length);
    memset(map_value, 0, sizeof(float)*length);
    voxel_points_.reserve(length);                  //存储体素化后的点云数据
    add_voxel_points_.reserve(length);
    count_img = 0;
    patch_size_total = patch_size * patch_size;     //patch的大小 特征匹配会用到
    patch_size_half = static_cast<int>(patch_size/2);
    patch_cache = new float[patch_size_total];
    stage_ = STAGE_FIRST_FRAME;
    pg_down.reset(new PointCloudXYZI());
    Map_points.reset(new PointCloudXYZI());
    Map_points_output.reset(new PointCloudXYZI());
    weight_scale_ = 10;                             //用于鲁棒估计或优化过程中的权重计算
    weight_function_.reset(new vk::robust_cost::HuberWeightFunction());
    // weight_function_.reset(new vk::robust_cost::TukeyWeightFunction());
    scale_estimator_.reset(new vk::robust_cost::UnitScaleEstimator());
    // scale_estimator_.reset(new vk::robust_cost::MADScaleEstimator());
}

//重置网格相关的数据结构到一个已知的初始状态
void LidarSelector::reset_grid()
{
    memset(grid_num, TYPE_UNKNOWN, sizeof(int)*length);
    memset(map_index, 0, sizeof(int)*length);
    fill_n(map_dist, length, 10000);
    std::vector<PointPtr>(length).swap(voxel_points_);
    std::vector<V3D>(length).swap(add_voxel_points_);
    voxel_points_.reserve(length);
    add_voxel_points_.reserve(length);
}

//计算投影方程雅可比矩阵
//详细可见公式推导
void LidarSelector::dpi(V3D p, MD(2,3)& J) 
{
    const double x = p[0];
    const double y = p[1];
    const double z_inv = 1./p[2];
    const double z_inv_2 = z_inv * z_inv;
    J(0,0) = fx * z_inv;                //这里的雅可比矩阵 -----> 针孔相机模型 ------> du_dp
    J(0,1) = 0.0;
    J(0,2) = -fx * x * z_inv_2;
    J(1,0) = 0.0;
    J(1,1) = fy * z_inv;
    J(1,2) = -fy * y * z_inv_2;
}


//传入图像以及特征点坐标
//计算图像中特定点周围的梯度强度，梯度强度通常用于评估图像中的边缘或特征点
//梯度值越大，表示该点周围的图像变化越剧烈，可能代表这是一个"好"的特征点。相反，如果梯度值很小，可能表示该点位于图像的平坦区域，可能不是一个理想的特征点。
float LidarSelector::CheckGoodPoints(cv::Mat img, V2D uv)
{
    const float u_ref = uv[0];      //函数首先提取 uv 中的 u 和 v 坐标
    const float v_ref = uv[1];  
    const int u_ref_i = floorf(uv[0]); //整数部分
    const int v_ref_i = floorf(uv[1]);
    const float subpix_u_ref = u_ref-u_ref_i;//小数部分
    const float subpix_v_ref = v_ref-v_ref_i;
    uint8_t* img_ptr = (uint8_t*) img.data + (v_ref_i)*width + (u_ref_i);//函数计算出图像中对应坐标点的指针位置
    //函数计算 u 和 v 方向的梯度（ gu 和 gv ）
    //这里使用了一种类似Sobel算子的方法来计算梯度
    float gu = 2*(img_ptr[1] - img_ptr[-1]) + img_ptr[1-width] - img_ptr[-1-width] + img_ptr[1+width] - img_ptr[-1+width];
    float gv = 2*(img_ptr[width] - img_ptr[-width]) + img_ptr[width+1] - img_ptr[-width+1] + img_ptr[width-1] - img_ptr[-width-1];
    return fabs(gu)+fabs(gv);
}


//从图像中提取一个图像块（patch）
void LidarSelector::getpatch(cv::Mat img, V2D pc, float* patch_tmp, int level)  // level:图像金字塔的层级
{
    //首先计算参考点的整数坐标和亚像素偏移。
    //scale 是基于 level 的尺度因子，用于在图像金字塔的不同层级间进行缩放
    const float u_ref = pc[0];
    const float v_ref = pc[1];
    const int scale =  (1<<level);
    const int u_ref_i = floorf(pc[0]/scale)*scale; 
    const int v_ref_i = floorf(pc[1]/scale)*scale;
    const float subpix_u_ref = (u_ref-u_ref_i)/scale;
    const float subpix_v_ref = (v_ref-v_ref_i)/scale;
    //双线性插值权重：
    //计算了四个权重 w_ref_tl, w_ref_tr, w_ref_bl, w_ref_br，用于双线性插值。
    const float w_ref_tl = (1.0-subpix_u_ref) * (1.0-subpix_v_ref);
    const float w_ref_tr = subpix_u_ref * (1.0-subpix_v_ref);
    const float w_ref_bl = (1.0-subpix_u_ref) * subpix_v_ref;
    const float w_ref_br = subpix_u_ref * subpix_v_ref;
    //外循环遍历行，内循环遍历列 patch_size 决定了提取的图像块大小
    for (int x=0; x<patch_size; x++) 
    {
        uint8_t* img_ptr = (uint8_t*) img.data + (v_ref_i-patch_size_half*scale+x*scale)*width + (u_ref_i-patch_size_half*scale);
        for (int y=0; y<patch_size; y++, img_ptr+=scale)
        {
            //对于图像块中的每个位置，使用双线性插值计算像素值
            //插值考虑了周围四个像素的加权和
            patch_tmp[patch_size_total*level+x*patch_size+y] = w_ref_tl*img_ptr[0] + w_ref_tr*img_ptr[scale] + w_ref_bl*img_ptr[scale*width] + w_ref_br*img_ptr[scale*width+scale];
        }
    }
}

void LidarSelector::addSparseMap(cv::Mat img, PointCloudXYZI::Ptr pg) 
{
    // double t0 = omp_get_wtime();
    reset_grid();   //重置网格相关的数据结构到一个已知的初始状态

    // double t_b1 = omp_get_wtime() - t0;
    // t0 = omp_get_wtime();
    // 循环遍历输入的点云 pg

    // 计算点在网格中的索引。
    // 使用 Shi-Tomasi 角点检测算法计算该点的"好"程度（vk::shiTomasiScore）。
    // 如果当前点的分数高于该网格位置已存储的分数，则更新该网格的信息。
    for (int i=0; i<pg->size(); i++)    
    {
        V3D pt(pg->points[i].x, pg->points[i].y, pg->points[i].z);
        V2D pc(new_frame_->w2c(pt));    //对每个点，将其从世界坐标系转换到相机坐标系   
        // 检查点是否在图像帧内
        if(new_frame_->cam_->isInFrame(pc.cast<int>(), (patch_size_half+1)*8)) // 20px is the patch size in the matcher
        {
            int index = static_cast<int>(pc[0]/grid_size)*grid_n_height + static_cast<int>(pc[1]/grid_size);//计算点在网格中的索引
            // float cur_value = CheckGoodPoints(img, pc);
            float cur_value = vk::shiTomasiScore(img, pc[0], pc[1]);    //使用 Shi-Tomasi 角点检测算法计算该点分数->"好"的程度（vk::shiTomasiScore）
            //如果当前点的分数高于该网格位置已存储的分数，则更新该网格的信息
            if (cur_value > map_value[index]) //&& (grid_num[index] != TYPE_MAP || map_value[index]<=10)) //! only add in not occupied grid
            {
                map_value[index] = cur_value;
                add_voxel_points_[index] = pt;
                grid_num[index] = TYPE_POINTCLOUD;
            }
        }
    }

    // double t_b2 = omp_get_wtime() - t0;
    // t0 = omp_get_wtime();
    
    int add=0;
    //循环遍历所有网格
    for (int i=0; i<length; i++) 
    {
        if (grid_num[i]==TYPE_POINTCLOUD)// && (map_value[i]>=10)) //! debug
        {
            V3D pt = add_voxel_points_[i];
            V2D pc(new_frame_->w2c(pt));//将点从世界坐标系转换到相机坐标系
            float* patch = new float[patch_size_total*3];
             //在三个不同的尺度上提取 patch
            getpatch(img, pc, patch, 0);//0，1，2 --- > level 0,1,2
            getpatch(img, pc, patch, 1);
            getpatch(img, pc, patch, 2);
            PointPtr pt_new(new Point(pt));
            Vector3d f = cam->cam2world(pc);
            FeaturePtr ftr_new(new Feature(patch, pc, f, new_frame_->T_f_w_, map_value[i], 0));
            ftr_new->img = new_frame_->img_pyr_[0];
            // ftr_new->ImgPyr.resize(5);
            // for(int i=0;i<5;i++) ftr_new->ImgPyr[i] = new_frame_->img_pyr_[i];
            ftr_new->id_ = new_frame_->id_;// 将选中的点转换为 Point 和 Feature 对象:Point -> Feature Point

            pt_new->addFrameRef(ftr_new);   
            pt_new->value = map_value[i];
            AddPoint(pt_new);   //将这些点添加到稀疏地图中
            add += 1;
        }
    }

    // double t_b3 = omp_get_wtime() - t0;

    printf("[ VIO ]: Add %d 3D points.\n", add);    //打印个数
    // printf("pg.size: %d \n", pg->size());
    // printf("B1. : %.6lf \n", t_b1);
    // printf("B2. : %.6lf \n", t_b2);
    // printf("B3. : %.6lf \n", t_b3);
}

//用于将新的点添加到体素化的特征地图中
void LidarSelector::AddPoint(PointPtr pt_new)
{
    V3D pt_w(pt_new->pos_[0], pt_new->pos_[1], pt_new->pos_[2]);
    double voxel_size = 0.5;
    float loc_xyz[3];
    //计算体素坐标
    for(int j=0; j<3; j++)
    {
      loc_xyz[j] = pt_w[j] / voxel_size;    //将点的世界坐标除以体素大小，得到体素坐标
      if(loc_xyz[j] < 0)
      {
        loc_xyz[j] -= 1.0;
      }
    }
    //使用计算出的体素坐标创建一个 VOXEL_KEY 这是一个类，用于唯一标识一个体素
    VOXEL_KEY position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);

    auto iter = feat_map.find(position); //在 feat_map 中查找当前体素键
    if(iter != feat_map.end())
    {
      iter->second->voxel_points.push_back(pt_new);     // 将新点添加到该体素的点列表中。
      iter->second->count++;                            // 增加该体素的点计数。
    }
    else                                                // 如果没有找到对应的体素：
    {
      VOXEL_POINTS *ot = new VOXEL_POINTS(0);           // 创建一个新的 VOXEL_POINTS 对象。 
      ot->voxel_points.push_back(pt_new);               // 将新点添加到这个新创建的体素对象中。
      feat_map[position] = ot;                          // 将新的体素对象添加到 feat_map 中。
    }
}

/*仿射变换矩阵
计算从参考帧到当前帧的图像块仿射变换矩阵
cam: 相机模型
px_ref: 参考帧中的像素坐标
f_ref: 参考帧中的归一化平面坐标
depth_ref: 参考点的深度
T_cur_ref: 从参考帧到当前帧的变换
level_ref: 参考像素对应的图像金字塔级别
pyramid_level: 当前处理的图像金字塔级别
halfpatch_size: 图像块半尺寸*/
void LidarSelector::getWarpMatrixAffine(
    const vk::AbstractCamera& cam,
    const Vector2d& px_ref,
    const Vector3d& f_ref,
    const double depth_ref,
    const SE3& T_cur_ref,
    const int level_ref,    // the corresponding pyrimid level of px_ref
    const int pyramid_level,
    const int halfpatch_size,
    Matrix2d& A_cur_ref)
{
    //归一化平面坐标->它代表了将3D点投影到一个假想的、位于相机前方单位距离处的平面上的坐标
    //归一化平面坐标通常是一个三维向量 (x, y, 1)
    //深度 ：Z ---> 归一化平面坐标 = (X/Z, Y/Z, 1)
    const Vector3d xyz_ref(f_ref*depth_ref);  //归一化平面坐标和深度结合 --> 3D点计算 ---->(x,y,z)
    //这里计算了图像块在u和v方向上的边缘点，并将它们转换到3D空间 (1<<level_ref)*(1<<pyramid_level) 考虑了图像金字塔的尺度。
    Vector3d xyz_du_ref(cam.cam2world(px_ref + Vector2d(halfpatch_size,0)*(1<<level_ref)*(1<<pyramid_level)));
    Vector3d xyz_dv_ref(cam.cam2world(px_ref + Vector2d(0,halfpatch_size)*(1<<level_ref)*(1<<pyramid_level)));

    // 为什么这么做？：
    // a. 平面假设：在小范围内（图像块大小），我们假设场景是平面的
    // 调整深度后，所有点都位于与相机平行的平面上 当所有点在同一深度时，它们在图像平面上的投影变换可以更好地用仿射变换近似。
    //这一步将边缘点的深度调整到与中心点相同。       ---xyz_ref[2] = Z   ---xyz_du_ref[2] = Z‘
    xyz_du_ref *= xyz_ref[2]/xyz_du_ref[2]; //[X',Y',Z'] * Z/Z' = [X'*Z/Z',Y*Z/Z',Z]
    xyz_dv_ref *= xyz_ref[2]/xyz_dv_ref[2];

    //T_cur_ref 是从参考帧到当前帧的变换矩阵 ref -> cur
    //将参考帧中的3D点变换到当前帧的3D点坐标系中，并投影到图像平面：
    // px_cur：中心点
    // px_du：u方向的边缘点 
    // px_dv：v方向的边缘点
    const Vector2d px_cur(cam.world2cam(T_cur_ref*(xyz_ref)));
    const Vector2d px_du(cam.world2cam(T_cur_ref*(xyz_du_ref)));
    const Vector2d px_dv(cam.world2cam(T_cur_ref*(xyz_dv_ref)));

    //计算仿射变换矩阵  
    // a) 差分近似：
    // (px_du - px_cur) 表示图像块在 u 方向的变化
    // (px_dv - px_cur) 表示图像块在 v 方向的变化
    // b) 归一化：
    // 除以 halfpatch_size 使得变换与图像块的实际大小无关
    // 这使得计算结果可以应用于不同尺寸的图像块
    A_cur_ref.col(0) = (px_du - px_cur)/halfpatch_size; //col(0) 表示矩阵的第一列 2x1
    A_cur_ref.col(1) = (px_dv - px_cur)/halfpatch_size;
    //输出的：A_cur_ref -> 2x2的仿射变换矩阵
    //A_cur_ref 的第一列表示单位 u 方向的变化 第二列表示单位 v 方向的变化
    //总结：这个仿射变换矩阵 A_cur_ref 描述了如何将参考帧中的图像块变形到当前帧中
}

//执行图像的仿射变换
void LidarSelector::warpAffine(
    const Matrix2d& A_cur_ref,
    const cv::Mat& img_ref,
    const Vector2d& px_ref,
    const int level_ref,
    const int search_level,
    const int pyramid_level,
    const int halfpatch_size,
    float* patch)   //用于存储结果的patch
{
  const int patch_size = halfpatch_size*2 ;     //补丁大小
  const Matrix2f A_ref_cur = A_cur_ref.inverse().cast<float>(); // --->A_cur_ref 的逆矩阵 float
  if(isnan(A_ref_cur(0,0))) //函数检查仿射变换矩阵是否包含 NaN 值，如果包含则输出警告并返回
  {
    printf("Affine warp is NaN, probably camera has no translation\n"); // TODO
    return;
  }
//   Perform the warp on a larger patch.
//   float* patch_ptr = patch;
//   const Vector2f px_ref_pyr = px_ref.cast<float>() / (1<<level_ref) / (1<<pyramid_level);
//   const Vector2f px_ref_pyr = px_ref.cast<float>() / (1<<level_ref);
    //遍历patch的每个像素 (y, x)
    //遍历patch的行 
  for (int y=0; y<patch_size; ++y)
  {
    //遍历patch的列
    for (int x=0; x<patch_size; ++x)//, ++patch_ptr)
    {
      // P[patch_size_total*level + x*patch_size+y]
      Vector2f px_patch(x-halfpatch_size, y-halfpatch_size);    //计算相对于补丁中心的位置 px_patch
      px_patch *= (1<<search_level);
      px_patch *= (1<<pyramid_level);                           //根据搜索级别和金字塔级别调整 px_patch
      const Vector2f px(A_ref_cur*px_patch + px_ref.cast<float>());
      if (px[0]<0 || px[1]<0 || px[0]>=img_ref.cols-1 || px[1]>=img_ref.rows-1) //如果超出范围
        patch[patch_size_total*pyramid_level + y*patch_size+x] = 0;             //设置对应的补丁像素值为0
        // *patch_ptr = 0;
      else
        patch[patch_size_total*pyramid_level + y*patch_size+x] = (float) vk::interpolateMat_8u(img_ref, px[0], px[1]);
        //如果在范围内,使用插值方法从参考图像中获取像素值,并赋值给patch

        //将计算得到的像素值存储在 patch 数组中,位置由 pyramid_level, y, 和 x 决定
    }
  }
}



// 归一化互相关（Normalized Cross-Correlation，NCC）算法    用于衡量两个patch之间的相似度
/*  ref_patch: 参考patch
    cur_patch: 当前patch
    patch_size: patch的大小（像素总数）*/
double LidarSelector::NCC(float* ref_patch, float* cur_patch, int patch_size)
{    
    //参考patch的平均值
    double sum_ref = std::accumulate(ref_patch, ref_patch + patch_size, 0.0);
    double mean_ref =  sum_ref / patch_size;

    //当前patch的平均值
    double sum_cur = std::accumulate(cur_patch, cur_patch + patch_size, 0.0);
    double mean_curr =  sum_cur / patch_size;

    //初始化NCC公式的分子和分母
    double numerator = 0, demoniator1 = 0, demoniator2 = 0;
    for (int i = 0; i < patch_size; i++) 
    {
        double n = (ref_patch[i] - mean_ref) * (cur_patch[i] - mean_curr);
        numerator += n;     //分子-> 与平均值差值乘积
        demoniator1 += (ref_patch[i] - mean_ref) * (ref_patch[i] - mean_ref);   //分母-> 与平均值差值平方->方差
        demoniator2 += (cur_patch[i] - mean_curr) * (cur_patch[i] - mean_curr);
    }
    return numerator / sqrt(demoniator1 * demoniator2 + 1e-10);     //NCC的值范围是 [-1, 1] 1->100%正相关 0->0%相关 -1->100%负相关
}

//用于确定在图像金字塔中进行搜索的最佳层级
int LidarSelector::getBestSearchLevel(
    const Matrix2d& A_cur_ref,
    const int max_level)//允许的最大搜索层级
{
    //仿射变换的行列式（D）表示了变换导致的面积变化 ---> D > 1 意味着面积扩大，D < 1 意味着面积缩小
    //起始条件 D > 3.0 表示当前帧中的特征在参考帧中对应一个明显更大的区域

    int search_level = 0;
    double D = A_cur_ref.determinant();   //D --> 仿射变换矩阵的行列式

    while(D > 3.0 && search_level < max_level)    //当 D > 3.0 且 search_level 小于 max_level 时，循环继续
    {
        search_level += 1;//每次将 search_level 提高一级，相当于将图像分辨率降低一半。在图像金字塔中，每上升一层，图像在每个维度上缩小一半，
        D *= 0.25;        //因此面积变为原来的 1/4 这就是为什么每次循环 D 乘以 0.25                         
                        
    }
    return search_level;
}

//这个函数的作用从一个带边界的patch中提取出中心的图像块
/*    带边界的图像块 (5x5)       提取后的图像块 (3x3)
    +---+---+---+---+---+      
    | * | * | * | * | * |      
    +---+---+---+---+---+      +---+---+---+
    | * | A | B | C | * |      | A | B | C |
    +---+---+---+---+---+  =>  +---+---+---+
    | * | D | E | F | * |      | D | E | F |
    +---+---+---+---+---+      +---+---+---+
    | * | G | H | I | * |      | G | H | I |
    +---+---+---+---+---+      +---+---+---+
    | * | * | * | * | * |      
    +---+---+---+---+---+
*/
void LidarSelector::createPatchFromPatchWithBorder(float* patch_with_border, float* patch_ref)
{
  float* ref_patch_ptr = patch_ref;
  //外层循环 y 方向
  for(int y=1; y<patch_size+1; ++y, ref_patch_ptr += patch_size)   // 从y=1开始，因为要跳过顶部边界 
  {                                                                //循环到 patch_size+1，因为要考虑底部边界 
    float* ref_patch_border_ptr = patch_with_border + y*(patch_size+2) + 1;
    for(int x=0; x<patch_size; ++x) //内层循环 x 方向                //每次迭代后，ref_patch_ptr 移动到下一行的开始
      ref_patch_ptr[x] = ref_patch_border_ptr[x];                  //复制 patch_size 个元素，忽略右边界
  }
}


//从稀疏地图中选择和添加特征点到当前帧
//使用体素化方法构建子特征地图
void LidarSelector::addFromSparseMap(cv::Mat img, PointCloudXYZI::Ptr pg)
{
    if(feat_map.size()<=0) return;
    // double ts0 = omp_get_wtime();
    //降采样
    pg_down->reserve(feat_map.size());
    downSizeFilter.setInputCloud(pg);
    downSizeFilter.filter(*pg_down);
    
    reset_grid();
    memset(map_value, 0, sizeof(float)*length);

    sub_sparse_map->reset();
    deque< PointPtr >().swap(sub_map_cur_frame_);

    float voxel_size = 0.5;
    
    unordered_map<VOXEL_KEY, float>().swap(sub_feat_map);
    unordered_map<int, Warp*>().swap(Warp_map);

    cv::Mat depth_img = cv::Mat::zeros(height, width, CV_32FC1);
    float* it = (float*)depth_img.data;

    double t_insert, t_depth, t_position;
    t_insert=t_depth=t_position=0;

    int loc_xyz[3];

    // printf("A0. initial depthmap: %.6lf \n", omp_get_wtime() - ts0);
    // double ts1 = omp_get_wtime();

    for(int i=0; i<pg_down->size(); i++)
    {
        // Transform Point to world coordinate
        V3D pt_w(pg_down->points[i].x, pg_down->points[i].y, pg_down->points[i].z);

        // Determine the key of hash table      
        for(int j=0; j<3; j++)
        {
            loc_xyz[j] = floor(pt_w[j] / voxel_size);
        }
        VOXEL_KEY position(loc_xyz[0], loc_xyz[1], loc_xyz[2]);

        auto iter = sub_feat_map.find(position);
        if(iter == sub_feat_map.end())
        {
            sub_feat_map[position] = 1.0;
        }
                    
        V3D pt_c(new_frame_->w2f(pt_w));

        V2D px;
        if(pt_c[2] > 0)
        {
            px[0] = fx * pt_c[0]/pt_c[2] + cx;
            px[1] = fy * pt_c[1]/pt_c[2] + cy;

            if(new_frame_->cam_->isInFrame(px.cast<int>(), (patch_size_half+1)*8))
            {
                float depth = pt_c[2];
                int col = int(px[0]);
                int row = int(px[1]);
                it[width*row+col] = depth;        
            }
        }
    }
    
    // imshow("depth_img", depth_img);
    // printf("A1: %.6lf \n", omp_get_wtime() - ts1);
    // printf("A11. calculate pt position: %.6lf \n", t_position);
    // printf("A12. sub_postion.insert(position): %.6lf \n", t_insert);
    // printf("A13. generate depth map: %.6lf \n", t_depth);
    // printf("A. projection: %.6lf \n", omp_get_wtime() - ts0);
    

    // double t1 = omp_get_wtime();

    for(auto& iter : sub_feat_map)
    {   
        VOXEL_KEY position = iter.first;
        // double t4 = omp_get_wtime();
        auto corre_voxel = feat_map.find(position);
        // double t5 = omp_get_wtime();

        if(corre_voxel != feat_map.end())
        {
            std::vector<PointPtr> &voxel_points = corre_voxel->second->voxel_points;
            int voxel_num = voxel_points.size();
            for (int i=0; i<voxel_num; i++)
            {
                PointPtr pt = voxel_points[i];
                if(pt==nullptr) continue;
                V3D pt_cam(new_frame_->w2f(pt->pos_));
                if(pt_cam[2]<0) continue;

                V2D pc(new_frame_->w2c(pt->pos_));

                FeaturePtr ref_ftr;
      
                if(new_frame_->cam_->isInFrame(pc.cast<int>(), (patch_size_half+1)*8)) // 20px is the patch size in the matcher
                {
                    int index = static_cast<int>(pc[0]/grid_size)*grid_n_height + static_cast<int>(pc[1]/grid_size);
                    grid_num[index] = TYPE_MAP;
                    Vector3d obs_vec(new_frame_->pos() - pt->pos_);

                    float cur_dist = obs_vec.norm();
                    float cur_value = pt->value;

                    if (cur_dist <= map_dist[index]) 
                    {
                        map_dist[index] = cur_dist;
                        voxel_points_[index] = pt;
                    } 

                    if (cur_value >= map_value[index])
                    {
                        map_value[index] = cur_value;
                    }
                }
            }    
        } 
    }
        
    // double t2 = omp_get_wtime();

    // cout<<"B. feat_map.find: "<<t2-t1<<endl;

    double t_2, t_3, t_4, t_5;
    t_2=t_3=t_4=t_5=0;

    for (int i=0; i<length; i++) 
    { 
        if (grid_num[i]==TYPE_MAP) //&& map_value[i]>10)
        {
            // double t_1 = omp_get_wtime();

            PointPtr pt = voxel_points_[i];

            if(pt==nullptr) continue;

            V2D pc(new_frame_->w2c(pt->pos_));
            V3D pt_cam(new_frame_->w2f(pt->pos_));
   
            bool depth_continous = false;
            for (int u=-patch_size_half; u<=patch_size_half; u++)
            {
                for (int v=-patch_size_half; v<=patch_size_half; v++)
                {
                    if(u==0 && v==0) continue;

                    float depth = it[width*(v+int(pc[1]))+u+int(pc[0])];

                    if(depth == 0.) continue;

                    double delta_dist = abs(pt_cam[2]-depth);

                    if(delta_dist > 1.5)
                    {                
                        depth_continous = true;
                        break;
                    }
                }
                if(depth_continous) break;
            }
            if(depth_continous) continue;

            // t_2 += omp_get_wtime() - t_1;

            // t_1 = omp_get_wtime();
            
            FeaturePtr ref_ftr;

            if(!pt->getCloseViewObs(new_frame_->pos(), ref_ftr, pc)) continue;

            // t_3 += omp_get_wtime() - t_1;

            float* patch_wrap = new float[patch_size_total*3];

            patch_wrap = ref_ftr->patch;

            // t_1 = omp_get_wtime();
           
            int search_level;
            Matrix2d A_cur_ref_zero;

            auto iter_warp = Warp_map.find(ref_ftr->id_);
            if(iter_warp != Warp_map.end())
            {
                search_level = iter_warp->second->search_level;
                A_cur_ref_zero = iter_warp->second->A_cur_ref;
            }
            else
            {
                getWarpMatrixAffine(*cam, ref_ftr->px, ref_ftr->f, (ref_ftr->pos() - pt->pos_).norm(), 
                new_frame_->T_f_w_ * ref_ftr->T_f_w_.inverse(), 0, 0, patch_size_half, A_cur_ref_zero);
                
                search_level = getBestSearchLevel(A_cur_ref_zero, 2);

                Warp *ot = new Warp(search_level, A_cur_ref_zero);
                Warp_map[ref_ftr->id_] = ot;
            }

            // t_4 += omp_get_wtime() - t_1;

            // t_1 = omp_get_wtime();

            for(int pyramid_level=0; pyramid_level<=0; pyramid_level++)
            {                
                warpAffine(A_cur_ref_zero, ref_ftr->img, ref_ftr->px, ref_ftr->level, search_level, pyramid_level, patch_size_half, patch_wrap);
            }

            getpatch(img, pc, patch_cache, 0);

            if(ncc_en)
            {
                double ncc = NCC(patch_wrap, patch_cache, patch_size_total);
                if(ncc < ncc_thre) continue;
            }

            float error = 0.0;
            for (int ind=0; ind<patch_size_total; ind++) 
            {
                error += (patch_wrap[ind]-patch_cache[ind]) * (patch_wrap[ind]-patch_cache[ind]);
            }
            if(error > outlier_threshold*patch_size_total) continue;
            
            sub_map_cur_frame_.push_back(pt);

            sub_sparse_map->align_errors.push_back(error);
            sub_sparse_map->propa_errors.push_back(error);
            sub_sparse_map->search_levels.push_back(search_level);
            sub_sparse_map->errors.push_back(error);
            sub_sparse_map->index.push_back(i);  //index
            sub_sparse_map->voxel_points.push_back(pt);
            sub_sparse_map->patch.push_back(patch_wrap);
            // sub_sparse_map->px_cur.push_back(pc);
            // sub_sparse_map->propa_px_cur.push_back(pc);
            // t_5 += omp_get_wtime() - t_1;
        }
    }
    // double t3 = omp_get_wtime();
    // cout<<"C. addSubSparseMap: "<<t3-t2<<endl;
    // cout<<"depthcontinuous: C1 "<<t_2<<" C2 "<<t_3<<" C3 "<<t_4<<" C4 "<<t_5<<endl;
    printf("[ VIO ]: choose %d points from sub_sparse_map.\n", int(sub_sparse_map->index.size()));
}

bool LidarSelector::align2D(        //alingn --> 对齐
    const cv::Mat& cur_img,         //当前图像
    float* ref_patch_with_border,   //这是指向带边界的参考图像块起始位置的指针
    float* ref_patch,
    const int n_iter,               //迭代次数
    Vector2d& cur_px_estimate,      //当前投影像素估计坐标
    int index)                      //索引
{
#ifdef __ARM_NEON__
  if(!no_simd)
    return align2D_NEON(cur_img, ref_patch_with_border, ref_patch, n_iter, cur_px_estimate);
#endif

  const int halfpatch_size_ = 4;
  const int patch_size_ = 8;        //patch = 8
  const int patch_area_ = 64;
  bool converged=false;             //收敛标志位

  // 计算模板的导数并准备逆组合
  float __attribute__((__aligned__(16))) ref_patch_dx[patch_area_]; //声明对齐的数组用于存储参考图像块的 x 和 y 方向导数
  float __attribute__((__aligned__(16))) ref_patch_dy[patch_area_];
  Matrix3f H; H.setZero();  //初始化 3x3 的 Hessian 矩阵 H 为零矩阵

  // 计算梯度和 Hessian 矩阵
  const int ref_step = patch_size_+2;   //参考图像块带有边界 比实际的patch大2个像素
  float* it_dx = ref_patch_dx;          //初始化指针it_dx和it_dy，分别指向存储x方向和y方向导数的数组。
  float* it_dy = ref_patch_dy;
  for(int y=0; y<patch_size_; ++y)      //外层循环，遍历图像块的每一行
  { //计算当前行起始像素的指针
    float* it = ref_patch_with_border + (y+1)*ref_step + 1;     //(y+1)*ref_step跳过上边界和之前的行  + 1跳过左边界，确保我们从实际的图像块内容开始。
    for(int x=0; x<patch_size_; ++x, ++it, ++it_dx, ++it_dy)    //内层循环，遍历当前行的每个像素 同时递增it、it_dx和it_dy指针
    {
      Vector3f J;       //创建一个3D向量J来存储Jacobian
      J[0] = 0.5 * (it[1] - it[-1]);    //计算x方向的导数，使用中心差分法 it[1]是右侧像素，it[-1]是左侧像素。
      J[1] = 0.5 * (it[ref_step] - it[-ref_step]); //计算y方向的导数 it[ref_step]是下方像素，it[-ref_step]是上方像素。
      J[2] = 1; 
      *it_dx = J[0];
      *it_dy = J[1];
      H += J*J.transpose();         // H = J*J^T J = [a, b, 1]
                                    // | a*a  a*b  a |
                                // H = | b*a  b*b  b |
                                    // | a    b    1 |
    }
  }
  Matrix3f Hinv = H.inverse();  //计算 Hessian 矩阵的逆
  float mean_diff = 0;

  // 计算新图像中的像素位置：
  float u = cur_px_estimate.x();    // 当前估计的x坐标
  float v = cur_px_estimate.y();

  // 收敛条件
  const float min_update_squared = 0.03*0.03;   //0.03*0.03
  const int cur_step = cur_img.step.p[0];
  float chi2 = 0;
  chi2 = sub_sparse_map->propa_errors[index];
  Vector3f update; update.setZero();
  for(int iter = 0; iter<n_iter; ++iter)        //迭代
  {
    int u_r = floor(u);
    int v_r = floor(v);
    //确保搜索区域在图像范围内
    if(u_r < halfpatch_size_ || v_r < halfpatch_size_ || u_r >= cur_img.cols-halfpatch_size_ || v_r >= cur_img.rows-halfpatch_size_)
      break;

    if(isnan(u) || isnan(v)) // TODO very rarely this can happen, maybe H is singular? should not be at corner.. check
      return false; 

    // compute interpolation weights
    // 双线性插值权重计算：
    float subpix_x = u-u_r; //像素差值
    float subpix_y = v-v_r;
    float wTL = (1.0-subpix_x)*(1.0-subpix_y);//权重系数
    float wTR = subpix_x * (1.0-subpix_y);
    float wBL = (1.0-subpix_x)*subpix_y;
    float wBR = subpix_x * subpix_y;

    // loop through search_patch, interpolate
    float* it_ref = ref_patch;
    float* it_ref_dx = ref_patch_dx;
    float* it_ref_dy = ref_patch_dy;
    float new_chi2 = 0.0;
    Vector3f Jres; Jres.setZero();
    //光流理论：假设：场景中的物体在短时间内的位移很小，且它们的亮度在连续帧之间保持不变
    for(int y=0; y<patch_size_; ++y)
    {
      uint8_t* it = (uint8_t*) cur_img.data + (v_r+y-halfpatch_size_)*cur_step + u_r-halfpatch_size_; 
      for(int x=0; x<patch_size_; ++x, ++it, ++it_ref, ++it_ref_dx, ++it_ref_dy)
      {
        float search_pixel = wTL*it[0] + wTR*it[1] + wBL*it[cur_step] + wBR*it[cur_step+1]; //根据上下左右四个像素值乘上权重得到亚像素的像素值
        float res = search_pixel - *it_ref + mean_diff;                     //残差 = 算出来的像素值 - 前一帧的像素值 + 光度补偿
        //梯度指向的是图像局部结构变化最显著的方向
        //I(x,y,t) - I(x+dx, y+dy, t+dt) = Ixdx + Iydy + It
        //最小化目标：我们要最小化 (Ixdx + Iydy + It)²
        //雅可比矩阵表示误差函数对参数的偏导数
        //偏导数 = res * Ix
        //梯度向量G = J^T * res 
        Jres[0] -= res*(*it_ref_dx);
        Jres[1] -= res*(*it_ref_dy);
        Jres[2] -= res;
        new_chi2 += res*res;
      }
    }

    if(iter > 0 && new_chi2 > chi2) //下一次的res > 上一次 -> 发散
    {
    //   cout << "error increased." << endl;
      //如果发散发生，代码会回退上一次的更新：
      u -= update[0];
      v -= update[1];
      break;
    }
    //如果当前迭代的误差没有发散，chi2 被更新为当前的 new_chi2，用于下次迭代时的发散检查
    chi2 = new_chi2;

    //记录对齐误差：在当前索引下，将当前的误差 new_chi2 存储到 sub_sparse_map->align_errors 中
    sub_sparse_map->align_errors[index] = new_chi2;

    update = Hinv * Jres; //3x3 3x1 = 3x1的列向量

    //update[0] 和 update[1] =  Delta x 和 Delta y
    u += update[0];
    v += update[1];

    
    mean_diff += update[2];//光度补偿

#if SUBPIX_VERBOSE
    cout << "Iter " << iter << ":"
         << "\t u=" << u << ", v=" << v
         << "\t update = " << update[0] << ", " << update[1]
//         << "\t new chi2 = " << new_chi2 << endl;
#endif

    if(update[0]*update[0]+update[1]*update[1] < min_update_squared)    // < 0.03*0.03 --> 收敛
    {
#if SUBPIX_VERBOSE
      cout << "converged." << endl;
#endif
      converged=true;   //成功收敛
      break;
    }
  }

  cur_px_estimate << u, v;
  return converged;
}

void LidarSelector::FeatureAlignment(cv::Mat img)
{
    // 获取特征点的总数
    int total_points = sub_sparse_map->index.size();
    // 如果没有特征点，直接返回
    if (total_points==0) return;
    
    // 初始化对齐标志数组，全部设置为0
    memset(align_flag, 0, length);

    // 成功对齐的特征点计数
    int FeatureAlignmentNum = 0;
       
    // 遍历所有特征点
    for (int i=0; i<total_points; i++) 
    {
        bool res;
        // 获取当前特征点的搜索层级
        int search_level = sub_sparse_map->search_levels[i];
        // 根据搜索层级缩放当前特征点的像素坐标
        Vector2d px_scaled(sub_sparse_map->px_cur[i]/(1<<search_level));
        
        // 对当前特征点进行2D对齐
        res = align2D(
            new_frame_->img_pyr_[search_level],  // 使用金字塔中对应层级的图像
            sub_sparse_map->patch_with_border[i],  // 带边界的特征块
            sub_sparse_map->patch[i],  // 特征块
            20,  // 最大迭代次数
            px_scaled,  // 当前估计的像素坐标（会被更新）
            i  // 特征点索引
        );
        
        // 将对齐后的坐标反缩放到原始图像尺寸
        sub_sparse_map->px_cur[i] = px_scaled * (1<<search_level);
        
        // 如果对齐成功
        if(res)
        {
            // 标记该特征点对齐成功
            align_flag[i] = 1;
            // 增加成功对齐的特征点计数
            FeatureAlignmentNum++;
        }
    }
}

float LidarSelector::UpdateState(cv::Mat img, float total_residual, int level)  //总残差 level --> 金字塔层级
{
    int total_points = sub_sparse_map->index.size();    // 获取特征点的总数
    if (total_points==0) return 0.;
    StatesGroup old_state = (*state);
    V2D pc;         //存储投影坐标
    MD(1,2) Jimg;   //1x2 图像雅可比
    MD(2,3) Jdpi;   //2x3 投影雅可比
    MD(1,3) Jdphi, Jdp, JdR, Jdt; // 定义多个1x3的矩阵，用于各种雅可比计算
    VectorXd z;     // 定义一个动态大小的向量，可能用于存储观测残差
    // VectorXd R;
    bool EKF_end = false;    // EKF结束标志位
    /* Compute J */
    float error=0.0, last_error=total_residual, patch_error=0.0, last_patch_error=0.0, propa_error=0.0;
    // MatrixXd H;
    bool z_init = true;
    const int H_DIM = total_points * patch_size_total;  // 计算H矩阵的维度
    
    // K.resize(H_DIM, H_DIM);
    z.resize(H_DIM);    // 调整z向量的大小
    z.setZero();
    // R.resize(H_DIM);
    // R.setZero();

    // H.resize(H_DIM, DIM_STATE);
    // H.setZero();
    H_sub.resize(H_DIM, 6); //调整H_sub矩阵的大小
    H_sub.setZero();
    
    for (int iteration=0; iteration<NUM_MAX_ITERATIONS; iteration++)    //迭代次数 NUM_MAX_ITERATIONS
    {
        // double t1 = omp_get_wtime();
        double count_outlier = 0;
     
        error = 0.0;
        propa_error = 0.0;
        n_meas_ =0;
        M3D Rwi(state->rot_end);    // 从状态中获取旋转矩阵
        V3D Pwi(state->pos_end);    // 从状态中获取位置向量
        Rcw = Rci * Rwi.transpose();// 计算相机到世界的旋转
        Pcw = -Rci*Rwi.transpose()*Pwi + Pci;// 计算相机到世界的平移
        Jdp_dt = Rci * Rwi.transpose();// 计算位置对平移的雅可比
        
        M3D p_hat;  //反对称矩阵 p^
        int i;

        for (i=0; i<sub_sparse_map->index.size(); i++) 
        {
            patch_error = 0.0;
            int search_level = sub_sparse_map->search_levels[i];    // 获取搜索级别
            int pyramid_level = level + search_level;               // 计算金字塔级别
            const int scale =  (1<<pyramid_level);                  // 计算尺度scale
            //Point -- > PointPtr
            PointPtr pt = sub_sparse_map->voxel_points[i];          // 获取Voxel points

            if(pt==nullptr) continue;

            V3D pf = Rcw * pt->pos_ + Pcw;          // 将点从世界坐标 -> 相机坐标
            pc = cam->world2cam(pf);                // 投影 -> 2D 
            // if((level==2 && iteration==0) || (level==1 && iteration==0) || level==0)
            {
                dpi(pf, Jdpi);                      // 计算投影雅可比
                p_hat << SKEW_SYM_MATRX(pf);        // 计算反对称矩阵
            }
            const float u_ref = pc[0];              //投影得到的 u v
            const float v_ref = pc[1];
            const int u_ref_i = floorf(pc[0]/scale)*scale; //向下取整
            const int v_ref_i = floorf(pc[1]/scale)*scale;
            const float subpix_u_ref = (u_ref-u_ref_i)/scale;//得到的
            const float subpix_v_ref = (v_ref-v_ref_i)/scale;
            const float w_ref_tl = (1.0-subpix_u_ref) * (1.0-subpix_v_ref); //插值计算四个角的权重
            const float w_ref_tr = subpix_u_ref * (1.0-subpix_v_ref);
            const float w_ref_bl = (1.0-subpix_u_ref) * subpix_v_ref;
            const float w_ref_br = subpix_u_ref * subpix_v_ref;
            
            float* P = sub_sparse_map->patch[i];    // 获取当前特征点的patch
            for (int x=0; x<patch_size; x++)     //cur_patch: 当前patch
            //patch_size: patch的大小（像素总数）*/
            {
                // 计算图像指针的起始位置 与之前一样
                uint8_t* img_ptr = (uint8_t*) img.data + (v_ref_i+x*scale-patch_size_half*scale)*width + u_ref_i-patch_size_half*scale;
                for (int y=0; y<patch_size; ++y, img_ptr+=scale) 
                {
                    // if((level==2 && iteration==0) || (level==1 && iteration==0) || level==0)
                    //{
                    // 计算图像梯度
                    float du = 0.5f * ((w_ref_tl*img_ptr[scale] + w_ref_tr*img_ptr[scale*2] + w_ref_bl*img_ptr[scale*width+scale] + w_ref_br*img_ptr[scale*width+scale*2])
                                -(w_ref_tl*img_ptr[-scale] + w_ref_tr*img_ptr[0] + w_ref_bl*img_ptr[scale*width-scale] + w_ref_br*img_ptr[scale*width]));
                    float dv = 0.5f * ((w_ref_tl*img_ptr[scale*width] + w_ref_tr*img_ptr[scale+scale*width] + w_ref_bl*img_ptr[width*scale*2] + w_ref_br*img_ptr[width*scale*2+scale])
                                -(w_ref_tl*img_ptr[-scale*width] + w_ref_tr*img_ptr[-scale*width+scale] + w_ref_bl*img_ptr[0] + w_ref_br*img_ptr[scale]));
                    // 设置图像雅可比 Jimg = dI_du
                    Jimg << du, dv;
                    Jimg = Jimg * (1.0/scale); // 缩放图像雅可比

                    //计算各种雅可比矩阵 -> 见笔记
                    Jdphi = Jimg * Jdpi * p_hat;            // dI_d(theta)
                    Jdp = -Jimg * Jdpi;                     //-dI_dp
                    JdR = Jdphi * Jdphi_dR + Jdp * Jdp_dR;  //第一项表示通过相机姿态变化导致的影响 第二项表示相机旋转通过改变3D点在相机坐标系中的位置间接对图像造成的影响。
                    Jdt = Jdp * Jdp_dt;                     //点 P 相对于相机平移 t 的变化率
                    //}
                    //计算residu
                    double res = w_ref_tl*img_ptr[0] + w_ref_tr*img_ptr[scale] + w_ref_bl*img_ptr[scale*width] + w_ref_br*img_ptr[scale*width+scale]  - P[patch_size_total*level + x*patch_size+y];
                    z(i*patch_size_total+x*patch_size+y) = res;// 存储残差
                    // float weight = 1.0;
                    // if(iteration > 0)
                    //     weight = weight_function_->value(res/weight_scale_); 
                    // R(i*patch_size_total+x*patch_size+y) = weight;       
                    patch_error +=  res*res;// 累加patch误差
                    n_meas_++;// 增加测量计数
                    // H.block<1,6>(i*patch_size_total+x*patch_size+y,0) << JdR*weight, Jdt*weight;
                    // if((level==2 && iteration==0) || (level==1 && iteration==0) || level==0)

                    // 更新H_sub矩阵
                    H_sub.block<1,6>(i*patch_size_total+x*patch_size+y,0) << JdR, Jdt;
                }
            }  

            sub_sparse_map->errors[i] = patch_error;    // 存储每个特征点的误差
            error += patch_error;// 计算总误差
        }

        // computeH += omp_get_wtime() - t1;

        error = error/n_meas_;  //平均

        // double t3 = omp_get_wtime();

        if (error <= last_error) //计算平均误差并与先前误差比较。
        {
            old_state = (*state);
            last_error = error;

            // K = (H.transpose() / img_point_cov * H + state->cov.inverse()).inverse() * H.transpose() / img_point_cov;
            // auto vec = (*state_propagat) - (*state);
            // G = K*H;
            // (*state) += (-K*z + vec - G*vec);

            auto &&H_sub_T = H_sub.transpose(); 
            H_T_H.block<6,6>(0,0) = H_sub_T * H_sub;// HTH

            //state->cov 当前状态的协方差矩阵   img_point_cov = 100 图像点的协方差（观测噪声）
            MD(DIM_STATE, DIM_STATE) &&K_1 = (H_T_H + (state->cov / img_point_cov).inverse()).inverse();
            //观测残差（HTz）和预测误差（vec）
            auto &&HTz = H_sub_T * z;
            // K = K_1.block<DIM_STATE,6>(0,0) * H_sub_T;

            //计算状态预测误差
            auto vec = (*state_propagat) - (*state);

            //计算增益矩阵 G 只关注状态的前6个维度（对应R和T）
            G.block<DIM_STATE,6>(0,0) = K_1.block<DIM_STATE,6>(0,0) * H_T_H.block<6,6>(0,0);

            //计算状态更新
            auto solution = - K_1.block<DIM_STATE,6>(0,0) * HTz + vec - G.block<DIM_STATE,6>(0,0) * vec.block<6,1>(0,0);

            (*state) += solution;//将计算得到的更新应用到当前状态

            //提取旋转和平移更新
            auto &&rot_add = solution.block<3,1>(0,0);
            auto &&t_add   = solution.block<3,1>(3,0);

            if ((rot_add.norm() * 57.3f < 0.001f) && (t_add.norm() * 100.0f < 0.001f))
            {
                EKF_end = true;
            }
        }
        else
        {
            (*state) = old_state;
            EKF_end = true;
        }

        // ekf_time += omp_get_wtime() - t3;

        if (iteration==NUM_MAX_ITERATIONS || EKF_end) 
        {
            break;
        }
    }
    return last_error;
} 

//计算当前帧相对于世界坐标系的位置和姿态
void LidarSelector::updateFrameState(StatesGroup state)
{
    // 结束时刻的旋转和位置信息
    // 将结束时的姿态转换为旋转矩阵 Rwi 和位置向量 Pwi
    M3D Rwi(state.rot_end);//是世界坐标系到 IMU
    V3D Pwi(state.pos_end);

    Rcw = Rci * Rwi.transpose();    //相机坐标系到世界坐标系的旋转和平移
    Pcw = -Rci*Rwi.transpose()*Pwi + Pci;

    //最终 Rcw Pcw 表示相机在世界坐标系中的位姿
    // 将旋转矩阵和位置向量组合成SE(3)表示的变换矩阵，设置为新帧的状态
    new_frame_->T_f_w_ = SE3(Rcw, Pcw);
}

/* 添加新的特征观测的过程
1.遍历所有3D点
2.将3D点投影到当前帧图像平面
3.判断是否需要添加新的观测（基于位姿变化、像素距离等条件）
4.如果需要添加新观测，则提取图像patch，计算特征得分，并创建新的特征观测
5.维护每个3D点的观测数量，保持在一个合理的范围内*/
void LidarSelector::addObservation(cv::Mat img)
{
    // 获取稀疏地图中点的总数
    int total_points = sub_sparse_map->index.size();
    if (total_points==0) return;  // 如果没有点，直接返回

    for (int i=0; i<total_points; i++) 
    {
        // 获取当前处理的3D点
        PointPtr pt = sub_sparse_map->voxel_points[i];
        if(pt==nullptr) continue;  // 如果点为空，跳过

        // 将3D点投影到当前帧的图像平面上
        V2D pc(new_frame_->w2c(pt->pos_));
        // 获取当前帧的位姿
        SE3 pose_cur = new_frame_->T_f_w_;
        bool add_flag = false;  // 是否添加新观测的标志

        // 以下注释掉的条件可能是用于筛选低误差的点
        // if (sub_sparse_map->errors[i]<= 100*patch_size_total && sub_sparse_map->errors[i]>0)
        {
            // 为当前点提取图像patch
            float* patch_temp = new float[patch_size_total*3];
            getpatch(img, pc, patch_temp, 0);  // 提取R通道patch
            getpatch(img, pc, patch_temp, 1);  // 提取G通道patch
            getpatch(img, pc, patch_temp, 2);  // 提取B通道patch

            // 获取该点的最后一次观测
            FeaturePtr last_feature =  pt->obs_.back();

            // 计算当前帧与上一次观测帧之间的位姿变化
            SE3 pose_ref = last_feature->T_f_w_;
            SE3 delta_pose = pose_ref * pose_cur.inverse();
            double delta_p = delta_pose.translation().norm();  // 位置变化
            // 计算旋转角度变化
            double delta_theta = (delta_pose.rotation_matrix().trace() > 3.0 - 1e-6) ? 0.0 : 
                                  std::acos(0.5 * (delta_pose.rotation_matrix().trace() - 1));            

            // 如果位置变化大于0.5或旋转角度变化大于10度，标记为添加新观测
            if(delta_p > 0.5 || delta_theta > 10) add_flag = true;

            // 计算图像平面上的像素距离
            Vector2d last_px = last_feature->px;
            double pixel_dist = (pc-last_px).norm();
            // 如果像素距离大于40，标记为添加新观测
            if(pixel_dist > 40) add_flag = true;
            
            // 限制每个3D点的观测数量不超过20
            if(pt->obs_.size()>=20)
            {
                FeaturePtr ref_ftr;
                // 删除视角最远的观测
                pt->getFurthestViewObs(new_frame_->pos(), ref_ftr);
                pt->deleteFeatureRef(ref_ftr);
            } 

            // 如果满足添加新观测的条件
            if(add_flag)
            {
                // 计算特征点的Shi-Tomasi得分
                pt->value = vk::shiTomasiScore(img, pc[0], pc[1]);
                // 将2D点转换为单位向量
                Vector3d f = cam->cam2world(pc);
                // 创建新的特征观测
                FeaturePtr ftr_new(new Feature(patch_temp, pc, f, new_frame_->T_f_w_, pt->value, sub_sparse_map->search_levels[i])); 
                ftr_new->img = new_frame_->img_pyr_[0];
                ftr_new->id_ = new_frame_->id_;
                // 将新的观测添加到3D点的观测列表中
                pt->addFrameRef(ftr_new);      
            }
        }
    }
}

void LidarSelector::ComputeJ(cv::Mat img) 
{
    //计算雅可比矩阵
    int total_points = sub_sparse_map->index.size();
    if (total_points==0) return;
    float error = 1e10;
    float now_error = error;

    for (int level=2; level>=0; level--) 
    {
        now_error = UpdateState(img, error, level); //主要计算雅可比的过程
    }
    if (now_error < error)
    {
        state->cov -= G*state->cov;     //更新协方差矩阵
    }
    updateFrameState(*state);
}


//这个函数的主要目的是在图像上可视化显示SLAM系统中的关键patch，并显示系统的运行频率
void LidarSelector::display_keypatch(double time)
{
    int total_points = sub_sparse_map->index.size();    //获取稀疏地图中的点的总数
    if (total_points==0) return;
    for(int i=0; i<total_points; i++)   //遍历所有关键点
    {
        PointPtr pt = sub_sparse_map->voxel_points[i];  
        V2D pc(new_frame_->w2c(pt->pos_));              //使用 w2c 函数将3D点从世界坐标系投影到相机坐标系。
        cv::Point2f pf;
        pf = cv::Point2f(pc[0], pc[1]);                 //转换为OpenCV的2D点格式
        if (sub_sparse_map->errors[i]<8000) // 5.5      //基于点的误差值选择颜色
            cv::circle(img_cp, pf, 6, cv::Scalar(0, 255, 0), -1, 8); //误差小于8000的点用绿色表示
        else
            cv::circle(img_cp, pf, 6, cv::Scalar(255, 0, 0), -1, 8); // 其他点用蓝色表示 在图像上画一个半径为6的实心圆
    }   
    std::string text = std::to_string(int(1/time))+" HZ";           //计算并格式化系统运行频率
    cv::Point2f origin;
    origin.x = 20;
    origin.y = 20;
    cv::putText(img_cp, text, origin, cv::FONT_HERSHEY_COMPLEX, 0.6, cv::Scalar(255, 255, 255), 1, 8, 0);
    //在图像左上角（坐标20,20）显示频率文本
}

V3F LidarSelector::getpixel(cv::Mat img, V2D pc) 
{
    // 获取浮点数坐标
    const float u_ref = pc[0];  // x坐标
    const float v_ref = pc[1];  // y坐标

    // 计算整数坐标（向下取整）
    const int u_ref_i = floorf(pc[0]);  // x的整数部分
    const int v_ref_i = floorf(pc[1]);  // y的整数部分

    // 计算亚像素偏移
    const float subpix_u_ref = (u_ref-u_ref_i);  // x的小数部分
    const float subpix_v_ref = (v_ref-v_ref_i);  // y的小数部分

    // 计算双线性插值的四个权重
    const float w_ref_tl = (1.0-subpix_u_ref) * (1.0-subpix_v_ref);  // 左上角权重
    const float w_ref_tr = subpix_u_ref * (1.0-subpix_v_ref);        // 右上角权重
    const float w_ref_bl = (1.0-subpix_u_ref) * subpix_v_ref;        // 左下角权重
    const float w_ref_br = subpix_u_ref * subpix_v_ref;              // 右下角权重

    // 计算图像数据的起始指针位置
    uint8_t* img_ptr = (uint8_t*) img.data + ((v_ref_i)*width + (u_ref_i))*3;

    // 对B、G、R三个通道分别进行双线性插值
    // B通道
    float B = w_ref_tl*img_ptr[0] + w_ref_tr*img_ptr[0+3] + 
              w_ref_bl*img_ptr[width*3] + w_ref_br*img_ptr[width*3+0+3];
    // G通道
    float G = w_ref_tl*img_ptr[1] + w_ref_tr*img_ptr[1+3] + 
              w_ref_bl*img_ptr[1+width*3] + w_ref_br*img_ptr[width*3+1+3];
    // R通道
    float R = w_ref_tl*img_ptr[2] + w_ref_tr*img_ptr[2+3] + 
              w_ref_bl*img_ptr[2+width*3] + w_ref_br*img_ptr[width*3+2+3];

    // 创建并返回包含插值结果的三维向量
    V3F pixel(B,G,R);
    return pixel;
}

//最主要的实现函数 以及统计计算时间
void LidarSelector::detect(cv::Mat img, PointCloudXYZI::Ptr pg) 
{
    if(width!=img.cols || height!=img.rows)         //图像尺寸不匹配
    {
        // std::cout<<"Resize the img scale !!!"<<std::endl;
        double scale = 0.5;
        cv::resize(img,img,cv::Size(img.cols*scale,img.rows*scale),0,0,CV_INTER_LINEAR);    //将图片img缩小至原尺寸的50%
    }
    img_rgb = img.clone();
    img_cp = img.clone();
    cv::cvtColor(img,img,CV_BGR2GRAY);              //转化为灰度图

    new_frame_.reset(new Frame(cam, img.clone()));  //克隆图像并使用相机参数，创建并重置一个新的帧对象
    updateFrameState(*state);                       //更新完新的状态（R T）

    if(stage_ == STAGE_FIRST_FRAME && pg->size()>10)
    {
        new_frame_->setKeyframe();                  //设置关键帧
        stage_ = STAGE_DEFAULT_FRAME;               //切换标志位
    }

    double t1 = omp_get_wtime();

    addFromSparseMap(img, pg);                      //稀疏地图

    double t3 = omp_get_wtime();

    addSparseMap(img, pg);

    double t4 = omp_get_wtime();
    
    // computeH = ekf_time = 0.0;
    
    ComputeJ(img);//计算雅可比

    double t5 = omp_get_wtime();

    addObservation(img);
    
    double t2 = omp_get_wtime();
    
    frame_count ++;
    ave_total = ave_total * (frame_count - 1) / frame_count + (t2 - t1) / frame_count;
    
    printf("[ VIO ]: time: addFromSparseMap: %.6f addSparseMap: %.6f ComputeJ: %.6f addObservation: %.6f total time: %.6f ave_total: %.6f.\n"
    , t3-t1, t4-t3, t5-t4, t2-t5, t2-t1, ave_total);

    display_keypatch(t2-t1);
} 

} // namespace lidar_selection