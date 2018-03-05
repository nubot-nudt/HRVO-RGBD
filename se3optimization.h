#ifndef SE3OPTIMIZATION_H
#define SE3OPTIMIZATION_H

//#include <pcl/point_cloud.h>
//#include <pcl/point_types.h>

#include "Thirdparty/g2o/g2o/core/sparse_optimizer.h"
#include "Thirdparty/g2o/g2o/core/block_solver.h"
#include "Thirdparty/g2o/g2o/core/solver.h"
#include "Thirdparty/g2o/g2o/core/robust_kernel_impl.h"
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_levenberg.h"
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_gauss_newton.h"
#include "Thirdparty/g2o/g2o/core/base_unary_edge.h"
#include "Thirdparty/g2o/g2o/core/hyper_graph_action.h"
#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"
#include "Thirdparty/g2o/g2o/types/types_seven_dof_expmap.h"
//#include "Thirdparty/g2o/g2o/solvers/cholmod/linear_solver_cholmod.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_dense.h"
//#include "ThirdParty/g2o/g2o/solvers/structure_only/structure_only_solver.h"

#include <opencv2/core/core.hpp>
#include "opencv2/imgproc/imgproc.hpp"
namespace Eigen
{
    typedef Matrix<float, 6, 1> Vector6f;
    typedef Matrix<float, 6, 6> Matrix6f;
}
namespace cvExtension
{
/// sparse mat processing functions.
/// WARRNING: the invalid pixels: 0 or nan
void sparseGaussianBlur(const cv::Mat &_src, CV_OUT cv::Mat &_dst, const cv::Size &_ksize, const double &_sigma );///Warning: less efficient
void sparseImageGradient(const cv::Mat &_image32f, CV_OUT cv::Mat &_grad_grayX, CV_OUT cv::Mat &_grad_grayY );
inline float getSparseSubPixel(const cv::Mat &image, const float &x, const float &y);
std::vector<cv::Point> extractSalientPixels(const cv::Mat &_grad2, const int _max_num=1000);
}

static inline Eigen::Isometry3f se3toSE3 (const Eigen::Vector6f& se3 )
{
    Eigen::Isometry3f iso_delta;
    Eigen::Vector3f rotat_part = se3.head<3>();
    const float angle = rotat_part.norm();
    if( angle==0 )
        iso_delta.linear() = Eigen::Matrix3f::Identity();
    else
    {
        rotat_part.normalize();
        iso_delta.linear() = Eigen::AngleAxisf(angle,rotat_part).toRotationMatrix();
    }
    iso_delta.translation() = se3.tail<3>();
    return iso_delta;
}

// project a 3d point into an image plane, the error is photometric error
// an unary edge with one vertex SE3Expmap (the pose of camera)
class EdgePhotometic: public g2o::BaseUnaryEdge< 1, double, g2o::VertexSE3Expmap>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
//    EdgePhotometic() {}
    EdgePhotometic ( const Eigen::Vector3d &point, const cv::Mat* const&p_image )
        : x_world_ ( point ), p_image_ ( p_image )
    {}
    void setCameraModel(const float&_fx, const float&_fy, const float&_cx, const float&_cy ){fx_=_fx,fy_=_fy,cx_=_cx,cy_=_cy;}
    void setGradientImage( const cv::Mat* const&p_gradX, const cv::Mat* const&p_gradY ){ p_gradX_=p_gradX , p_gradY_=p_gradY ; }
    virtual void computeError();
    virtual void linearizeOplus();
    virtual bool read ( std::istream& in ) {return false;}
    virtual bool write ( std::ostream& out ) const {return false;}
protected:
    const Eigen::Vector3d x_world_;   // 3D point in world frame
    double fx_ = 535.4, cx_ = 320.1, fy_ = 539.2, cy_ = 247.6;//kinect1
    const cv::Mat* p_image_=nullptr;    // reference image
    const cv::Mat* p_gradX_=nullptr;    // reference image
    const cv::Mat* p_gradY_=nullptr;    // reference image
};
class EdgeInverseDepth: public EdgePhotometic
{
    float SENSE_RANGE_MIN = 0.4;
    float SENSE_RANGE_MAX = 10;
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EdgeInverseDepth( const Eigen::Vector3d &point, const cv::Mat* const&p_image )
        : EdgePhotometic( point, p_image )
    {}
    void setSenseRange(const float&_min_depth, const float&_max_depth ){SENSE_RANGE_MIN=_min_depth,SENSE_RANGE_MAX=_max_depth;}
    virtual void computeError();//几何信息与光度信息的使用方法几乎一模一样，只是残差计算不同。光度学残差的参考值是一个恒定的亮度值，而几何残差的参考值是随优化位姿变化的
    virtual void linearizeOplus();//相比于光度学，多了一个逆深度随位姿变化的影响
};
//class PreIterationAction : public g2o::HyperGraphAction
//{
//  public:
//    virtual HyperGraphAction* operator()(const g2o::HyperGraph* graph, Parameters* parameters = 0)
//    {
//      std::cout << "PreIterationAction called!!" << std::endl;
//      return this;
//    }
//};
//class ComputeErrorAction : public g2o::HyperGraphAction
//{
//  public:
//    virtual HyperGraphAction* operator()(const g2o::HyperGraph* graph, Parameters* parameters = 0)
//    {
//      std::cout << "ComputeErrorAction called!!" << std::endl;
//      return this;
//    }
//};
/*
class SE3Optimization
{
public:
    float OPTI_PRECISION = 0.002;
    float WEIGHT_PHO2GEO = 1.482;
    SE3Optimization();
    void setCameraIntrinsic(const float &_fx, const float &_cx, const float &_fy, const float &_cy );

    void init( const std::vector<pcl::PointXYZI, Eigen::aligned_allocator<pcl::PointXYZI> > &_map_pho_pts,  const std::vector<pcl::PointXYZ, Eigen::aligned_allocator<pcl::PointXYZ> > &_map_geo_pts, const cv::Mat &_gray_img8U, const cv::Mat &_depth_img );
    Eigen::Vector6f optimizeOnce( Eigen::Isometry3f &_init_iso );//return error
    bool optimize(  Eigen::Isometry3f &_init_iso );

    void drawPoints(cv::Mat &_image, const Isometry3f &_iso,
                       std::vector<pcl::PointXYZI, Eigen::aligned_allocator<pcl::PointXYZI> > _pho_pts,
                       std::vector<pcl::PointXYZ, Eigen::aligned_allocator<pcl::PointXYZ> > _geo_pts ) const;
    void drawMapPoints(cv::Mat &_image, const Isometry3f &_iso = Isometry3f::Identity() ) const
    {
        drawPoints( _image, _iso, pho_pts_, geo_pts_ );
    }

    std::vector<pcl::PointXYZI, Eigen::aligned_allocator<pcl::PointXYZI> > pho_pts_;//photometric points
    std::vector<pcl::PointXYZ, Eigen::aligned_allocator<pcl::PointXYZ> > geo_pts_;//geometric points
    cv::Mat gray32f_;//gray image of the current frame in FLOAT format (range: 0~255)
    cv::Mat rgb_show;// show the gray image
    cv::Mat depth32f_, inv_depth32f_;// unit: mm.
    cv::Mat grad_grayX_, grad_grayY_, grad_inv_depthX_, grad_inv_depthY_;
    int width_, height_;
    std::vector<float> errors_pho_, error2s_pho_, weights_pho_;
    std::vector<float> errors_geo_, error2s_geo_, weights_geo_;
    std::vector<Eigen::Vector6f, Eigen::aligned_allocator<Eigen::Vector6f > > jacobian_pho_, jacobian_geo_;
    Eigen::Vector6f generateJacobian( const Eigen::Vector3f &_pt_cam, const Eigen::Vector2f &_pt_img, const cv::Mat &_gradX, const cv::Mat &_gradY ) const;
    double pho_gain_, brightness_;
public:
    double fx_ = 535.4, cx_ = 320.1, fy_ = 539.2, cy_ = 247.6;//kinect1
    double _1_fx, _1_fy;
    g2o::SparseOptimizer optimizer;
};

inline float getSubPixel(const cv::Mat &image, const float &x, const float &y);

static inline pcl::PointXYZI operator * (const Eigen::Matrix<float,3,3>& m, const pcl::PointXYZI& pt)
{
    pcl::PointXYZI pt_return(pt);
    Eigen::Map<Eigen::Vector3f> eig_pt( (float*)pt_return.data );// share the same memory
    eig_pt = m*eig_pt;
    return pt_return;
}
static inline pcl::PointXYZ operator * (const Eigen::Matrix<float,3,3>& m, const pcl::PointXYZ& pt)
{
    pcl::PointXYZ pt_return(pt);
    Eigen::Map<Eigen::Vector3f> eig_pt( (float*)pt_return.data );// share the same memory
    eig_pt = m*eig_pt;
    return pt_return;
}
static inline pcl::PointXYZI operator * (const Eigen::Isometry3f& m, const pcl::PointXYZI& pt)
{
    pcl::PointXYZI pt_return(pt);
    Eigen::Map<Eigen::Vector3f> eig_pt( (float*)pt_return.data );// share the same memory
    eig_pt = m*eig_pt;
    return pt_return;
}
static inline pcl::PointXYZ operator * (const Eigen::Isometry3f& m, const pcl::PointXYZ& pt)
{
    pcl::PointXYZ pt_return(pt);
    Eigen::Map<Eigen::Vector3f> eig_pt( (float*)pt_return.data );// share the same memory
    eig_pt = m*eig_pt;
    return pt_return;
}

class PostIterationAction : public g2o::HyperGraphAction
{
  public:
    PostIterationAction( SE3Optimization* _p_opti ) : p_opti(_p_opti){}
    virtual HyperGraphAction* operator()(const g2o::HyperGraph* graph, Parameters* parameters = 0);
    SE3Optimization* p_opti;
};*/
#endif // SE3OPTIMIZATION_H
