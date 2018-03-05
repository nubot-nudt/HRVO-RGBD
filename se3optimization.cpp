#include "se3optimization.h"

inline float
cvExtension::getSparseSubPixel(const cv::Mat &image, const float &x, const float &y)
{
    assert( image.channels()==1 );
    if( x<0 || x>image.cols-1 || y<0 || y>image.rows-1 )
        return std::numeric_limits<float>::quiet_NaN();

    int l = floor(x), r = ceil(x), u = floor(y), d = ceil(y);
    uchar *data_ul = image.data + u*image.step[0] + l*image.step[1];
    uchar *data_dl = data_ul + image.step[0];
    uchar *data_ur = data_ul + image.step[1];
    uchar *data_dr = data_ur + image.step[0];
    float v_ul, v_dl, v_ur, v_dr;
    switch ( image.type() )
    {
    case CV_32FC1:
        v_ul = *(float*)data_ul;
        v_dl = *(float*)data_dl;
        v_ur = *(float*)data_ur;
        v_dr = *(float*)data_dr;
        break;
    case CV_16UC1:
        v_ul = *(int16_t*)data_ul;
        v_dl = *(int16_t*)data_dl;
        v_ur = *(int16_t*)data_ur;
        v_dr = *(int16_t*)data_dr;
        break;
    case CV_8UC1:
        v_ul = *(uchar*)data_ul;
        v_dl = *(uchar*)data_dl;
        v_ur = *(uchar*)data_ur;
        v_dr = *(uchar*)data_dr;
        break;
    default:
        std::cerr << "Unknown image type for getSubPixel()! =" << image.type() << std::endl;
        std::cerr << image.rows << "x" << image.cols << "  cn=" << image.channels() << std::endl;
        return std::numeric_limits<float>::quiet_NaN();
        break;
    }
    float v_l, v_r;
    if( d!=u )
    {
        if     ( std::isnan(v_dl) || v_dl==0 ) v_l = v_ul;
        else if( std::isnan(v_ul) || v_ul==0 ) v_l = v_dl;
        else                                   v_l = v_ul*(d-y) + v_dl*(y-u);
        if     ( std::isnan(v_dr) || v_dr==0 ) v_r = v_ur;
        else if( std::isnan(v_ur) || v_ur==0 ) v_r = v_dr;
        else                                   v_r = v_ur*(d-y) + v_dr*(y-u);
    }
    else
        v_l = v_ul, v_r = v_ur;
    if( l!=r )
    {
        if     ( std::isnan(v_l) || v_l==0 ) return v_r;
        else if( std::isnan(v_r) || v_r==0 ) return v_l;
        else                                 return v_l*(r-x) + v_r*(x-l);
    }
    else
        return v_r;
}
void cvExtension::sparseGaussianBlur(const cv::Mat &_src, cv::Mat &_dst, const cv::Size &_ksize, const double &_sigma )//less efficient
{
    assert( _src.type()==CV_32FC1 || _src.type()==CV_64FC1 );

    cv::Mat blur;
    cv::GaussianBlur( _src,blur,_ksize,_sigma );

    cv::Mat weight = cv::abs( _src );
    cv::threshold( weight, weight, 0, 1, CV_THRESH_BINARY );
    cv::GaussianBlur( weight,weight,_ksize,_sigma );

    cv::divide( blur, weight, _dst );

    /*
    assert( _ksize%2==1 && _sigma>0 );
    assert( width>0 && height>0 && src.channels()==1 );
    cv::Mat src32F;
    _src.convertTo( src32F, CV_32F );
    const int width = src32F.cols;
    const int height= src32F.rows;

    static cv::Mat kernel = cv::getGaussianKernel( _ksize, _sigma, CV_64F );//高斯核是对称的
    double *kernel_data = (double *)kernel.data;
    int kr = _ksize/2;
    std::vector<float> pixel_k( kr+1, 0 );
    cv::Mat integrate = cv::Mat::zeros( height, width, CV_32FC1 );
    cv::Mat weight    = cv::Mat::zeros( height, width, CV_32FC1 );
    ///blur X direction
    uchar * row_src = src32F.data;
    for(int h=0; h<height; h++)
    {
        uchar *p_src = (uchar *)row_src;
        row_src += src32F.step[0];
        for( int w=0; w<width; w++ )
        {
            float value = *(float*)p_src;
            p_src += src32F.step[1];
            if( std::isnan(value) || value==0 )
                continue;

            for( int i=0; i<=kr; i++ )
                pixel_k[kr-i] = value * kernel_data[i];

            int w1 = std::max( 0, w-kr ), w2 = std::min( width-1, w+kr );
            uchar *p_int = integrate.data + h*integrate.step[0] + w1*integrate.step[1];
            uchar *p_wei =    weight.data + h*   weight.step[0] + w1*   weight.step[1];
            for( int j=w1; j<=w2; j++)
            {
                int r = abs(j-w);
                *(float*)p_int += pixel_k[r];
                *(float*)p_wei += kernel_data[r];
                p_int += integrate.step[1];
                p_wei +=    weight.step[1];
            }
        }
    }
    /// over the weight
    uchar * row_int = integrate.data;
    uchar * row_wei = weight.data;
    for(int h=0; h<height; h++)
    {
        float *p_int = (float *)row_int;
        float *p_wei = (float *)row_wei;
        row_int += integrate.step[0];
        row_wei += weight.step[0];
        for( int w=0; w<width; w++ )
        {
            if( *p_wei!=0 ) *p_int /= *p_wei;
            p_int ++;
            p_wei ++;
        }
    }
    ///blur Y direction
    src32F = integrate.clone();
    integrate.setTo(0.0f);
    weight.setTo(0.0f);
    uchar * col_src = src32F.data;
    for(int w=0; w<width; w++)
    {
        uchar *p_src = (uchar *)col_src;
        col_src += src32F.step[1];
        for( int h=0; h<height; h++ )
        {
            float value = *(float*)p_src;
            p_src += src32F.step[0];
            if( std::isnan(value) || value==0 )
                continue;

            for( int i=0; i<=kr; i++ )
                pixel_k[kr-i] = value * kernel_data[i];

            int h1 = std::max( 0, h-kr ), h2 = std::min( height-1, h+kr );
            uchar *p_int = integrate.data + h1*integrate.step[0] + w*integrate.step[1];
            uchar *p_wei =    weight.data + h1*   weight.step[0] + w*   weight.step[1];
            for( int j=h1; j<=h2; j++)
            {
                int r = abs(j-w);
                *(float*)p_int += pixel_k[r];
                *(float*)p_wei += kernel_data[r];
                p_int += integrate.step[0];
                p_wei +=    weight.step[0];
            }
        }
    }
    /// over the weight
    row_int = integrate.data;
    row_wei = weight.data;
    for(int h=0; h<height; h++)
    {
        float *p_int = (float *)row_int;
        float *p_wei = (float *)row_wei;
        row_int += integrate.step[0];
        row_wei += weight.step[0];
        for( int w=0; w<width; w++ )
        {
            if( *p_wei!=0 ) *p_int /= *p_wei;
            p_int ++;
            p_wei ++;
        }
    }
    integrate.convertTo( _dst, _src.type() );
*/
}
void cvExtension::sparseImageGradient(const cv::Mat &_image32f, CV_OUT cv::Mat &_grad_grayX, CV_OUT cv::Mat &_grad_grayY)
{
    assert( _image32f.type()==CV_32FC1  );
    /// calculate the gradient of the image, with the hypothesis that the image is sparse
    const int height = _image32f.rows;
    const int width  = _image32f.cols;
    _grad_grayX = cv::Mat::zeros( height, width, CV_32FC1 );
    _grad_grayY = cv::Mat::zeros( height, width, CV_32FC1 );
    uchar * row_src = _image32f.data;
    const int step0_id = _image32f.step[0] / 4;//float step
    uchar * row_gx = _grad_grayX.data;
    uchar * row_gy = _grad_grayY.data;
    for(int h=0; h<height; h++)
    {
        float *p_src = (float *)row_src;
        float *p_gx = (float *)row_gx;
        float *p_gy = (float *)row_gy;
        row_src += _image32f.step[0];
        row_gx += _grad_grayX.step[0];
        row_gy += _grad_grayY.step[0];
        for( int w=0; w<width; w++ )
        {
            float l, r, u, d;
            if( w==0 )            l = *p_src,                r = *(p_src+1);
            else if( w==width-1 ) l = *(p_src-1),            r = *p_src;
            else                  l = *(p_src-1)/2.0,        r = *(p_src+1)/2.0;
            if( h==0 )            u = * p_src,               d = *(p_src+step0_id);
            else if( h==height-1 )u = *(p_src-step0_id),     d = * p_src;
            else                  u = *(p_src-step0_id)/2.0, d = *(p_src+step0_id)/2.0;
            if( !(std::isnan(l)||l==0) && !(std::isnan(r)||r==0) ) *p_gx = r - l;
            else                                                   *p_gx = 0.0;
            if( !(std::isnan(u)||u==0) && !(std::isnan(d)||d==0) ) *p_gy = d - u;
            else             *p_gy = 0.0;
            p_src ++;
            p_gx ++;
            p_gy ++;
        }
    }
}

std::vector<cv::Point> cvExtension::extractSalientPixels(const cv::Mat &_grad2, const int _max_num)
{
    assert(_grad2.type()==32FC1);
    double minValue = 0;
    double maxValue = 0;
    cv::minMaxLoc(_grad2, &minValue, &maxValue, 0, 0); //找到直方图中的最大值和最小值

    const int size = 1000;//直方图横坐标的区间数 即横坐标被分成多少份
    float value_range[2] = { 0, (float)(maxValue*1.0001) };
    const float *ranges[1] = { value_range };   // 输入图像每个通道的像素的值域(这里需要为const类型)
    int channels = 0;//图像的通道 灰度图的通道数为0
    cv::Mat hist;//得到的直方图
    cv::calcHist(&_grad2, 1, &channels, cv::Mat(), hist, 1,//得到的直方图的维数 灰度图的维数为1
                 &size, ranges);

    double threshold = 0;
    float num = 0;
    for( int i=1; i<=size; i++ )
    {
        float &bin_value = hist.at<float>(hist.rows-i);
        if( num+bin_value > _max_num )
        {
            threshold = (value_range[1]-value_range[0]) * (size-i+1) / size;
            break;
        }
        else
            num += bin_value;
    }

    std::vector<cv::Point> salient_pts;
    salient_pts.reserve( num );
    uchar *prow = _grad2.data;
    for(int row=0; row<_grad2.rows; row++ )
    {
        float * pcol = (float*) prow;
        for(int col=0; col<_grad2.cols; col++)
        {
            if( *pcol >= threshold )
                salient_pts.push_back( cv::Point(col,row) );
            pcol++;
        }
        prow += _grad2.step[0];
    }

    return salient_pts;
}

void EdgePhotometic::computeError()
{
    const g2o::VertexSE3Expmap* v_se3  =static_cast<const g2o::VertexSE3Expmap*> ( _vertices[0] );
    const Eigen::Vector3d pt_cam = v_se3->estimate().map ( x_world_ );
    if( std::isnan(pt_cam[2]) || pt_cam[2]<0 )
    {
        _error ( 0,0 ) = 0.0;
        this->setLevel ( 1 );
        return;
    }
    const Eigen::Vector2d pt_img( pt_cam[0]/pt_cam[2]*fx_+cx_, pt_cam[1]/pt_cam[2]*fy_+cy_ );
    if( pt_img[0]<0 || pt_img[0]>p_image_->cols-1 || pt_img[1]<0 || pt_img[1]>p_image_->rows-1 )
    {
        _error ( 0,0 ) = 0.0;
        this->setLevel ( 1 );
        return;
    }
    float value = cvExtension::getSparseSubPixel(*p_image_,pt_img[0],pt_img[1]);
    if( std::isnan(value) || value==0 )// invalid measurement
    {
        _error ( 0,0 ) = 0.0;
        this->setLevel ( 1 );
        return;
    }
    _error ( 0,0 ) = value - _measurement;
//    std::cout << "pho err=" << _error << std::endl;
}
void EdgeInverseDepth::computeError()
{
    const g2o::VertexSE3Expmap* v_se3  =static_cast<const g2o::VertexSE3Expmap*> ( _vertices[0] );
    const Eigen::Vector3d pt_cam = v_se3->estimate().map ( x_world_ );
    if( std::isnan(pt_cam[2]) || pt_cam[2]<SENSE_RANGE_MIN || pt_cam[2]>SENSE_RANGE_MAX )
    {
        _error ( 0,0 ) = 0;
        this->setLevel ( 1 );
        return;
    }
    _measurement = 1.0 / pt_cam[2]; //the only difference with EdgePhotometic here
    const Eigen::Vector2d pt_img( pt_cam[0]/pt_cam[2]*fx_+cx_, pt_cam[1]/pt_cam[2]*fy_+cy_ );
    if( pt_img[0]<0 || pt_img[0]>p_image_->cols-1 || pt_img[1]<0 || pt_img[1]>p_image_->rows-1 )
    {
        _error ( 0,0 ) = 0;
        this->setLevel ( 1 );
        return;
    }
    float value = cvExtension::getSparseSubPixel(*p_image_,pt_img[0],pt_img[1]);
    if( std::isnan(value) || value==0 )// invalid measurement
    {
        _error ( 0,0 ) = 0;
        this->setLevel ( 1 );
        return;
    }
    _error ( 0,0 ) = value - _measurement;
//    std::cout << "depth error=" << value << "-" << _measurement << "=" << _error(0,0) << std::endl;
}
void EdgePhotometic::linearizeOplus( )
{
    if ( level() == 1 )
    {
        _jacobianOplusXi = Eigen::Matrix<double, 1, 6>::Zero();
        return;
    }
    g2o::VertexSE3Expmap* v_se3 = static_cast<g2o::VertexSE3Expmap*> ( _vertices[0] );
    Eigen::Vector3d _pt_cam = v_se3->estimate().map ( x_world_ );
    const double &x = _pt_cam[0], &y = _pt_cam[1], &z = _pt_cam[2];

    const double _1_z = 1.0/z, _1_zz = 1.0/(z*z);
    const double x_zz = x*_1_zz, y_zz = y*_1_zz;
    /// the se3 is in (\omega, \epsilon) formt, where \omega is so(3) and \epsilon is translation
    Eigen::Matrix<double,2,6> jac_CamNor_ksai;
    jac_CamNor_ksai( 0,0 ) = -y*x_zz;
    jac_CamNor_ksai( 0,1 ) = 1+( x*x_zz );
    jac_CamNor_ksai( 0,2 ) = -y*_1_z;
    jac_CamNor_ksai( 0,3 ) = _1_z;
    jac_CamNor_ksai( 0,4 ) = 0;
    jac_CamNor_ksai( 0,5 ) = -x_zz;
    jac_CamNor_ksai( 1,0 ) = - ( 1+y*y_zz );
    jac_CamNor_ksai( 1,1 ) = x*y_zz;
    jac_CamNor_ksai( 1,2 ) = x*_1_z;
    jac_CamNor_ksai( 1,3 ) = 0;
    jac_CamNor_ksai( 1,4 ) = _1_z;
    jac_CamNor_ksai( 1,5 ) = -y_zz;
    //上面计算的是地图三维点经过坐标变换，在图像上的像素位置变化

    Eigen::Matrix<double,1,2> jac_pixel_CamNor;
    float u = x*fx_*_1_z + cx_;
    float v = y*fy_*_1_z + cy_;
    const float gx = cvExtension::getSparseSubPixel( *p_gradX_, u, v );
    const float gy = cvExtension::getSparseSubPixel( *p_gradY_, u, v );
    jac_pixel_CamNor( 0 ) = gx * fx_;
    jac_pixel_CamNor( 1 ) = gy * fy_;

    _jacobianOplusXi = jac_pixel_CamNor * jac_CamNor_ksai;
}
void EdgeInverseDepth::linearizeOplus( )
{
    if ( level() == 1 )
    {
        _jacobianOplusXi = Eigen::Matrix<double, 1, 6>::Zero();
        return;
    }
    g2o::VertexSE3Expmap* v_se3 = static_cast<g2o::VertexSE3Expmap*> ( _vertices[0] );
    Eigen::Vector3d _pt_cam = v_se3->estimate().map ( x_world_ );
    const double &x = _pt_cam[0], &y = _pt_cam[1], &z = _pt_cam[2];

    const double _1_z = 1.0/z, _1_zz = 1.0/(z*z);
    const double x_zz = x*_1_zz, y_zz = y*_1_zz;
    /// the se3 is in (\omega, \epsilon) formt, where \omega is so(3) and \epsilon is translation
    Eigen::Matrix<double,2,6> jac_CamNor_ksai;
    jac_CamNor_ksai( 0,0 ) = -y*x_zz;
    jac_CamNor_ksai( 0,1 ) = 1+( x*x_zz );
    jac_CamNor_ksai( 0,2 ) = -y*_1_z;
    jac_CamNor_ksai( 0,3 ) = _1_z;
    jac_CamNor_ksai( 0,4 ) = 0;
    jac_CamNor_ksai( 0,5 ) = -x_zz;
    jac_CamNor_ksai( 1,0 ) = - ( 1+y*y_zz );
    jac_CamNor_ksai( 1,1 ) = x*y_zz;
    jac_CamNor_ksai( 1,2 ) = x*_1_z;
    jac_CamNor_ksai( 1,3 ) = 0;
    jac_CamNor_ksai( 1,4 ) = _1_z;
    jac_CamNor_ksai( 1,5 ) = -y_zz;
    //上面计算的是地图三维点经过坐标变换，在图像上的像素位置变化

    Eigen::Matrix<double,1,2> jac_pixel_CamNor;
    float u = x*fx_*_1_z + cx_;
    float v = y*fy_*_1_z + cy_;
    const float gx = cvExtension::getSparseSubPixel( *p_gradX_, u, v );
    const float gy = cvExtension::getSparseSubPixel( *p_gradY_, u, v );
    jac_pixel_CamNor( 0 ) = gx * fx_;
    jac_pixel_CamNor( 1 ) = gy * fy_;

    Eigen::Matrix<double,1,6> jac_InvDep_ksai;
    jac_InvDep_ksai(0) =  y*_1_zz;
    jac_InvDep_ksai(1) = -x*_1_zz;
    jac_InvDep_ksai(2) =  0;
    jac_InvDep_ksai(3) =  0;
    jac_InvDep_ksai(4) =  0;
    jac_InvDep_ksai(5) =  _1_zz;

    _jacobianOplusXi = jac_pixel_CamNor * jac_CamNor_ksai + jac_InvDep_ksai;
}
/*
SE3Optimization::SE3Optimization()
{
    fx_ = 535.4, cx_ = 320.1, fy_ = 539.2, cy_ = 247.6;//kinect1
//    fx_ = 530.75, cx_ = 478.15, fy_ = 529.11, cy_ = 263.22;//kinect2
    _1_fx = 1.0/fx_;
    _1_fy = 1.0/fy_;
    pho_gain_ = 1;
    brightness_ = 0;
}
void SE3Optimization::setCameraIntrinsic(const float &_fx, const float &_cx, const float &_fy, const float &_cy )
{
    fx_ = _fx, cx_ = _cx, fy_ = _fy, cy_ = _cy;
    _1_fx = 1.0/fx_;
    _1_fy = 1.0/fy_;
}

void SE3Optimization::init( const std::vector<pcl::PointXYZI, Eigen::aligned_allocator<pcl::PointXYZI> > &_map_pho_pts,  const std::vector<pcl::PointXYZ, Eigen::aligned_allocator<pcl::PointXYZ> > &_map_geo_pts, const cv::Mat &_gray_img8U, const cv::Mat &_depth_img )
{
    assert( _gray_img8U.rows==_depth_img.rows && _gray_img8U.cols==_depth_img.cols );
    width_ = _depth_img.cols, height_ = _depth_img.rows;
    pho_pts_ = _map_pho_pts;
    geo_pts_ = _map_geo_pts;
    _gray_img8U.convertTo( gray32f_, CV_32FC1 );
    gray32f_ *= WEIGHT_PHO2GEO/255.0;// from 0~255 to 0~1
    cv::cvtColor( _gray_img8U, rgb_show, CV_GRAY2RGB );
    _depth_img.convertTo( depth32f_, CV_32FC1 );
    inv_depth32f_ = depth32f_.clone();
    uchar * row_d = depth32f_.data;
    uchar * row_id = inv_depth32f_.data;
    for(int h=0; h<height_; h++)
    {
        float *p_d  = (float *)row_d;
        float *p_id = (float *)row_id;
        row_d += depth32f_.step[0];
        row_id += inv_depth32f_.step[0];
        for( int w=0; w<width_; w++ )
        {
            if( isfinite(*p_d) &&  *p_d!=0 ) *p_id = 1.0 / *p_d;
            else                             *p_id = 0;// Infinity depth means Zero inv-depth
            p_d++;
            p_id++;
        }
    }
    /// calculate the gradient of the gray image, with the hypothesis that the image is dense
    int BLUR_SIZE = height_/10;
    if( BLUR_SIZE%2==0 ) BLUR_SIZE ++;
    /// 注意: 先平滑图像再差分，和，先差分再平滑，是等效的
    /// 证明: 卷积的交换律和结合律。
    cv::Mat gray32f_blur;
    cv::GaussianBlur( gray32f_,gray32f_blur,cv::Size(BLUR_SIZE,BLUR_SIZE),BLUR_SIZE );
    grad_grayX_ = cv::Mat::zeros( height_, width_, CV_32FC1 );
    cv::Mat mat1 = gray32f_blur( cv::Rect(0,0,width_-2,height_) );
    cv::Mat mat2 = gray32f_blur( cv::Rect(2,0,width_-2,height_) );
    cv::Mat grad = ( mat2 - mat1 ) / 2.0;
    grad.copyTo( grad_grayX_( cv::Rect(1,0,width_-2,height_) ) );
    cv::Mat edge = gray32f_blur.col(1) - gray32f_blur.col(0);
    edge.copyTo( grad_grayX_( cv::Rect(0,0,1,height_) ) );
    edge = gray32f_blur.col(width_-1) - gray32f_blur.col(width_-2);
    edge.copyTo( grad_grayX_( cv::Rect(width_-1,0,1,height_) ) );

    grad_grayY_ = cv::Mat::zeros( height_, width_, CV_32FC1 );
    mat1 = gray32f_blur( cv::Rect(0,0,width_,height_-2) );
    mat2 = gray32f_blur( cv::Rect(0,2,width_,height_-2) );
    grad = ( mat2 - mat1 ) / 2.0;
    grad.copyTo( grad_grayY_( cv::Rect(0,1,width_,height_-2) ) );
    edge = gray32f_blur.row(1) - gray32f_blur.row(0);
    edge.copyTo( grad_grayY_( cv::Rect(0,0,width_,1) ) );
    edge = gray32f_blur.row(height_-1) - gray32f_blur.row(height_-2);
    edge.copyTo( grad_grayY_( cv::Rect(0,height_-1,width_,1) ) );

    /// calculate the gradient of the inverse depth image, with the hypothesis that the image is NOT dense
    grad_inv_depthX_ = cv::Mat::zeros( height_, width_, CV_32FC1 );
    grad_inv_depthY_ = cv::Mat::zeros( height_, width_, CV_32FC1 );
    row_id = inv_depth32f_.data;
    const int step0_id = inv_depth32f_.step[0] / 4;//float step
    uchar * row_gx = grad_inv_depthX_.data;
    uchar * row_gy = grad_inv_depthY_.data;
    for(int h=0; h<height_; h++)
    {
        float *p_id = (float *)row_id;
        float *p_gx = (float *)row_gx;
        float *p_gy = (float *)row_gy;
        row_id += inv_depth32f_.step[0];
        row_gx += grad_inv_depthX_.step[0];
        row_gy += grad_inv_depthY_.step[0];
        for( int w=0; w<width_; w++ )
        {
            float l, r, u, d;
            if( w==0 )             l = *p_id,                r = *(p_id+1);
            else if( w==width_-1 ) l = *(p_id-1),            r = *p_id;
            else                   l = *(p_id-1)/2.0,        r = *(p_id+1)/2.0;
            if( h==0 )             u = * p_id,               d = *(p_id+step0_id);
            else if( h==height_-1 )u = *(p_id-step0_id),     d = * p_id;
            else                   u = *(p_id-step0_id)/2.0, d = *(p_id+step0_id)/2.0;
            if( l>0 && r>0 ) *p_gx = r - l;
            else             *p_gx = 0.0;
            if( u>0 && d>0 ) *p_gy = d - u;
            else             *p_gy = 0.0;
            p_id ++;
            p_gx ++;
            p_gy ++;
        }
    }
    cv::GaussianBlur( grad_inv_depthX_,grad_inv_depthX_,cv::Size(BLUR_SIZE,BLUR_SIZE),BLUR_SIZE );
    cv::GaussianBlur( grad_inv_depthY_,grad_inv_depthY_,cv::Size(BLUR_SIZE,BLUR_SIZE),BLUR_SIZE );

}

Vector6f
SE3Optimization::optimizeOnce(Isometry3f &_init_iso )
{
    ///对于一个目标函数为 1/2*W*err^2 的优化问题, 其中W为Geman-McClure权重，err为误差(测量值减真实值)。
    /// 在求偏导的时候，W认为是常值，但是每次迭代都要重新计算权重。
    /// 使用L-M method解得: (Jt*W*J+lambda)^-1*deta=-Jt*W*err
    /// 其中deta即为迭代增量，但是实际还要乘以一个合适的步长。J为雅可比矩阵。lambda是L-M method方法引入的因子(此处不做赘述)
    /// a.值得注意的是，雅可比的计算，一般参考资料中写的公式是地图点发生位置变化而引起的像素变化，
    /// 因此该函数的优化结果也是地图点相对当前帧的位姿变化(增量)。
    /// b.值得注意的是，在判断优化迭代是否收敛时，不能使用目标函数作为参考。
    /// 这个带权重的误差随着迭代，它可能并不减小。因为每次迭代的权重是变化的，一个地图点越接近他的真实目标像素，虽然误差更小，但是权重也变大了。
    /// 相似的道理，不带权重的误差也不能作为判断收敛的依据

    /////////// 1. calculate error and weight/////////////////////
    /// 1.1 calculate the photometric error
    errors_pho_.clear();
    errors_pho_.reserve( pho_pts_.size() );
    error2s_pho_.clear();
    error2s_pho_.reserve( pho_pts_.size() );
    std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f> > pt_cam_pho;
    std::vector<Eigen::Vector2f, Eigen::aligned_allocator<Eigen::Vector2f> > pt_img_pho;
    pt_cam_pho.reserve( pho_pts_.size() );
    pt_img_pho.reserve( pho_pts_.size() );
    double variance_pho = 0;
    for( const pcl::PointXYZI &pt : pho_pts_ )
    {
        const Eigen::Map<Eigen::Vector3f> pt_map( (float*)pt.data );// share the same memory
        Eigen::Vector3f pt_cam = _init_iso * pt_map;
        if( isnan(pt_cam[2]) || pt_cam[2]<SENSE_RANGE_MIN || pt_cam[2]>SENSE_RANGE_MAX )
            continue;
        Eigen::Vector2f pt_img( pt_cam[0]/pt_cam[2]*fx_+cx_, pt_cam[1]/pt_cam[2]*fy_+cy_ );
        if( pt_img[0]<0 || pt_img[0]>gray32f_.cols-1 || pt_img[1]<0 || pt_img[1]>gray32f_.rows-1 )
            continue;
        pt_cam_pho.push_back( pt_cam );
        pt_img_pho.push_back( pt_img );
        const float intensity = getSubPixel(gray32f_, pt_img[0], pt_img[1] );
        float error = intensity*pho_gain_ + brightness_ - pt.intensity;
//        std::cout << "pho err=" << intensity << "-" << pt.intensity << std::endl;
        float error2 = error * error;
        errors_pho_.push_back( error );
        error2s_pho_.push_back( error2 );
        variance_pho += error2;
    }
    const int pho_cnt = errors_pho_.size();
    variance_pho = variance_pho / pho_cnt;
    /// 1.2 calculate the Geman-McClure weight of photometric
    weights_pho_.clear();
    weights_pho_.reserve( pho_cnt );
    for( const float &error2 : error2s_pho_ )
    {
        float weight_sqr = variance_pho / ( variance_pho + error2 );
        weights_pho_.push_back( weight_sqr*weight_sqr );
    }
    /// 1.3 calculate the geometric error
    errors_geo_.clear();
    errors_geo_.reserve( geo_pts_.size() );
    error2s_geo_.clear();
    error2s_geo_.reserve( geo_pts_.size() );
    std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f> > pt_cam_geo;
    std::vector<Eigen::Vector2f, Eigen::aligned_allocator<Eigen::Vector2f> > pt_img_geo;
    pt_cam_geo.reserve( pho_pts_.size() );
    double variance_geo = 0;
    for( const pcl::PointXYZ pt : geo_pts_ )
    {
        const Eigen::Map<Eigen::Vector3f> pt_map( (float*)pt.data );// share the same memory
        Eigen::Vector3f pt_cam = _init_iso * pt_map;
        if( isnan(pt_cam[2]) || pt_cam[2]<SENSE_RANGE_MIN || pt_cam[2]>SENSE_RANGE_MAX )
            continue;
        Eigen::Vector2f pt_img( pt_cam[0]/pt_cam[2]*fx_+cx_, pt_cam[1]/pt_cam[2]*fy_+cy_ );
        if( pt_img[0]<0 || pt_img[0]>depth32f_.cols-1 || pt_img[1]<0 || pt_img[1]>depth32f_.rows-1 )
            continue;
        const float &inv_depth = getSubPixel( inv_depth32f_, pt_img[0], pt_img[1] );
        if( std::isnan(inv_depth) || inv_depth==0 )
            continue;
        pt_cam_geo.push_back( pt_cam );
        pt_img_geo.push_back( pt_img );
        float error = inv_depth - 1.0/pt_cam[2];
//        std::cout << "geo err=" << 1.0/pt_cam[2] << "-" << inv_depth << std::endl;
        float error2 = error * error;
        errors_geo_.push_back( error );
        error2s_geo_.push_back( error2 );
        variance_geo += error2;
    }
    const int geo_cnt = errors_geo_.size();
    variance_geo = variance_geo / geo_cnt;
    /// 1.4 calculate the Geman-McClure weight of geometric
    weights_geo_.clear();
    weights_geo_.reserve( geo_cnt );
    for( const float &error2 : error2s_geo_ )
    {
        float weight_sqr = variance_geo / ( variance_geo + error2 );
        weights_geo_.push_back( weight_sqr*weight_sqr );
    }

    /////////// 2. calculate jacobian /////////////////////////////////////////
    jacobian_pho_.clear();
    jacobian_pho_.reserve( pho_cnt );
    for (int i = 0; i < pho_cnt; i++)
    {
        Eigen::Vector6f jac = generateJacobian( pt_cam_pho[i], pt_img_pho[i], grad_grayX_, grad_grayY_ );
        jacobian_pho_.push_back( jac );
    }
    jacobian_geo_.clear();
    jacobian_geo_.reserve( errors_pho_.size() );
    for (int i = 0; i < geo_cnt; i++)
    {
        Eigen::Vector6f jac = generateJacobian( pt_cam_geo[i], pt_img_geo[i], grad_inv_depthX_, grad_inv_depthY_ );
        jacobian_geo_.push_back( jac );
    }

    /////////// 3. solve linear faunction of L-M method: (Jt*W*J+lambda)^-1*deta=-Jt*W*err   ////
    float error_return;
    const Eigen::Map<Eigen::MatrixXf> Jt_p( jacobian_pho_[0].data(), 6, jacobian_pho_.size() );//eig矩阵默认按列存储
    const Eigen::Map<Eigen::VectorXf> W_p( &weights_pho_[0], weights_pho_.size(), 1 );
    const Eigen::Map<Eigen::VectorXf> E_p( &errors_pho_[0], errors_pho_.size(), 1 );
    const Eigen::Map<Eigen::VectorXf> E2_p( &error2s_pho_[0], errors_pho_.size(), 1 );
    const Eigen::Map<Eigen::MatrixXf> Jt_g( jacobian_geo_[0].data(), 6, jacobian_geo_.size() );
    const Eigen::Map<Eigen::VectorXf> W_g( &weights_geo_[0], weights_geo_.size(), 1 );
    const Eigen::Map<Eigen::VectorXf> E_g( &errors_geo_[0], errors_geo_.size(), 1 );
    const Eigen::Map<Eigen::VectorXf> E2_g( &error2s_geo_[0], errors_geo_.size(), 1 );
    const Eigen::Matrix6f lambda = Eigen::Vector6f::Constant(20000).asDiagonal();// 20000
    Eigen::Matrix6f H;
    Eigen::Vector6f B;
    if( pho_cnt>6 && geo_cnt>6 )
    {
        H = Jt_p * W_p.asDiagonal() * Jt_p.transpose() + Jt_g * W_g.asDiagonal() * Jt_g.transpose() + lambda;
        B = -Jt_p * W_p.asDiagonal() * E_p - Jt_g * W_g.asDiagonal() * E_g;
        error_return = W_p.dot(E2_p)/pho_cnt + W_g.dot(E2_g)/geo_cnt;
    }
    else if( pho_cnt>6 )
    {
        H = Jt_p * W_p.asDiagonal() * Jt_p.transpose() + lambda;
        B = -Jt_p * W_p.asDiagonal() * E_p;
        error_return = W_p.dot(E2_p)/pho_cnt;
    }
    else if( geo_cnt>6 )
    {
        H = Jt_g * W_g.asDiagonal() * Jt_g.transpose() + lambda;
        B = -Jt_g * W_g.asDiagonal() * E_g;
        error_return = W_g.dot(E2_g)/geo_cnt;
    }
    else//缺少有效的地图点,优化失败
    {
        std::cout << "not enough map points after transform" << std::endl;
        return Eigen::Vector6f::Zero();
    }

    Eigen::Vector6f delta = H.colPivHouseholderQr().solve(B);
    if( isnan(delta[0]) )//解方程失败
        return Eigen::Vector6f::Zero();
    float step = 4;
    delta *= step;
    return delta;
}

Eigen::Vector6f
SE3Optimization::generateJacobian( const Eigen::Vector3f &_pt_cam, const Eigen::Vector2f &_pt_img, const cv::Mat &_gradX, const cv::Mat &_gradY )
const
{
    const float &x = _pt_cam[0], &y = _pt_cam[1], &z = _pt_cam[2];
    const float _1_z = 1.0/z, _1_zz = 1.0/(z*z);
    const float x_zz = x*_1_zz, y_zz = y*_1_zz;
    /// the se3 is in (\omega, \epsilon) formt, where \omega is so(3) and \epsilon is translation
    Eigen::Matrix<float,2,6> jac_CamNor_ksai;
    jac_CamNor_ksai( 0,0 ) = -y*x_zz;
    jac_CamNor_ksai( 0,1 ) = 1+( x*x_zz );
    jac_CamNor_ksai( 0,2 ) = -y*_1_z;
    jac_CamNor_ksai( 0,3 ) = _1_z;
    jac_CamNor_ksai( 0,4 ) = 0;
    jac_CamNor_ksai( 0,5 ) = -x_zz;
    jac_CamNor_ksai( 1,0 ) = - ( 1+y*y_zz );
    jac_CamNor_ksai( 1,1 ) = x*y_zz;
    jac_CamNor_ksai( 1,2 ) = x*_1_z;
    jac_CamNor_ksai( 1,3 ) = 0;
    jac_CamNor_ksai( 1,4 ) = _1_z;
    jac_CamNor_ksai( 1,5 ) = -y_zz;
    //上面计算的是三维点经过坐标变换，在图像上的像素位置变化

    Eigen::Matrix<float,1,2> jac_pixel_CamNor;
    const float &u = _pt_img[0], &v = _pt_img[1];
    const float gx = getSubPixel( _gradX, u, v );
    const float gy = getSubPixel( _gradY, u, v );
    jac_pixel_CamNor( 0 ) = gx * fx_;
    jac_pixel_CamNor( 1 ) = gy * fy_;
    return jac_pixel_CamNor * jac_CamNor_ksai;
}

bool SE3Optimization::optimize(  Eigen::Isometry3f &_init_iso )
{

    // 初始化g2o
    optimizer.clear();
    g2o::BlockSolverX::LinearSolverType * linearSolver = new g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>();
    g2o::BlockSolverX* blockSolver = new g2o::BlockSolverX(linearSolver);
    g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton( blockSolver ); // G-N
//    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg ( blockSolver ); // L-M
//    solver->setMaxTrialsAfterFailure(100);
    optimizer.setAlgorithm ( solver );
    optimizer.setVerbose( true );//显示详细信息

    int id = 0;
    // 添加顶点
    g2o::VertexSE3Expmap *pose = new g2o::VertexSE3Expmap();
    g2o::SE3Quat pose_se3( _init_iso.linear().cast<double>(), _init_iso.translation().cast<double>() );
    pose->setEstimate( pose_se3 );
    pose->setId ( id++ );
    optimizer.addVertex ( pose );
    // 添加边
    for( const pcl::PointXYZI pt : pho_pts_ )
    {
        const Eigen::Map<Eigen::Vector3f> pt_map( (float*)pt.data );
        Eigen::Vector3d pt_cam = pt_map.cast<double>();
        EdgePhotometic* edge = new EdgePhotometic ( pt_cam, &gray32f_, &grad_grayX_, &grad_grayY_ );
        edge->setVertex ( 0, pose );
        edge->setMeasurement ( pt.intensity/255.0*WEIGHT_PHO2GEO );// from 0~255 to 0~1
        edge->setInformation ( Eigen::Matrix<double,1,1>::Identity() );
//        edge->setRobustKernel( new g2o::RobustKernelHuber() );
        edge->setId( id++ );
        optimizer.addEdge ( edge );
    }
    for( const pcl::PointXYZ pt : geo_pts_ )
    {
        const Eigen::Map<Eigen::Vector3f> pt_map( (float*)pt.data );
        Eigen::Vector3d pt_cam = pt_map.cast<double>();
        EdgeInverseDepth* edge = new EdgeInverseDepth ( pt_cam, &inv_depth32f_, &grad_inv_depthX_, &grad_inv_depthY_ );
        edge->setVertex ( 0, pose );
        edge->setMeasurement ( 1.0/pt_cam[2] );//对于EdgeInverseDepth而言，Measurement是在线计算的，此处赋值没有意义
        edge->setInformation ( Eigen::Matrix<double,1,1>::Identity() );
//        edge->setRobustKernel( new g2o::RobustKernelHuber() );
        edge->setId( id++ );
        optimizer.addEdge ( edge );
    }

//    PreIterationAction *p_pra = new PreIterationAction;
//    optimizer.addComputeErrorAction( p_pra );
//    ComputeErrorAction *p_cea = new ComputeErrorAction;
//    optimizer.addComputeErrorAction( p_cea );
    PostIterationAction *p_pta = new PostIterationAction( this );
    optimizer.addComputeErrorAction( p_pta );

    optimizer.initializeOptimization();
    optimizer.optimize(10);

    pose_se3 = pose->estimate();
    _init_iso.linear() =  pose_se3.rotation().toRotationMatrix().cast<float>();
    _init_iso.translation() = pose_se3.translation().cast<float>();
}

void
SE3Optimization::drawPoints( cv::Mat &_image, const Eigen::Isometry3f &_iso,
                                std::vector<pcl::PointXYZI, Eigen::aligned_allocator<pcl::PointXYZI> > _pho_pts,
                                std::vector<pcl::PointXYZ, Eigen::aligned_allocator<pcl::PointXYZ> > _geo_pts ) const
{
    for( const pcl::PointXYZI &pt : _pho_pts )
    {
        pcl::PointXYZI pt2 = _iso*pt;
        cv::Point pt_img( pt2.x/pt2.z*fx_+cx_, pt2.y/pt2.z*fy_+cy_ );
        if( pt_img.x<0 || pt_img.x>gray32f_.cols || pt_img.y<0 || pt_img.y>gray32f_.rows )
            continue;
        cv::circle( _image, pt_img, 1, CV_RGB(255,0,0), CV_FILLED );
    }
    for( pcl::PointXYZ pt : _geo_pts )
    {
        pcl::PointXYZ pt2 = _iso*pt;
        cv::Point pt_img( pt2.x/pt2.z*fx_+cx_, pt2.y/pt2.z*fy_+cy_ );
        if( pt_img.x<0 || pt_img.x>gray32f_.cols || pt_img.y<0 || pt_img.y>gray32f_.rows )
            continue;
        cv::circle( _image, pt_img, 1, CV_RGB(0,0,255), CV_FILLED );
    }
}

g2o::HyperGraphAction* PostIterationAction::operator()(const g2o::HyperGraph* graph, Parameters* parameters)
{
    const g2o::VertexSE3Expmap* v_se3  =static_cast<const g2o::VertexSE3Expmap*> ( graph->vertex(0) );
    g2o::SE3Quat pose_se3 = v_se3->estimate();
    Eigen::Isometry3f iso;
    iso.linear() =  pose_se3.rotation().toRotationMatrix().cast<float>();
    iso.translation() = pose_se3.translation().cast<float>();

    cv::Mat image_show = p_opti->rgb_show.clone();
    p_opti->drawMapPoints( image_show, iso );
    cv::imshow( "cur_frame", image_show );
    cv::waitKey(5);
    return this;
}
*/
