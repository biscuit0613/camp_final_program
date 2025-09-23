#ifndef PNP_SOLVER_H
#define PNP_SOLVER_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <Eigen/Dense>

// PnP解算器类，用于将2D图像点解算为3D世界坐标
class PnPSolver {
public:
    PnPSolver(cv::Mat CameraMatrix, cv::Mat DistCoeffs, double BallDiameter);  // 构造函数，初始化相机参数和球直径
    bool Solve(const std::vector<cv::Point2f>& ImagePoints, Eigen::Vector3d& Position);  // 解算方法，返回3D位置

private:
    cv::Mat CameraMatrix;  // 相机内参矩阵
    cv::Mat DistCoeffs;    // 畸变系数
    std::vector<cv::Point3f> ObjectPoints;  // 3D对象点（球的五个特征点）
};

#endif