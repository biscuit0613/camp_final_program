#include "pnp_solver.h"
#include <iostream>

// 构造函数：初始化PnP解算器
// 参数：CameraMatrix - 相机内参矩阵，DistCoeffs - 畸变系数，BallDiameter - 篮球直径
PnPSolver::PnPSolver(cv::Mat CameraMatrix, cv::Mat DistCoeffs, double BallDiameter)
    : CameraMatrix(CameraMatrix), DistCoeffs(DistCoeffs) {
    // 初始化3D对象点：篮球的五个特征点（中心和四个边缘点）
    double r = BallDiameter / 2.0;
    ObjectPoints = {
        cv::Point3f(0, 0, 0),      // 中心点
        cv::Point3f(r, 0, 0),      // 右边缘
        cv::Point3f(-r, 0, 0),     // 左边缘
        cv::Point3f(0, r, 0),      // 上边缘
        cv::Point3f(0, -r, 0)      // 下边缘
    };
}

// 解算方法：使用PnP算法将2D图像点转换为3D世界坐标
// 参数：ImagePoints - 2D图像点，Position - 输出3D位置
// 返回：解算是否成功
bool PnPSolver::Solve(const std::vector<cv::Point2f>& ImagePoints, Eigen::Vector3d& Position) {
    if (ImagePoints.size() != ObjectPoints.size()) {
        std::cerr << "Error: Number of image points does not match object points." << std::endl;
        return false;
    }

    // 使用OpenCV的solvePnP函数进行姿态解算
    cv::Mat rvec, tvec;
    bool success = cv::solvePnP(ObjectPoints, ImagePoints, CameraMatrix, DistCoeffs, rvec, tvec);

    if (success) {
        // 将旋转向量转换为旋转矩阵
        cv::Mat R;
        cv::Rodrigues(rvec, R);

        // 计算3D位置：tvec是相机坐标系下的位置
        Position = Eigen::Vector3d(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));
    }

    return success;
}
