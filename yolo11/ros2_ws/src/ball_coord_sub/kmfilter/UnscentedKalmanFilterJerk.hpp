#pragma once
#include <Eigen/Dense>
#include <vector>

/*
AI辅助：在jerk的基础上让ai写了无迹卡尔曼滤波器的版本
*/


// 无迹卡尔曼滤波器（UKF）版本的jerk模型，用于处理非线性状态转移
class UnscentedKalmanFilterJerk {
public:
    UnscentedKalmanFilterJerk();
    void init(const Eigen::Vector3d& pos);
    void predict(float dt);
    void update(const Eigen::Vector3d& pos, float dt);
    Eigen::Vector3d getPosition() const;
    Eigen::Vector3d getVelocity() const;

private:
    Eigen::Matrix<double, 12, 1> state_; // [x, y, z, vx, vy, vz, ax, ay, az, jx, jy, jz]
    Eigen::Matrix<double, 12, 12> P_;     // 协方差
    Eigen::Matrix<double, 12, 12> Q_;     // 过程噪声
    Eigen::Matrix<double, 3, 3> R_;       // 观测噪声
    bool initialized_ = false;

    // UKF参数
    double alpha_ = 1e-3;  // 调节参数
    double beta_ = 2.0;   // 高斯分布参数
    double kappa_ = 0.0;  // 调节参数
    double lambda_;       // 计算参数
    double gamma_;        // 计算参数
    int n_ = 12;          // 状态维度
    int num_sigma_ = 25;  // sigma points数量 (2n+1)

    // 生成sigma points
    Eigen::MatrixXd generateSigmaPoints();
    // 预测sigma points
    Eigen::MatrixXd predictSigmaPoints(const Eigen::MatrixXd& sigmaPoints, float dt);
    // 计算均值和协方差
    void computeMeanAndCovariance(const Eigen::MatrixXd& sigmaPoints, Eigen::VectorXd& mean, Eigen::MatrixXd& cov);
    // 非线性预测函数
    Eigen::VectorXd processModel(const Eigen::VectorXd& state, float dt);
    // 观测函数
    Eigen::VectorXd measurementModel(const Eigen::VectorXd& state);
};
