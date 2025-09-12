#pragma once
#include <Eigen/Dense>

class KF {
public:
    KF();
    void init(const Eigen::Vector3d& pos, double t);
    void predict(double t);
    void update(const Eigen::Vector3d& pos, double t);
    Eigen::Vector3d getPosition() const;
    Eigen::Vector3d getVelocity() const;

private:
    Eigen::Matrix<double, 6, 1> state_; // [x, y, z, vx, vy, vz]
    Eigen::Matrix<double, 6, 6> P_;     // 协方差
    Eigen::Matrix<double, 6, 6> Q_;     // 过程噪声
    Eigen::Matrix<double, 3, 3> R_;     // 观测噪声
    double last_time_ = 0;
    bool initialized_ = false;
};
