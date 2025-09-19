#pragma once
#include <Eigen/Dense>

class KF {
public:
    KF();
    void init(const Eigen::Vector3d& pos);
    void predict(float dt);
    void update(const Eigen::Vector3d& pos, float dt);
    Eigen::Vector3d getPosition() const;
    Eigen::Vector3d getVelocity() const;

private:
    Eigen::Matrix<double, 9, 1> state_; // [x, y, z, vx, vy, vz, ax, ay, az]
    Eigen::Matrix<double, 9, 9> P_;     // 协方差
    Eigen::Matrix<double, 9, 9> Q_;     // 过程噪声
    Eigen::Matrix<double, 3, 3> R_;     // 观测噪声
    bool initialized_ = false;
};
