#include "KF.hpp"

KF::KF() {
    state_.setZero();
    P_.setIdentity();
    Q_.setIdentity();
    Q_ *= 1e-3; // 过程噪声可调
    R_.setIdentity();
    R_ *= 1e-2; // 观测噪声可调
    last_time_ = 0;
    initialized_ = false;
}

void KF::init(const Eigen::Vector3d& pos, double t) {
    state_.head<3>() = pos;
    state_.tail<3>().setZero();
    P_.setIdentity();
    last_time_ = t;
    initialized_ = true;
}

void KF::predict(double t) {
    if (!initialized_) return;
    double dt = t - last_time_;
    if (dt <= 0) return;
    // 状态转移矩阵
    Eigen::Matrix<double, 6, 6> F = Eigen::Matrix<double, 6, 6>::Identity();
    F(0, 3) = dt; F(1, 4) = dt; F(2, 5) = dt;
    // 预测
    state_ = F * state_;
    P_ = F * P_ * F.transpose() + Q_;
    last_time_ = t;
}

void KF::update(const Eigen::Vector3d& pos, double t) {
    if (!initialized_) {
        init(pos, t);
        return;
    }
    predict(t);
    // 观测矩阵
    Eigen::Matrix<double, 3, 6> H = Eigen::Matrix<double, 3, 6>::Zero();
    H(0, 0) = 1; H(1, 1) = 1; H(2, 2) = 1;
    // 卡尔曼增益
    Eigen::Matrix3d S = H * P_ * H.transpose() + R_;
    Eigen::Matrix<double, 6, 3> K = P_ * H.transpose() * S.inverse();
    // 更新
    Eigen::Vector3d y = pos - H * state_;
    state_ = state_ + K * y;
    P_ = (Eigen::Matrix<double, 6, 6>::Identity() - K * H) * P_;
}

Eigen::Vector3d KF::getPosition() const {
    return state_.head<3>();
}

Eigen::Vector3d KF::getVelocity() const {
    return state_.tail<3>();
}
