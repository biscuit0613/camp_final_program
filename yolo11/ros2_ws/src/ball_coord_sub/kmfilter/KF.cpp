#include "KF.hpp"

KF::KF() {
    state_.setZero();
    P_.setIdentity();
    Q_.setIdentity();
    Q_ *= 1e-3; // 过程噪声：预测不确定性，调大更依赖观测
    R_.setIdentity();
    R_ *= 1e-3; // 观测噪声：调小更信任观测（测量结果置信度更高）
    initialized_ = false;
}

void KF::init(const Eigen::Vector3d& pos) {
    state_.head<3>() = pos;//x, y, z
    state_.segment<3>(3).setZero(); // vx, vy, vz
    state_.tail<3>().setZero();     // ax, ay, az
    P_.setIdentity();
    initialized_ = true;
}

void KF::predict(float dt) {
    if (!initialized_ || dt <= 0) return;
    // 状态转移矩阵 (9x9 for constant acceleration model)
    Eigen::Matrix<double, 9, 9> F = Eigen::Matrix<double, 9, 9>::Identity();
    F(0, 3) = dt; F(0, 6) = 0.5 * dt * dt;  // x
    F(1, 4) = dt; F(1, 7) = 0.5 * dt * dt;  // y
    F(2, 5) = dt; F(2, 8) = 0.5 * dt * dt;  // z
    F(3, 6) = dt;  // vx
    F(4, 7) = dt;  // vy
    F(5, 8) = dt;  // vz
    // 预测
    state_ = F * state_;
    P_ = F * P_ * F.transpose() + Q_;
}

void KF::update(const Eigen::Vector3d& pos, float dt) {
    if (!initialized_) {
        init(pos);
        return;
    }
    predict(dt);
    // 观测矩阵 (3x9, 只观测位置)
    Eigen::Matrix<double, 3, 9> H = Eigen::Matrix<double, 3, 9>::Zero();
    H(0, 0) = 1; H(1, 1) = 1; H(2, 2) = 1;
    // 卡尔曼增益
    Eigen::Matrix3d S = H * P_ * H.transpose() + R_;
    Eigen::Matrix<double, 9, 3> K = P_ * H.transpose() * S.inverse();
    // 更新
    Eigen::Vector3d y = pos - H * state_;
    state_ = state_ + K * y;
    Eigen::Matrix<double, 9, 9> I = Eigen::Matrix<double, 9, 9>::Identity();
    P_ = (I - K * H) * P_ * (I - K * H).transpose() + K * R_ * K.transpose();
}

Eigen::Vector3d KF::getPosition() const {
    return state_.head<3>();
}

Eigen::Vector3d KF::getVelocity() const {
    return state_.segment<3>(3);
}
