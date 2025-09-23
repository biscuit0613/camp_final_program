#include "KF.hpp"

//最初版本的滤波器，九维，位置速度加速度

KF::KF() {
    // 初始化状态向量为零
    state_.setZero();

    // 初始化协方差矩阵 P_
    P_.setIdentity();
    P_ *= 1e-3; // 初始不确定性较小，表示对初始状态较有信心

    // 初始化过程噪声矩阵 Q_
    Q_.setZero();
    float sigma_a = 0.7; // 加速度噪声标准差（过程噪声来自于学长的篮球技术）
    for (int i = 0; i < 3; ++i) {
        Q_(i, i) = 0.25 * sigma_a * sigma_a;       // 位置噪声
        Q_(i + 3, i + 3) = sigma_a * sigma_a;      // 速度噪声
        Q_(i + 6, i + 6) = sigma_a * sigma_a;      // 加速度噪声
    }

    // 初始化观测噪声矩阵 R_
    R_.setZero();
    float sigma_pos = 0.000001; // 位置测量误差标准差（视觉定位）
    for (int i = 0; i < 3; ++i) {
        R_(i, i) = sigma_pos * sigma_pos;
    }

    // 初始化标志位
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
    // predict(dt);
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
