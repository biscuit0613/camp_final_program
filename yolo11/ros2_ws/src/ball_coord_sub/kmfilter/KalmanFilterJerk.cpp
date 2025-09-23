#include "KalmanFilterJerk.hpp"

KalmanFilterJerk::KalmanFilterJerk() {
    // 初始化状态向量为零
    state_.setZero();

    // 初始化协方差矩阵 P_
    P_.setIdentity();
    P_ *= 1e-3; // 初始不确定性较小，表示对初始状态较有信心

    // 初始化过程噪声矩阵 Q_
    Q_.setZero();
    double sigma_a = 0.3; // 加速度噪声标准差（过程噪声来自于学长的篮球技术，以及篮球撞到然后反弹之类的不可预测的过程）
    double sigma_jerk = 5.0; // jerk噪声
    for (int i = 0; i < 3; ++i) {
        Q_(i, i) = 0.25 * sigma_a * sigma_a;       // 位置噪声
        Q_(i + 3, i + 3) = sigma_a * sigma_a;      // 速度噪声
        Q_(i + 6, i + 6) = sigma_a * sigma_a;      // 加速度噪声
        Q_(i + 9, i + 9) = sigma_jerk * sigma_jerk; // jerk噪声
    }

    // 初始化观测噪声矩阵 R_
    R_.setZero();
    double sigma_pos = 0.0001; // 位置测量误差标准差（来自yolo检测，相对来说比较准确）
    for (int i = 0; i < 3; ++i) {
        R_(i, i) = sigma_pos * sigma_pos;
    }

    // 初始化观测矩阵 H (3x12) - 只观测位置
    H_.setZero();
    H_.block<3,3>(0,0) = Eigen::Matrix3d::Identity();

    // 初始化状态转移矩阵 F_
    F_.setIdentity();

    // 初始化标志位
    initialized_ = false;
}

void KalmanFilterJerk::init(const Eigen::Vector3d& pos) {
    state_.head<3>() = pos;//x, y, z
    state_.segment<3>(3).setZero(); // vx, vy, vz
    state_.segment<3>(6).setZero(); // ax, ay, az
    state_.tail<3>().setZero();     // jx, jy, jz
    P_.setIdentity();
    initialized_ = true;
}

void KalmanFilterJerk::predict(float dt) {
    if (!initialized_ || dt <= 0) return;

    // 更新状态转移矩阵 F_
    F_.setIdentity();
    // 位置 ← 速度
    F_.block<3,3>(0,3) = Eigen::Matrix3d::Identity() * dt;
    // 位置 ← 加速度
    F_.block<3,3>(0,6) = Eigen::Matrix3d::Identity() * 0.5 * dt * dt;
    // 位置 ← jerk
    F_.block<3,3>(0,9) = Eigen::Matrix3d::Identity() * (1.0/6.0) * dt * dt * dt;
    // 速度 ← 加速度
    F_.block<3,3>(3,6) = Eigen::Matrix3d::Identity() * dt;
    // 速度 ← jerk
    F_.block<3,3>(3,9) = Eigen::Matrix3d::Identity() * 0.5 * dt * dt;
    // 加速度 ← jerk
    F_.block<3,3>(6,9) = Eigen::Matrix3d::Identity() * dt;

    // 更新过程噪声 Q_
    Q_.setZero();
    double sigma_pos = 0.01;  // 位置过程噪声
    double sigma_vel = 0.5;   // 速度过程噪声
    double sigma_acc = 5.0;   // 加速度过程噪声
    double sigma_jerk = 5.0;  // jerk过程噪声
    for (int i = 0; i < 3; ++i) {
        Q_(i, i) = sigma_pos * sigma_pos * dt;       // 位置噪声
        Q_(i + 3, i + 3) = sigma_vel * sigma_vel * dt; // 速度噪声
        Q_(i + 6, i + 6) = sigma_acc * sigma_acc * dt; // 加速度噪声
        Q_(i + 9, i + 9) = sigma_jerk * sigma_jerk * dt; // jerk噪声
    }

    // 预测
    state_ = F_ * state_;
    P_ = F_ * P_ * F_.transpose() + Q_;
}

void KalmanFilterJerk::update(const Eigen::Vector3d& pos, float dt) {
    if (!initialized_) {
        init(pos);
        return;
    }
    // 观测矩阵 (3x12, 只观测位置)
    // H_ 已初始化
    // 卡尔曼增益
    Eigen::Matrix3d S = H_ * P_ * H_.transpose() + R_;
    Eigen::Matrix<double, 12, 3> K = P_ * H_.transpose() * S.inverse();
    // 更新
    Eigen::Vector3d y = pos - H_ * state_;
    state_ = state_ + K * y;
    Eigen::Matrix<double, 12, 12> I = Eigen::Matrix<double, 12, 12>::Identity();
    P_ = (I - K * H_) * P_ * (I - K * H_).transpose() + K * R_ * K.transpose();
}

Eigen::Vector3d KalmanFilterJerk::getPosition() const {
    return state_.head<3>();
}

Eigen::Vector3d KalmanFilterJerk::getVelocity() const {
    return state_.segment<3>(3);
}