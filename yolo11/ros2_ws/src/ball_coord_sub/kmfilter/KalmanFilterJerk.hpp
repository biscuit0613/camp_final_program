#pragma once
#include <Eigen/Dense>
#include <vector>

//有一个比较邪门的思路，从6维的匀速滤波器，升级到9维的匀加速滤波器，效果好了很多，再升级到12维的变加速度滤波器应该效果能更好一些
//上网查了一下jerk就是加速度的变化率的英文，用来反映篮球加速度的变化。
class KalmanFilterJerk {
public:
    KalmanFilterJerk();
    void init(const Eigen::Vector3d& pos);
    void predict(float dt);
    void update(const Eigen::Vector3d& pos, float dt);
    Eigen::Vector3d getPosition() const;
    Eigen::Vector3d getVelocity() const;

private:
    Eigen::Matrix<double, 12, 1> state_; // [x, y, z, vx, vy, vz, ax, ay, az, jx, jy, jz]
    Eigen::Matrix<double, 12, 12> P_;     // 协方差
    Eigen::Matrix<double, 12, 12> Q_;     // 过程噪声
    Eigen::Matrix<double, 3, 3> R_;     // 观测噪声
    Eigen::Matrix<double, 12, 12> F_;     // 状态转移矩阵
    Eigen::Matrix<double, 3, 12> H_;     // 观测矩阵
    bool initialized_ = false;
};