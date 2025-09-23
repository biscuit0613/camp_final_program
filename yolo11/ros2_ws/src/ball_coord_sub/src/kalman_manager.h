#ifndef KALMAN_MANAGER_H
#define KALMAN_MANAGER_H

#include <unordered_map>
#include "../kmfilter/UnscentedKalmanFilterJerk.hpp"
// #include "../kmfilter/KalmanFilterJerk.hpp"
#include <Eigen/Dense>
// #include "../kmfilter/KF.hpp"
//这里有三个卡尔曼那滤波器，现在是没有注释掉的（带jerk和无迹的）效果好一些


//根据球id的卡尔曼滤波管理器类，用于管理多个篮球的卡尔曼滤波跟踪
class KalmanManager {
public:
    void Update(int ObjId, const Eigen::Vector3d& Obs, double Dt);  // 更新指定ID的卡尔曼滤波器，使用观测值和时间间隔
    void Predict(int ObjId, double Dt);  // 预测指定ID的卡尔曼滤波器状态，使用时间间隔
    Eigen::Vector3d GetPosition(int ObjId);  // 获取指定ID篮球的当前位置
    // bool HasFilter(int ObjId);  // 检查是否存在指定ID的卡尔曼滤波器，测试用

private:
    std::unordered_map<int, UnscentedKalmanFilterJerk> KfMap; // 整了一个字典，key是目标的id,值是对应的UKF实例

    // std::unordered_map<int, KalmanFilterJerk> KfMap; // 整了一个字典，key是目标的id,值是对应的卡尔曼滤波实例
};

#endif
