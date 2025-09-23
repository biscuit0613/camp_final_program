#include "kalman_manager.h"

// 更新指定ID的卡尔曼滤波器
void KalmanManager::Update(int ObjId, const Eigen::Vector3d& Obs, double Dt) {
    KfMap[ObjId].update(Obs, Dt);
}

// 预测指定ID的卡尔曼滤波器状态

void KalmanManager::Predict(int ObjId, double Dt) {
    KfMap[ObjId].predict(Dt);
}

// 获取指定ID篮球的当前位置返回3D位置向量
Eigen::Vector3d KalmanManager::GetPosition(int ObjId) {
    return KfMap[ObjId].getPosition();
}

//测试用
// // 检查是否存在指定ID的卡尔曼滤波器
// // 返回：是否存在该ID的滤波器
// bool KalmanManager::HasFilter(int ObjId) {
//     return KfMap.count(ObjId) > 0;
// }
