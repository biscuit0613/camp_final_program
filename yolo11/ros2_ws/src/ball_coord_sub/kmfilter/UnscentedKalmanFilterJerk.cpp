#include "UnscentedKalmanFilterJerk.hpp"

UnscentedKalmanFilterJerk::UnscentedKalmanFilterJerk() {
    // 初始化状态向量为零
    state_.setZero();

    // 初始化协方差矩阵 P_
    P_.setIdentity();
    P_ *= 1e-3; // 初始不确定性较小

    // 初始化过程噪声矩阵 Q_
    Q_.setZero();
    double sigma_a = 0.1; // 加速度噪声标准差，减少以提高稳定性
    double sigma_jerk = 1.0; // jerk噪声，减少
    for (int i = 0; i < 3; ++i) {
        Q_(i, i) = 0.25 * sigma_a * sigma_a;       // 位置噪声
        Q_(i + 3, i + 3) = sigma_a * sigma_a;      // 速度噪声
        Q_(i + 6, i + 6) = sigma_a * sigma_a;      // 加速度噪声
        Q_(i + 9, i + 9) = sigma_jerk * sigma_jerk; // jerk噪声
    }

    // 初始化观测噪声矩阵 R_
    R_.setZero();
    double sigma_pos = 0.00001; // 位置测量误差标准差
    for (int i = 0; i < 3; ++i) {
        R_(i, i) = sigma_pos * sigma_pos;
    }

    // 计算UKF参数
    lambda_ = alpha_ * alpha_ * (n_ + kappa_) - n_;
    gamma_ = std::sqrt(n_ + lambda_);
    num_sigma_ = 2 * n_ + 1;
}

void UnscentedKalmanFilterJerk::init(const Eigen::Vector3d& pos) {
    state_.head<3>() = pos;
    state_.segment<3>(3).setZero(); // vx, vy, vz
    state_.segment<3>(6).setZero(); // ax, ay, az
    state_.tail<3>().setZero();     // jx, jy, jz
    P_.setIdentity();
    initialized_ = true;
}

void UnscentedKalmanFilterJerk::predict(float dt) {
    if (!initialized_ || dt <= 0) return;

    // 生成sigma points
    Eigen::MatrixXd sigmaPoints = generateSigmaPoints();

    // 预测sigma points
    Eigen::MatrixXd predSigmaPoints = predictSigmaPoints(sigmaPoints, dt);

    // 计算预测均值和协方差
    Eigen::VectorXd predMean(n_);
    Eigen::MatrixXd predCov(n_, n_);
    computeMeanAndCovariance(predSigmaPoints, predMean, predCov);

    // 更新状态和协方差
    state_ = predMean;
    P_ = predCov + Q_;
}

void UnscentedKalmanFilterJerk::update(const Eigen::Vector3d& pos, float dt) {
    if (!initialized_) {
        init(pos);
        return;
    }

    // 生成sigma points
    Eigen::MatrixXd sigmaPoints = generateSigmaPoints();

    // 预测观测
    Eigen::MatrixXd predMeasSigmaPoints(3, num_sigma_);
    for (int i = 0; i < num_sigma_; ++i) {
        predMeasSigmaPoints.col(i) = measurementModel(sigmaPoints.col(i));
    }

    // 计算预测观测均值和协方差
    Eigen::VectorXd predMeasMean(3);
    Eigen::MatrixXd predMeasCov(3, 3);
    computeMeanAndCovariance(predMeasSigmaPoints, predMeasMean, predMeasCov);

    // 交叉协方差
    Eigen::MatrixXd crossCov(n_, 3);
    crossCov.setZero();
    for (int i = 0; i < num_sigma_; ++i) {
        Eigen::VectorXd diffState = sigmaPoints.col(i) - state_;
        Eigen::Vector3d diffMeas = predMeasSigmaPoints.col(i) - predMeasMean;
        crossCov += (i == 0 ? lambda_ / (n_ + lambda_) : 1.0 / (2 * (n_ + lambda_))) * diffState * diffMeas.transpose();
    }

    // 卡尔曼增益
    Eigen::Matrix3d S = predMeasCov + R_;
    Eigen::MatrixXd K = crossCov * S.inverse();

    // 更新
    Eigen::Vector3d innov = pos - predMeasMean;
    state_ += K * innov;
    P_ -= K * S * K.transpose();
}

Eigen::Vector3d UnscentedKalmanFilterJerk::getPosition() const {
    return state_.head<3>();
}

Eigen::Vector3d UnscentedKalmanFilterJerk::getVelocity() const {
    return state_.segment<3>(3);
}

Eigen::MatrixXd UnscentedKalmanFilterJerk::generateSigmaPoints() {
    Eigen::MatrixXd sigmaPoints(n_, num_sigma_);
    sigmaPoints.col(0) = state_;

    Eigen::MatrixXd sqrtP = P_.llt().matrixL();
    for (int i = 0; i < n_; ++i) {
        sigmaPoints.col(i + 1) = state_ + gamma_ * sqrtP.col(i);
        sigmaPoints.col(i + 1 + n_) = state_ - gamma_ * sqrtP.col(i);
    }
    return sigmaPoints;
}

Eigen::MatrixXd UnscentedKalmanFilterJerk::predictSigmaPoints(const Eigen::MatrixXd& sigmaPoints, float dt) {
    Eigen::MatrixXd predSigmaPoints(n_, num_sigma_);
    for (int i = 0; i < num_sigma_; ++i) {
        predSigmaPoints.col(i) = processModel(sigmaPoints.col(i), dt);
    }
    return predSigmaPoints;
}

void UnscentedKalmanFilterJerk::computeMeanAndCovariance(const Eigen::MatrixXd& sigmaPoints, Eigen::VectorXd& mean, Eigen::MatrixXd& cov) {
    int dim = sigmaPoints.rows();
    mean.setZero();
    cov.setZero();

    // 计算均值
    for (int i = 0; i < num_sigma_; ++i) {
        double weight = (i == 0) ? lambda_ / (n_ + lambda_) : 1.0 / (2 * (n_ + lambda_));
        mean += weight * sigmaPoints.col(i);
    }

    // 计算协方差
    for (int i = 0; i < num_sigma_; ++i) {
        double weight = (i == 0) ? lambda_ / (n_ + lambda_) + (1 - alpha_*alpha_ + beta_) : 1.0 / (2 * (n_ + lambda_));
        Eigen::VectorXd diff = sigmaPoints.col(i) - mean;
        cov += weight * diff * diff.transpose();
    }
}

Eigen::VectorXd UnscentedKalmanFilterJerk::processModel(const Eigen::VectorXd& state, float dt) {
    Eigen::VectorXd newState = state;
    // 位置更新
    newState.head<3>() += state.segment<3>(3) * dt + 0.5 * state.segment<3>(6) * dt * dt + (1.0/6.0) * state.tail<3>() * dt * dt * dt;
    // 速度更新
    newState.segment<3>(3) += state.segment<3>(6) * dt + 0.5 * state.tail<3>() * dt * dt;
    // 加速度更新
    newState.segment<3>(6) += state.tail<3>() * dt;
    // jerk保持不变
    return newState;
}

Eigen::VectorXd UnscentedKalmanFilterJerk::measurementModel(const Eigen::VectorXd& state) {
    return state.head<3>();
}
