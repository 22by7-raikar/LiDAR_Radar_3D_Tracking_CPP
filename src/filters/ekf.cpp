#include "fusion/filters/ekf.h"

#include "fusion/models.h"

#include <Eigen/Cholesky>

namespace fusion {

Ekf::Ekf(double accel_noise, double yaw_accel_noise, const NoiseParams& noise)
    : accel_noise_(accel_noise), yaw_accel_noise_(yaw_accel_noise), noise_(noise) {}

void Ekf::reset(const State& x, const StateCov& P) {
    x_ = x;
    x_(kYaw) = wrapAngle(x_(kYaw));
    P_ = P;
}

void Ekf::predict(double dt) {
    if (dt <= 0.0) return;
    const StateCov F = ctrvJacobian(x_, dt);
    const Eigen::Matrix<double, 5, 2> G = ctrvNoiseJacobian(x_, dt);
    const Eigen::Vector2d q(accel_noise_ * accel_noise_, yaw_accel_noise_ * yaw_accel_noise_);
    x_ = ctrvPredict(x_, dt);
    x_(kYaw) = wrapAngle(x_(kYaw));
    Eigen::MatrixXd P = F * P_ * F.transpose() + G * q.asDiagonal() * G.transpose();
    conditionCovariance(P);
    P_ = P;
}

Innovation Ekf::innovation(const Measurement& m) const {
    const Eigen::MatrixXd H = measurementJacobian(m.sensor, x_, m.ego);
    return {residual(m.sensor, m.z, measure(m.sensor, x_, m.ego)),
            H * P_ * H.transpose() + measurementNoise(m.sensor, noise_)};
}

Innovation Ekf::update(const Measurement& m) {
    const Eigen::MatrixXd H = measurementJacobian(m.sensor, x_, m.ego);
    const Eigen::MatrixXd R = measurementNoise(m.sensor, noise_);
    Innovation inn{residual(m.sensor, m.z, measure(m.sensor, x_, m.ego)), H * P_ * H.transpose() + R};
    const Eigen::MatrixXd K = inn.S.ldlt().solve(H * P_).transpose();
    x_ += K * inn.y;
    x_(kYaw) = wrapAngle(x_(kYaw));
    const Eigen::MatrixXd I_KH = Eigen::MatrixXd::Identity(kStateDim, kStateDim) - K * H;
    Eigen::MatrixXd P = I_KH * P_ * I_KH.transpose() + K * R * K.transpose();
    conditionCovariance(P);
    P_ = P;
    return inn;
}

}  // namespace fusion
