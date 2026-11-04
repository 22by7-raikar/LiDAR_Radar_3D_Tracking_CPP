/// @file
/// @brief CTRV motion model and the lidar and radar measurement models.
#pragma once

#include "fusion/config.h"
#include "fusion/types.h"

#include <Eigen/Core>

namespace fusion {

/// @brief Yaw rates below this are treated as straight-line motion [rad/s].
constexpr double kStraightYawRate = 1e-4;

/// @brief Constant turn rate and velocity (CTRV) motion over @p dt seconds.
///
/// @p nu_a and @p nu_yawdd are the longitudinal and yaw accelerations, held
/// constant over the step; they enter the position and yaw through the usual
/// dt²/2 terms. Yaw is not wrapped.
State ctrvPredict(const State& x, double dt, double nu_a = 0.0, double nu_yawdd = 0.0);

/// @brief Jacobian of ctrvPredict() with respect to the state, at zero noise.
StateCov ctrvJacobian(const State& x, double dt);

/// @brief Jacobian of ctrvPredict() with respect to [nu_a, nu_yawdd].
Eigen::Matrix<double, 5, 2> ctrvNoiseJacobian(const State& x, double dt);

/// @brief Predicted measurement h(x) of a world-frame state seen from @p ego.
///
/// Lidar: the position in the ego frame. Radar: range and bearing of that
/// position, and the range rate of the velocity relative to the ego vehicle.
Eigen::VectorXd measure(Sensor sensor, const State& x, const EgoPose& ego = {});

/// @brief Jacobian of measure() with respect to the state.
Eigen::MatrixXd measurementJacobian(Sensor sensor, const State& x, const EgoPose& ego = {});

/// @brief Measurement noise covariance R of @p sensor.
Eigen::MatrixXd measurementNoise(Sensor sensor, const NoiseParams& noise);

/// @brief Residual a − b with the radar bearing wrapped to [−pi, pi].
Eigen::VectorXd residual(Sensor sensor, const Eigen::VectorXd& a, const Eigen::VectorXd& b);

/// @brief World-frame position of a detection (radar converted from polar).
Eigen::Vector2d detectionPosition(const Measurement& m);

/// @brief Covariance of detectionPosition() in the world frame, linearised for radar.
Eigen::Matrix2d detectionPositionCov(const Measurement& m, const NoiseParams& noise);

}  // namespace fusion
