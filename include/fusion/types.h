/// @file
/// @brief State, measurement and angle conventions shared by every module.
#pragma once

#include <Eigen/Core>

#include <cmath>
#include <string>

namespace fusion {

/// @brief CTRV state [px, py, v, yaw, yaw_rate] in metres, m/s, rad and rad/s.
using State = Eigen::Matrix<double, 5, 1>;
/// @brief Covariance of a State.
using StateCov = Eigen::Matrix<double, 5, 5>;

/// @brief Number of state variables.
constexpr int kStateDim = 5;
/// @brief Index of the heading in State; residuals in this component are wrapped.
constexpr int kYaw = 3;

/// @brief Sensor that produced a Measurement.
enum class Sensor {
    Lidar,  ///< Position z = [x, y] in the ego frame.
    Radar,  ///< Polar z = [range, bearing, range rate] in the ego frame; range rate is relative.
};

/// @brief Lower-case sensor name, as used in CSV files and configs.
const char* sensorName(Sensor sensor);

/// @brief Measurement dimension of a sensor (2 for lidar, 3 for radar).
int measurementDim(Sensor sensor);

/// @brief Pose and velocity of the ego vehicle (and its sensors) in the world frame.
///
/// Comes from odometry. Sensors measure in the ego frame (x forward, y left);
/// tracks live in the world frame, and the measurement models use this pose
/// to connect the two.
struct EgoPose {
    double x = 0.0;    ///< Position x [m].
    double y = 0.0;    ///< Position y [m].
    double yaw = 0.0;  ///< Heading [rad].
    double vx = 0.0;   ///< Velocity x [m/s].
    double vy = 0.0;   ///< Velocity y [m/s].
};

/// @brief One detection from one sensor.
struct Measurement {
    Sensor sensor = Sensor::Lidar;  ///< Source sensor; selects the measurement model.
    double time = 0.0;              ///< Time stamp [s].
    Eigen::VectorXd z;              ///< Measured values in the ego frame, see Sensor.
    EgoPose ego;                    ///< Ego pose at #time.
    /// Simulated object that caused the detection, or -1 for clutter. Only the
    /// oracle association and the evaluation read it; the tracker never does.
    int truth_id = -1;
};

/// @brief Wraps an angle to [-pi, pi].
///
/// Uses std::remainder, so it costs the same for any input and returns NaN
/// for NaN instead of looping forever.
inline double wrapAngle(double angle) { return std::remainder(angle, 2.0 * M_PI); }

/// @brief Velocity vector [vx, vy] of a CTRV state.
inline Eigen::Vector2d velocityXY(const State& x) {
    return {x(2) * std::cos(x(kYaw)), x(2) * std::sin(x(kYaw))};
}

}  // namespace fusion
