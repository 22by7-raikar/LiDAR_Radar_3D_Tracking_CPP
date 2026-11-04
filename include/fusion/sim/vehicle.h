/// @file
/// @brief Simulated traffic vehicle with scripted controls.
#pragma once

#include "fusion/config.h"

#include <Eigen/Core>

namespace fusion {

/// @brief True state of a simulated vehicle, in the world frame.
struct VehicleState {
    double x = 0.0;         ///< Position x [m].
    double y = 0.0;         ///< Position y [m].
    double v = 0.0;         ///< Speed [m/s].
    double yaw = 0.0;       ///< Heading [rad].
    double yaw_rate = 0.0;  ///< Heading rate [rad/s].

    /// @brief Velocity [vx, vy] [m/s].
    Eigen::Vector2d velocity() const;
};

/// @brief A vehicle following a script of acceleration and steering changes.
///
/// Kinematic bicycle model: heading rate v·tan(δ)/L for steering angle δ and
/// wheelbase L, integrated with the heading at the middle of each step.
class Vehicle {
public:
    /// @param params Start state, size and script.
    /// @param id Identity, used as Measurement::truth_id.
    Vehicle(const VehicleParams& params, int id);

    /// @brief Applies any instruction due at @p time and advances by @p dt.
    void step(double time, double dt);

    /// @brief Current true state.
    VehicleState state() const;

    /// @brief Parameters (size, name, script).
    const VehicleParams& params() const { return params_; }

    /// @brief Identity.
    int id() const { return id_; }

private:
    VehicleParams params_;
    int id_;
    double x_, y_, v_, yaw_;
    double acceleration_ = 0.0, steering_ = 0.0;
    size_t next_instruction_ = 0;
};

}  // namespace fusion
