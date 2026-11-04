#include "fusion/sim/vehicle.h"

#include <cmath>

namespace fusion {

Eigen::Vector2d VehicleState::velocity() const {
    return {v * std::cos(yaw), v * std::sin(yaw)};
}

Vehicle::Vehicle(const VehicleParams& params, int id)
    : params_(params), id_(id), x_(params.x), y_(params.y), v_(params.velocity),
      yaw_(params.angle) {}

void Vehicle::step(double time, double dt) {
    // Small tolerance so an instruction at t = 0.5 s fires on the tick that
    // lands on 0.5 s despite rounding in the tick time.
    while (next_instruction_ < params_.instructions.size() &&
           time + 1e-9 >= params_.instructions[next_instruction_].time) {
        acceleration_ = params_.instructions[next_instruction_].acceleration;
        steering_ = params_.instructions[next_instruction_].steering;
        ++next_instruction_;
    }
    const double v_mid = v_ + 0.5 * acceleration_ * dt;
    const double yaw_rate = v_mid * std::tan(steering_) / params_.wheelbase;
    const double yaw_mid = yaw_ + 0.5 * yaw_rate * dt;
    x_ += v_mid * std::cos(yaw_mid) * dt;
    y_ += v_mid * std::sin(yaw_mid) * dt;
    yaw_ += yaw_rate * dt;
    v_ += acceleration_ * dt;
}

VehicleState Vehicle::state() const {
    return {x_, y_, v_, yaw_, v_ * std::tan(steering_) / params_.wheelbase};
}

}  // namespace fusion
