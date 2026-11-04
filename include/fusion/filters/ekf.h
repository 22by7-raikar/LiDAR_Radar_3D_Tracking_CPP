/// @file
/// @brief Extended Kalman filter with the CTRV model.
#pragma once

#include "fusion/filters/filter.h"

namespace fusion {

/// @brief Extended Kalman filter on the CTRV state.
///
/// Linearises the motion model with ctrvJacobian() and the radar model with
/// measurementJacobian(). Process noise is G·diag(σa², σψ̈²)·Gᵀ with G from
/// ctrvNoiseJacobian(); covariance updates use the Joseph form. Included as
/// the reference point for what the unscented transform gains.
class Ekf : public Filter {
public:
    /// @param accel_noise Longitudinal acceleration noise [m/s²].
    /// @param yaw_accel_noise Yaw acceleration noise [rad/s²].
    /// @param noise Measurement noise.
    Ekf(double accel_noise, double yaw_accel_noise, const NoiseParams& noise);

    void reset(const State& x, const StateCov& P) override;
    void predict(double dt) override;
    Innovation innovation(const Measurement& m) const override;
    Innovation update(const Measurement& m) override;
    State state() const override { return x_; }
    StateCov covariance() const override { return P_; }
    std::unique_ptr<Filter> clone() const override { return std::make_unique<Ekf>(*this); }

private:
    double accel_noise_, yaw_accel_noise_;
    NoiseParams noise_;
    State x_ = State::Zero();
    StateCov P_ = StateCov::Identity();
};

}  // namespace fusion
