/// @file
/// @brief Unscented Kalman filter with the CTRV model, and a naive variant for comparison.
#pragma once

#include "fusion/filters/filter.h"

namespace fusion {

/// @brief Sigma points and weights of one unscented transform.
struct SigmaWeights {
    double lambda = 0.0;  ///< Spread parameter.
    Eigen::VectorXd wm;   ///< Mean weights.
    Eigen::VectorXd wc;   ///< Covariance weights.
};

/// @brief Weights for an n-dimensional unscented transform.
SigmaWeights sigmaWeights(int n, const SigmaPointParams& params);

/// @brief Weighted mean of sigma points, with a circular mean for row @p angle_row.
/// @param points One sigma point per column.
/// @param wm Mean weights.
/// @param angle_row Row holding an angle, or -1 for none.
Eigen::VectorXd sigmaMean(const Eigen::MatrixXd& points, const Eigen::VectorXd& wm, int angle_row);

/// @brief Unscented Kalman filter on the CTRV state.
///
/// Robustness measures (each one missing from NaiveUkf):
/// - The mean of the yaw (and of the radar bearing) is circular, so sigma
///   points on both sides of ±pi average to ±pi rather than to 0.
/// - Every update draws sigma points from the current estimate, instead of
///   reusing the points of the last prediction, which are stale after another
///   update at the same time stamp.
/// - The lidar update is linear and uses the Joseph form; gains are computed
///   with an LDLT solve instead of an explicit inverse.
/// - Covariances are kept symmetric, and a covariance whose Cholesky
///   factorisation fails is repaired rather than used as is.
/// - The sigma-point scheme is configurable; see SigmaPointParams.
class Ukf : public Filter {
public:
    /// @param sigma Sigma-point scheme.
    /// @param accel_noise Longitudinal acceleration noise [m/s²].
    /// @param yaw_accel_noise Yaw acceleration noise [rad/s²].
    /// @param noise Measurement noise.
    Ukf(const SigmaPointParams& sigma, double accel_noise, double yaw_accel_noise,
        const NoiseParams& noise);

    void reset(const State& x, const StateCov& P) override;
    void predict(double dt) override;
    Innovation innovation(const Measurement& m) const override;
    Innovation update(const Measurement& m) override;
    State state() const override { return x_; }
    StateCov covariance() const override { return P_; }
    std::unique_ptr<Filter> clone() const override { return std::make_unique<Ukf>(*this); }

    /// @brief Number of covariance repairs so far (0 in a healthy run).
    int repairs() const { return repairs_; }

private:
    // Radar unscented transform: predicted mean, innovation covariance and
    // state/measurement cross-covariance.
    void radarTransform(const EgoPose& ego, Eigen::VectorXd& z_pred, Eigen::MatrixXd& S,
                        Eigen::MatrixXd& Tc) const;

    SigmaPointParams sigma_;
    double accel_noise_, yaw_accel_noise_;
    NoiseParams noise_;
    SigmaWeights aug_weights_, state_weights_;
    State x_ = State::Zero();
    StateCov P_ = StateCov::Identity();
    int repairs_ = 0;
};

/// @brief A textbook UKF without the robustness measures of Ukf.
///
/// Julier sigma points (lambda = 3 − n_aug), arithmetic means of angles, the
/// radar update reusing the predicted sigma points (with a zero-length
/// prediction before each update to refresh them), P = (I − KH)P for lidar,
/// explicit inverses and no covariance checks. This is how a UKF is commonly
/// first written; docs/DESIGN.md measures what each missing measure costs.
class NaiveUkf : public Filter {
public:
    /// @param accel_noise Longitudinal acceleration noise [m/s²].
    /// @param yaw_accel_noise Yaw acceleration noise [rad/s²].
    /// @param noise Measurement noise.
    NaiveUkf(double accel_noise, double yaw_accel_noise, const NoiseParams& noise);

    void reset(const State& x, const StateCov& P) override;
    void predict(double dt) override;
    Innovation innovation(const Measurement& m) const override;
    Innovation update(const Measurement& m) override;
    State state() const override { return x_; }
    StateCov covariance() const override { return P_; }
    std::unique_ptr<Filter> clone() const override {
        return std::make_unique<NaiveUkf>(*this);
    }

private:
    void prediction(double dt);
    double accel_noise_, yaw_accel_noise_;
    NoiseParams noise_;
    State x_ = State::Zero();
    StateCov P_ = StateCov::Identity();
    Eigen::MatrixXd sigma_pred_;
    Eigen::VectorXd weights_;
    int aug_dim_ = 7;
    double lambda_ = -4.0;
    bool predicted_ = false;  // a prediction ran since the last update
};

}  // namespace fusion
