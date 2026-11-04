/// @file
/// @brief Interacting multiple model (IMM) filter over CTRV unscented Kalman filters.
#pragma once

#include "fusion/filters/ukf.h"

#include <vector>

namespace fusion {

/// @brief IMM filter: a bank of Ukf instances with different process noise.
///
/// Each cycle (Blom and Bar-Shalom):
/// 1. **Mixing.** Every model restarts from a mixture of all models' estimates,
///    weighted by the probability that the target switched into it.
/// 2. **Prediction** of each model.
/// 3. **Update** of each model with the measurement.
/// 4. **Mode probabilities** from each model's measurement likelihood.
/// 5. **Output**: the probability-weighted mixture (moment matched).
///
/// Switching follows a continuous-time Markov chain with ImmParams::switch_rate,
/// so the transition matrix depends on the step length: sensors at different
/// rates see the same switching behaviour, and a zero step mixes nothing.
/// Yaw is mixed with circular means.
class Imm : public Filter {
public:
    /// @param params Filter parameters; ImmParams::models defines the bank.
    /// @param noise Measurement noise.
    Imm(const FilterParams& params, const NoiseParams& noise);

    void reset(const State& x, const StateCov& P) override;
    void predict(double dt) override;
    Innovation innovation(const Measurement& m) const override;
    Innovation update(const Measurement& m) override;
    State state() const override;
    StateCov covariance() const override;
    std::unique_ptr<Filter> clone() const override { return std::make_unique<Imm>(*this); }

    /// @brief Current mode probabilities, one per model.
    const Eigen::VectorXd& modeProbabilities() const { return mu_; }

    /// @brief Transition matrix Π(dt), Π(i, j) = P(model j at t + dt | model i at t).
    Eigen::MatrixXd transition(double dt) const;

private:
    std::vector<Ukf> models_;
    Eigen::VectorXd mu_;
    Eigen::VectorXd mu_initial_;
    double switch_rate_;
};

/// @brief Moment-matched mean and covariance of a Gaussian mixture of states.
///
/// The yaw mean is circular and yaw differences are wrapped.
void mixStates(const std::vector<State>& means, const std::vector<StateCov>& covs,
               const Eigen::VectorXd& weights, State& mean, StateCov& cov);

}  // namespace fusion
