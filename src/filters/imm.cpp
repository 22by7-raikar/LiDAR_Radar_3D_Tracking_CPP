#include "fusion/filters/imm.h"

#include <cmath>

namespace fusion {

void mixStates(const std::vector<State>& means, const std::vector<StateCov>& covs,
               const Eigen::VectorXd& weights, State& mean, StateCov& cov) {
    mean.setZero();
    double s = 0.0, c = 0.0;
    for (size_t i = 0; i < means.size(); ++i) {
        mean += weights(i) * means[i];
        s += weights(i) * std::sin(means[i](kYaw));
        c += weights(i) * std::cos(means[i](kYaw));
    }
    mean(kYaw) = std::atan2(s, c);
    cov.setZero();
    for (size_t i = 0; i < means.size(); ++i) {
        State d = means[i] - mean;
        d(kYaw) = wrapAngle(d(kYaw));
        cov += weights(i) * (covs[i] + d * d.transpose());
    }
}

Imm::Imm(const FilterParams& params, const NoiseParams& noise)
    : switch_rate_(params.imm.switch_rate) {
    for (const auto& m : params.imm.models)
        models_.emplace_back(params.sigma, m.accel_noise, m.yaw_accel_noise, noise);
    mu_initial_ = Eigen::Map<const Eigen::VectorXd>(params.imm.initial_probabilities.data(),
                                                     params.imm.initial_probabilities.size());
    mu_initial_ /= mu_initial_.sum();
    mu_ = mu_initial_;
}

void Imm::reset(const State& x, const StateCov& P) {
    for (auto& m : models_) m.reset(x, P);
    mu_ = mu_initial_;
}

Eigen::MatrixXd Imm::transition(double dt) const {
    // exp(dt·Λ) for a generator with equal switching rates r between all
    // N models: stay = 1/N + (N−1)/N·e^(−N·r·dt), switch = (1 − e^(−N·r·dt))/N.
    const int n = static_cast<int>(models_.size());
    const double decay = std::exp(-n * switch_rate_ * dt);
    const double off = (1.0 - decay) / n;
    Eigen::MatrixXd T = Eigen::MatrixXd::Constant(n, n, off);
    T.diagonal().setConstant(1.0 / n + (n - 1.0) / n * decay);
    return T;
}

void Imm::predict(double dt) {
    if (dt <= 0.0) return;
    const int n = static_cast<int>(models_.size());
    const Eigen::MatrixXd T = transition(dt);
    const Eigen::VectorXd c = T.transpose() * mu_;  // predicted mode probabilities

    std::vector<State> means;
    std::vector<StateCov> covs;
    for (const auto& m : models_) {
        means.push_back(m.state());
        covs.push_back(m.covariance());
    }
    for (int j = 0; j < n; ++j) {
        Eigen::VectorXd w(n);
        for (int i = 0; i < n; ++i) w(i) = T(i, j) * mu_(i) / std::max(c(j), 1e-300);
        State x0;
        StateCov P0;
        mixStates(means, covs, w, x0, P0);
        models_[j].reset(x0, P0);
        models_[j].predict(dt);
    }
    mu_ = c;
}

Innovation Imm::innovation(const Measurement& m) const {
    std::vector<Innovation> parts;
    for (const auto& model : models_) parts.push_back(model.innovation(m));
    Innovation out{Eigen::VectorXd::Zero(parts[0].y.size()),
                   Eigen::MatrixXd::Zero(parts[0].S.rows(), parts[0].S.cols())};
    for (size_t i = 0; i < parts.size(); ++i) out.y += mu_(i) * parts[i].y;
    for (size_t i = 0; i < parts.size(); ++i) {
        const Eigen::VectorXd d = parts[i].y - out.y;
        out.S += mu_(i) * (parts[i].S + d * d.transpose());
    }
    return out;
}

Innovation Imm::update(const Measurement& m) {
    const Innovation combined = innovation(m);
    const int n = static_cast<int>(models_.size());
    Eigen::VectorXd log_w(n);
    for (int i = 0; i < n; ++i)
        log_w(i) = std::log(std::max(mu_(i), 1e-300)) + models_[i].update(m).logLikelihood();
    // Normalise in the log domain: likelihoods of a far-off radar return can
    // underflow to zero for every model.
    const double top = log_w.maxCoeff();
    mu_ = (log_w.array() - top).exp();
    mu_ /= mu_.sum();
    // Keep every model alive so that a switch can always be detected.
    mu_ = mu_.cwiseMax(1e-6);
    mu_ /= mu_.sum();
    return combined;
}

State Imm::state() const {
    State x;
    StateCov P;
    std::vector<State> means;
    std::vector<StateCov> covs;
    for (const auto& m : models_) {
        means.push_back(m.state());
        covs.push_back(m.covariance());
    }
    mixStates(means, covs, mu_, x, P);
    return x;
}

StateCov Imm::covariance() const {
    State x;
    StateCov P;
    std::vector<State> means;
    std::vector<StateCov> covs;
    for (const auto& m : models_) {
        means.push_back(m.state());
        covs.push_back(m.covariance());
    }
    mixStates(means, covs, mu_, x, P);
    return P;
}

}  // namespace fusion
