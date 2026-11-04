#include "fusion/filters/ukf.h"

#include "fusion/models.h"

#include <Eigen/Cholesky>
#include <Eigen/LU>

#include <cmath>

namespace fusion {
namespace {

constexpr int kAugDim = 7;  // state + [nu_a, nu_yawdd]

// Lower-triangular square root of P, repairing P first if it is not positive
// definite.
Eigen::MatrixXd squareRoot(Eigen::MatrixXd& P, int& repairs) {
    Eigen::LLT<Eigen::MatrixXd> llt(P);
    if (llt.info() != Eigen::Success) {
        if (conditionCovariance(P)) ++repairs;
        llt.compute(P);
    }
    return llt.matrixL();
}

// Columns x ± sqrt(n + lambda)·L.col(i) around x.
Eigen::MatrixXd sigmaPoints(const Eigen::VectorXd& x, const Eigen::MatrixXd& L, double lambda) {
    const int n = static_cast<int>(x.size());
    Eigen::MatrixXd X(n, 2 * n + 1);
    X.col(0) = x;
    const double scale = std::sqrt(n + lambda);
    for (int i = 0; i < n; ++i) {
        X.col(i + 1) = x + scale * L.col(i);
        X.col(i + 1 + n) = x - scale * L.col(i);
    }
    return X;
}

}  // namespace

SigmaWeights sigmaWeights(int n, const SigmaPointParams& params) {
    SigmaWeights w;
    const int count = 2 * n + 1;
    if (params.scheme == "julier") {
        w.lambda = 3.0 - n;
        w.wm = Eigen::VectorXd::Constant(count, 0.5 / (w.lambda + n));
        w.wm(0) = w.lambda / (w.lambda + n);
        w.wc = w.wm;
    } else {
        const double a2 = params.alpha * params.alpha;
        w.lambda = a2 * (n + params.kappa) - n;
        w.wm = Eigen::VectorXd::Constant(count, 0.5 / (w.lambda + n));
        w.wm(0) = w.lambda / (w.lambda + n);
        w.wc = w.wm;
        w.wc(0) += 1.0 - a2 + params.beta;
    }
    return w;
}

Eigen::VectorXd sigmaMean(const Eigen::MatrixXd& points, const Eigen::VectorXd& wm,
                          int angle_row) {
    Eigen::VectorXd mean = points * wm;
    if (angle_row >= 0) {
        const double s = points.row(angle_row).array().sin().matrix().dot(wm);
        const double c = points.row(angle_row).array().cos().matrix().dot(wm);
        mean(angle_row) = std::atan2(s, c);
    }
    return mean;
}

Ukf::Ukf(const SigmaPointParams& sigma, double accel_noise, double yaw_accel_noise,
         const NoiseParams& noise)
    : sigma_(sigma),
      accel_noise_(accel_noise),
      yaw_accel_noise_(yaw_accel_noise),
      noise_(noise),
      aug_weights_(sigmaWeights(kAugDim, sigma)),
      state_weights_(sigmaWeights(kStateDim, sigma)) {}

void Ukf::reset(const State& x, const StateCov& P) {
    x_ = x;
    x_(kYaw) = wrapAngle(x_(kYaw));
    P_ = P;
}

void Ukf::predict(double dt) {
    if (dt <= 0.0) return;

    // Augment with the two acceleration noises so that they pass through the
    // nonlinear model with the state.
    Eigen::VectorXd x_aug = Eigen::VectorXd::Zero(kAugDim);
    x_aug.head<kStateDim>() = x_;
    Eigen::MatrixXd P_aug = Eigen::MatrixXd::Zero(kAugDim, kAugDim);
    P_aug.topLeftCorner<kStateDim, kStateDim>() = P_;
    P_aug(5, 5) = accel_noise_ * accel_noise_;
    P_aug(6, 6) = yaw_accel_noise_ * yaw_accel_noise_;

    const Eigen::MatrixXd L = squareRoot(P_aug, repairs_);

    const Eigen::MatrixXd X = sigmaPoints(x_aug, L, aug_weights_.lambda);
    Eigen::MatrixXd Xp(kStateDim, X.cols());
    for (int i = 0; i < X.cols(); ++i)
        Xp.col(i) = ctrvPredict(X.col(i).head<kStateDim>(), dt, X(5, i), X(6, i));

    x_ = sigmaMean(Xp, aug_weights_.wm, kYaw);
    Eigen::MatrixXd P = Eigen::MatrixXd::Zero(kStateDim, kStateDim);
    for (int i = 0; i < Xp.cols(); ++i) {
        State d = Xp.col(i) - x_;
        d(kYaw) = wrapAngle(d(kYaw));
        P += aug_weights_.wc(i) * d * d.transpose();
    }
    if (conditionCovariance(P)) ++repairs_;
    P_ = P;
}

void Ukf::radarTransform(const EgoPose& ego, Eigen::VectorXd& z_pred, Eigen::MatrixXd& S,
                         Eigen::MatrixXd& Tc) const {
    Eigen::MatrixXd P = P_;
    int repairs = 0;
    const Eigen::MatrixXd L = squareRoot(P, repairs);
    const Eigen::MatrixXd X = sigmaPoints(x_, L, state_weights_.lambda);

    Eigen::MatrixXd Z(3, X.cols());
    for (int i = 0; i < X.cols(); ++i) Z.col(i) = measure(Sensor::Radar, X.col(i), ego);
    z_pred = sigmaMean(Z, state_weights_.wm, 1);

    S = measurementNoise(Sensor::Radar, noise_);
    Tc = Eigen::MatrixXd::Zero(kStateDim, 3);
    for (int i = 0; i < X.cols(); ++i) {
        const Eigen::VectorXd dz = residual(Sensor::Radar, Z.col(i), z_pred);
        State dx = X.col(i) - x_;
        dx(kYaw) = wrapAngle(dx(kYaw));
        S += state_weights_.wc(i) * dz * dz.transpose();
        Tc += state_weights_.wc(i) * dx * dz.transpose();
    }
}

Innovation Ukf::innovation(const Measurement& m) const {
    if (m.sensor == Sensor::Lidar) {
        const Eigen::MatrixXd H = measurementJacobian(Sensor::Lidar, x_, m.ego);
        return {m.z - measure(Sensor::Lidar, x_, m.ego), H * P_ * H.transpose() + measurementNoise(Sensor::Lidar, noise_)};
    }
    Eigen::VectorXd z_pred;
    Eigen::MatrixXd S, Tc;
    radarTransform(m.ego, z_pred, S, Tc);
    return {residual(Sensor::Radar, m.z, z_pred), S};
}

Innovation Ukf::update(const Measurement& m) {
    if (m.sensor == Sensor::Lidar) {
        // Linear model: the unscented transform would reproduce the Kalman
        // update exactly, so use the cheaper closed form with the Joseph
        // covariance update.
        const Eigen::MatrixXd H = measurementJacobian(Sensor::Lidar, x_, m.ego);
        const Eigen::MatrixXd R = measurementNoise(Sensor::Lidar, noise_);
        Innovation inn{m.z - measure(Sensor::Lidar, x_, m.ego), H * P_ * H.transpose() + R};
        const Eigen::MatrixXd K = inn.S.ldlt().solve(H * P_).transpose();
        x_ += K * inn.y;
        x_(kYaw) = wrapAngle(x_(kYaw));
        const Eigen::MatrixXd I_KH = Eigen::MatrixXd::Identity(kStateDim, kStateDim) - K * H;
        Eigen::MatrixXd P = I_KH * P_ * I_KH.transpose() + K * R * K.transpose();
        if (conditionCovariance(P)) ++repairs_;
        P_ = P;
        return inn;
    }
    Eigen::VectorXd z_pred;
    Eigen::MatrixXd S, Tc;
    radarTransform(m.ego, z_pred, S, Tc);
    Innovation inn{residual(Sensor::Radar, m.z, z_pred), S};
    const Eigen::MatrixXd K = S.ldlt().solve(Tc.transpose()).transpose();
    x_ += K * inn.y;
    x_(kYaw) = wrapAngle(x_(kYaw));
    Eigen::MatrixXd P = P_ - K * S * K.transpose();
    if (conditionCovariance(P)) ++repairs_;
    P_ = P;
    return inn;
}

// ---------------------------------------------------------------------------
// Naive variant. Deliberately without the robustness measures; see the header.

NaiveUkf::NaiveUkf(double accel_noise, double yaw_accel_noise, const NoiseParams& noise)
    : accel_noise_(accel_noise), yaw_accel_noise_(yaw_accel_noise), noise_(noise) {
    lambda_ = 3 - aug_dim_;
    sigma_pred_ = Eigen::MatrixXd::Zero(kStateDim, 2 * aug_dim_ + 1);
    weights_ = Eigen::VectorXd::Constant(2 * aug_dim_ + 1, 0.5 / (lambda_ + aug_dim_));
    weights_(0) = lambda_ / (lambda_ + aug_dim_);
}

void NaiveUkf::reset(const State& x, const StateCov& P) {
    x_ = x;
    P_ = P;
    predicted_ = false;
}

void NaiveUkf::prediction(double dt) {
    Eigen::VectorXd x_aug = Eigen::VectorXd::Zero(aug_dim_);
    x_aug.head(kStateDim) = x_;
    Eigen::MatrixXd P_aug = Eigen::MatrixXd::Zero(aug_dim_, aug_dim_);
    P_aug.topLeftCorner(kStateDim, kStateDim) = P_;
    P_aug(5, 5) = accel_noise_ * accel_noise_;
    P_aug(6, 6) = yaw_accel_noise_ * yaw_accel_noise_;
    const Eigen::MatrixXd L = P_aug.llt().matrixL();  // no check
    const Eigen::MatrixXd X = sigmaPoints(x_aug, L, lambda_);
    for (int i = 0; i < X.cols(); ++i)
        sigma_pred_.col(i) = ctrvPredict(X.col(i).head<kStateDim>(), dt, X(5, i), X(6, i));
    x_ = sigma_pred_ * weights_;  // arithmetic mean, yaw included
    P_.setZero();
    for (int i = 0; i < sigma_pred_.cols(); ++i) {
        State d = sigma_pred_.col(i) - x_;
        d(kYaw) = wrapAngle(d(kYaw));
        P_ += weights_(i) * d * d.transpose();
    }
}

void NaiveUkf::predict(double dt) {
    if (dt <= 0.0) return;
    prediction(dt);
    predicted_ = true;
}

Innovation NaiveUkf::innovation(const Measurement& m) const {
    if (m.sensor == Sensor::Lidar) {
        const Eigen::MatrixXd H = measurementJacobian(Sensor::Lidar, x_, m.ego);
        return {m.z - measure(Sensor::Lidar, x_, m.ego), H * P_ * H.transpose() + measurementNoise(Sensor::Lidar, noise_)};
    }
    // Uses the stored predicted sigma points.
    Eigen::MatrixXd Z(3, sigma_pred_.cols());
    for (int i = 0; i < Z.cols(); ++i) Z.col(i) = measure(Sensor::Radar, sigma_pred_.col(i), m.ego);
    const Eigen::VectorXd z_pred = Z * weights_;
    Eigen::MatrixXd S = measurementNoise(Sensor::Radar, noise_);
    for (int i = 0; i < Z.cols(); ++i) {
        const Eigen::VectorXd dz = residual(Sensor::Radar, Z.col(i), z_pred);
        S += weights_(i) * dz * dz.transpose();
    }
    return {residual(Sensor::Radar, m.z, z_pred), S};
}

Innovation NaiveUkf::update(const Measurement& m) {
    // Refreshes the predicted sigma points before every update, with a zero-
    // length prediction when no time has passed (second sensor at a time stamp).
    if (!predicted_) prediction(0.0);
    predicted_ = false;

    if (m.sensor == Sensor::Lidar) {
        const Eigen::MatrixXd H = measurementJacobian(Sensor::Lidar, x_, m.ego);
        const Eigen::MatrixXd R = measurementNoise(Sensor::Lidar, noise_);
        Innovation inn{m.z - measure(Sensor::Lidar, x_, m.ego), H * P_ * H.transpose() + R};
        const Eigen::MatrixXd K = P_ * H.transpose() * inn.S.inverse();
        x_ += K * inn.y;
        P_ = (Eigen::MatrixXd::Identity(kStateDim, kStateDim) - K * H) * P_;
        return inn;
    }
    const Innovation inn = innovation(m);
    Eigen::MatrixXd Z(3, sigma_pred_.cols());
    for (int i = 0; i < Z.cols(); ++i) Z.col(i) = measure(Sensor::Radar, sigma_pred_.col(i), m.ego);
    const Eigen::VectorXd z_pred = Z * weights_;
    Eigen::MatrixXd Tc = Eigen::MatrixXd::Zero(kStateDim, 3);
    for (int i = 0; i < Z.cols(); ++i) {
        const Eigen::VectorXd dz = residual(Sensor::Radar, Z.col(i), z_pred);
        State dx = sigma_pred_.col(i) - x_;
        dx(kYaw) = wrapAngle(dx(kYaw));
        Tc += weights_(i) * dx * dz.transpose();
    }
    const Eigen::MatrixXd K = Tc * inn.S.inverse();
    x_ += K * inn.y;
    x_(kYaw) = wrapAngle(x_(kYaw));
    P_ -= K * inn.S * K.transpose();
    return inn;
}

}  // namespace fusion
