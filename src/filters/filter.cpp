#include "fusion/filters/filter.h"

#include "fusion/filters/ekf.h"
#include "fusion/filters/imm.h"
#include "fusion/filters/ukf.h"

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>

#include <cmath>
#include <stdexcept>

namespace fusion {

double Innovation::nis() const { return y.dot(S.ldlt().solve(y)); }

double Innovation::logLikelihood() const {
    const auto ldlt = S.ldlt();
    const double log_det = ldlt.vectorD().array().log().sum();
    return -0.5 * (y.dot(ldlt.solve(y)) + log_det + y.size() * std::log(2.0 * M_PI));
}

bool conditionCovariance(Eigen::MatrixXd& P, double floor) {
    P = 0.5 * (P + P.transpose());
    if (P.llt().info() == Eigen::Success) return false;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(P);
    const Eigen::VectorXd values = eig.eigenvalues().cwiseMax(floor);
    P = eig.eigenvectors() * values.asDiagonal() * eig.eigenvectors().transpose();
    P = 0.5 * (P + P.transpose());
    return true;
}

std::unique_ptr<Filter> makeFilter(const FilterParams& params, const NoiseParams& noise) {
    if (params.type == "ukf")
        return std::make_unique<Ukf>(params.sigma, params.accel_noise, params.yaw_accel_noise,
                                     noise);
    if (params.type == "ekf")
        return std::make_unique<Ekf>(params.accel_noise, params.yaw_accel_noise, noise);
    if (params.type == "imm") return std::make_unique<Imm>(params, noise);
    if (params.type == "naive_ukf")
        return std::make_unique<NaiveUkf>(params.accel_noise, params.yaw_accel_noise, noise);
    throw std::invalid_argument("unknown filter type: " + params.type);
}

}  // namespace fusion
