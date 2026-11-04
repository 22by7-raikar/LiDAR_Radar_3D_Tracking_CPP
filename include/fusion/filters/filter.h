/// @file
/// @brief Common interface of the state estimators and the factory that builds them.
#pragma once

#include "fusion/config.h"
#include "fusion/types.h"

#include <Eigen/Core>

#include <memory>

namespace fusion {

/// @brief Predicted measurement and its covariance, for gating and likelihoods.
struct Innovation {
    Eigen::VectorXd y;  ///< Residual z − ẑ (bearing wrapped for radar).
    Eigen::MatrixXd S;  ///< Innovation covariance.

    /// @brief Normalised innovation squared yᵀ S⁻¹ y.
    double nis() const;
    /// @brief Gaussian log-likelihood log N(y; 0, S).
    double logLikelihood() const;
};

/// @brief A recursive state estimator for one target.
///
/// The tracker calls predict() up to each measurement time and then update().
/// innovation() is const so that association can score a measurement against
/// many tracks before committing to one.
class Filter {
public:
    virtual ~Filter() = default;

    /// @brief Sets the state and covariance (used at initialisation).
    virtual void reset(const State& x, const StateCov& P) = 0;

    /// @brief Propagates the estimate by @p dt seconds; dt ≤ 0 does nothing.
    virtual void predict(double dt) = 0;

    /// @brief Innovation of @p m against the current estimate, without changing it.
    virtual Innovation innovation(const Measurement& m) const = 0;

    /// @brief Corrects the estimate with @p m.
    /// @return The innovation before the correction (for NIS statistics).
    virtual Innovation update(const Measurement& m) = 0;

    /// @brief Current state estimate.
    virtual State state() const = 0;

    /// @brief Current state covariance.
    virtual StateCov covariance() const = 0;

    /// @brief Deep copy (used to predict ahead for evaluation and display).
    virtual std::unique_ptr<Filter> clone() const = 0;
};

/// @brief Builds the filter selected by FilterParams::type.
std::unique_ptr<Filter> makeFilter(const FilterParams& params, const NoiseParams& noise);

/// @brief Forces symmetry and repairs a covariance that has lost positive
/// definiteness by clamping its eigenvalues at @p floor.
/// @return true if a repair was needed.
bool conditionCovariance(Eigen::MatrixXd& P, double floor = 1e-9);

}  // namespace fusion
