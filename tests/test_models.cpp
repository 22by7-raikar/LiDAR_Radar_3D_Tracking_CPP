#include "fusion/models.h"

#include <gtest/gtest.h>

using namespace fusion;

namespace {

// Central-difference Jacobian of f at x.
template <typename F>
Eigen::MatrixXd numericJacobian(F f, const State& x, int rows) {
    Eigen::MatrixXd J(rows, kStateDim);
    for (int i = 0; i < kStateDim; ++i) {
        State a = x, b = x;
        a(i) += 1e-6;
        b(i) -= 1e-6;
        J.col(i) = (f(a) - f(b)) / 2e-6;
    }
    return J;
}

}  // namespace

TEST(Angles, WrapsToPlusMinusPi) {
    EXPECT_NEAR(wrapAngle(3.0 * M_PI / 2.0), -M_PI / 2.0, 1e-12);
    EXPECT_NEAR(wrapAngle(-7.0 * M_PI), M_PI, 1e-9);  // ±pi are the same angle
    EXPECT_NEAR(std::abs(wrapAngle(1e6)), std::abs(std::remainder(1e6, 2 * M_PI)), 1e-9);
    EXPECT_TRUE(std::isnan(wrapAngle(std::nan(""))));
}

TEST(Ctrv, StraightAndTurningMotion) {
    State x;
    x << 1.0, 2.0, 10.0, 0.0, 0.0;
    const State s = ctrvPredict(x, 0.5);
    EXPECT_NEAR(s(0), 6.0, 1e-12);
    EXPECT_NEAR(s(1), 2.0, 1e-12);

    // A quarter circle of radius v/w ends at (r, r) from the start, heading pi/2.
    x << 0.0, 0.0, 2.0, 0.0, 0.5;
    const double t = (M_PI / 2.0) / 0.5, r = 2.0 / 0.5;
    const State q = ctrvPredict(x, t);
    EXPECT_NEAR(q(0), r, 1e-9);
    EXPECT_NEAR(q(1), r, 1e-9);
    EXPECT_NEAR(q(kYaw), M_PI / 2.0, 1e-12);
}

TEST(Ctrv, JacobianMatchesNumericDifferentiation) {
    for (const double w : {0.4, -0.7, 0.01}) {
        State x;
        x << 3.0, -1.0, 7.0, 0.6, w;
        const double dt = 0.1;
        const Eigen::MatrixXd num =
            numericJacobian([&](const State& s) -> Eigen::VectorXd { return ctrvPredict(s, dt); }, x, 5);
        EXPECT_LT((ctrvJacobian(x, dt) - num).cwiseAbs().maxCoeff(), 1e-5) << "yaw rate " << w;
    }
}

TEST(Ctrv, JacobianIsContinuousAtZeroYawRate) {
    // The straight-line branch uses the w → 0 limit of the turning formulas.
    State x;
    x << 3.0, -1.0, 7.0, 0.6, 0.0;
    State y = x;
    y(4) = 1e-3;
    EXPECT_LT((ctrvJacobian(x, 0.1) - ctrvJacobian(y, 0.1)).cwiseAbs().maxCoeff(), 1e-4);
}

TEST(Radar, JacobianMatchesNumericDifferentiation) {
    State x;
    x << 12.0, -5.0, 6.0, 0.3, 0.1;
    const Eigen::MatrixXd num = numericJacobian(
        [](const State& s) -> Eigen::VectorXd { return measure(Sensor::Radar, s); }, x, 3);
    EXPECT_LT((measurementJacobian(Sensor::Radar, x) - num).cwiseAbs().maxCoeff(), 1e-6);
}

TEST(EgoFrame, LidarSeesTheRelativePosition) {
    EgoPose ego{10.0, 5.0, M_PI / 2.0, 0.0, 25.0};  // facing +y
    State x;
    x << 10.0, 25.0, 25.0, M_PI / 2.0, 0.0;  // 20 m ahead of it
    const Eigen::VectorXd z = measure(Sensor::Lidar, x, ego);
    EXPECT_NEAR(z(0), 20.0, 1e-12);
    EXPECT_NEAR(z(1), 0.0, 1e-12);
    // Same speed and heading: no relative motion.
    EXPECT_NEAR(measure(Sensor::Radar, x, ego)(2), 0.0, 1e-12);
    // Round trip through detectionPosition.
    Measurement m;
    m.sensor = Sensor::Lidar;
    m.z = z;
    m.ego = ego;
    EXPECT_LT((detectionPosition(m) - x.head<2>()).norm(), 1e-12);
}

TEST(EgoFrame, RadarJacobianMatchesNumericDifferentiation) {
    const EgoPose ego{3.0, -2.0, 0.4, 20.0, 3.0};
    State x;
    x << 30.0, 8.0, 27.0, 0.25, 0.05;
    for (const Sensor s : {Sensor::Lidar, Sensor::Radar}) {
        const Eigen::MatrixXd num = numericJacobian(
            [&](const State& v) -> Eigen::VectorXd { return measure(s, v, ego); }, x,
            measurementDim(s));
        EXPECT_LT((measurementJacobian(s, x, ego) - num).cwiseAbs().maxCoeff(), 1e-6);
    }
}

TEST(Radar, RangeRateIsRadialVelocity) {
    State x;
    x << 0.0, 10.0, 3.0, M_PI / 2.0, 0.0;  // moving straight away
    EXPECT_NEAR(measure(Sensor::Radar, x)(2), 3.0, 1e-12);
    x(kYaw) = 0.0;  // moving across the line of sight
    EXPECT_NEAR(measure(Sensor::Radar, x)(2), 0.0, 1e-12);
}

TEST(Radar, ResidualWrapsBearing) {
    Eigen::Vector3d a(10.0, M_PI - 0.01, 0.0), b(10.0, -M_PI + 0.01, 0.0);
    EXPECT_NEAR(residual(Sensor::Radar, a, b)(1), -0.02, 1e-12);
}

TEST(Radar, PositionCovarianceGrowsAcrossTheBeamWithRange) {
    NoiseParams n;
    Measurement m;
    m.sensor = Sensor::Radar;
    m.z = Eigen::Vector3d(30.0, 0.0, 0.0);
    const Eigen::Matrix2d C = detectionPositionCov(m, n);
    EXPECT_NEAR(C(0, 0), n.radar_range_std * n.radar_range_std, 1e-12);
    EXPECT_NEAR(C(1, 1), std::pow(30.0 * n.radar_bearing_std, 2), 1e-12);
}
