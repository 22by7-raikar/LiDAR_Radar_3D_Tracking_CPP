#include "fusion/filters/ekf.h"
#include "fusion/filters/imm.h"
#include "fusion/filters/ukf.h"
#include "fusion/models.h"

#include <Eigen/Cholesky>
#include <Eigen/LU>
#include <gtest/gtest.h>

#include <random>

using namespace fusion;

namespace {

Measurement lidarAt(double x, double y, double t = 0.0) {
    Measurement m;
    m.sensor = Sensor::Lidar;
    m.time = t;
    m.z = Eigen::Vector2d(x, y);
    return m;
}

StateCov somePositiveDefinite() {
    Eigen::Matrix<double, 5, 5> A;
    A << 1.0, 0.2, 0.0, 0.1, 0.0, 0.0, 0.8, 0.1, 0.0, 0.0, 0.0, 0.0, 1.5, 0.2, 0.1, 0.0, 0.0,
        0.0, 0.3, 0.05, 0.0, 0.0, 0.0, 0.0, 0.2;
    return A * A.transpose();
}

}  // namespace

TEST(SigmaPoints, WeightsSumToOneAndReproduceMeanAndCovariance) {
    for (const std::string scheme : {"julier", "merwe"}) {
        SigmaPointParams p;
        p.scheme = scheme;
        const int n = 5;
        const SigmaWeights w = sigmaWeights(n, p);
        EXPECT_NEAR(w.wm.sum(), 1.0, 1e-12) << scheme;

        // For a linear map the unscented transform is exact.
        Eigen::VectorXd mean(n);
        mean << 1, 2, 3, 0.5, 0.1;
        const Eigen::MatrixXd P = somePositiveDefinite();
        const Eigen::MatrixXd L = P.llt().matrixL();
        Eigen::MatrixXd X(n, 2 * n + 1);
        X.col(0) = mean;
        for (int i = 0; i < n; ++i) {
            X.col(i + 1) = mean + std::sqrt(n + w.lambda) * L.col(i);
            X.col(i + 1 + n) = mean - std::sqrt(n + w.lambda) * L.col(i);
        }
        const Eigen::VectorXd m = X * w.wm;
        Eigen::MatrixXd C = Eigen::MatrixXd::Zero(n, n);
        for (int i = 0; i < X.cols(); ++i) C += w.wc(i) * (X.col(i) - m) * (X.col(i) - m).transpose();
        EXPECT_LT((m - mean).norm(), 1e-12) << scheme;
        // Merwe's covariance weight adds (1 - alpha² + beta) to the centre
        // point, which sits on the mean and so changes nothing here.
        EXPECT_LT((C - P).cwiseAbs().maxCoeff(), 1e-9) << scheme;
    }
}

TEST(SigmaPoints, CircularMeanAcrossPi) {
    Eigen::MatrixXd X(1, 3);
    X << M_PI - 0.1, -M_PI + 0.1, M_PI;
    const Eigen::Vector3d w(1.0 / 3, 1.0 / 3, 1.0 / 3);
    // The arithmetic mean would be about pi/3; the circular mean is pi.
    EXPECT_NEAR(std::abs(sigmaMean(X, w, 0)(0)), M_PI, 1e-9);
}

TEST(Ukf, LidarUpdateEqualsLinearKalmanFilter) {
    NoiseParams noise;
    Ukf ukf(SigmaPointParams{}, 1.0, 1.0, noise);
    State x;
    x << 1.0, 2.0, 3.0, 0.4, 0.0;
    const StateCov P = somePositiveDefinite();
    ukf.reset(x, P);
    const Measurement m = lidarAt(1.3, 1.8);
    ukf.update(m);

    Eigen::Matrix<double, 2, 5> H = Eigen::Matrix<double, 2, 5>::Zero();
    H(0, 0) = H(1, 1) = 1.0;
    const Eigen::Matrix2d R = Eigen::Matrix2d::Identity() * noise.lidar_std * noise.lidar_std;
    const Eigen::Matrix2d S = H * P * H.transpose() + R;
    const Eigen::Matrix<double, 5, 2> K = P * H.transpose() * S.inverse();
    const State x_kf = x + K * (m.z - H * x);
    const StateCov P_kf = (StateCov::Identity() - K * H) * P;
    EXPECT_LT((ukf.state() - x_kf).norm(), 1e-12);
    EXPECT_LT((ukf.covariance() - P_kf).cwiseAbs().maxCoeff(), 1e-12);
}

TEST(Ukf, MatchesEkfWhenUncertaintyIsSmall) {
    NoiseParams noise;
    State x;
    x << 15.0, -3.0, 8.0, 0.2, 0.05;
    const StateCov P = StateCov::Identity() * 1e-6;
    Ukf ukf(SigmaPointParams{}, 1e-3, 1e-3, noise);
    Ekf ekf(1e-3, 1e-3, noise);
    ukf.reset(x, P);
    ekf.reset(x, P);
    ukf.predict(0.1);
    ekf.predict(0.1);
    EXPECT_LT((ukf.state() - ekf.state()).norm(), 1e-6);
    Measurement m;
    m.sensor = Sensor::Radar;
    m.z = measure(Sensor::Radar, ukf.state()) + Eigen::Vector3d(0.1, 0.002, 0.05);
    EXPECT_NEAR(ukf.innovation(m).nis(), ekf.innovation(m).nis(), 1e-3);
}

TEST(Ukf, RadarTargetDirectlyBehind) {
    // A target behind the ego vehicle has bearing ≈ ±pi, so its sigma points
    // map to bearings on both sides of the wrap. The naive UKF averages them
    // arithmetically (≈ 0) and its bearing innovation variance explodes; the
    // circular mean keeps it at the sensor noise level.
    NoiseParams noise;
    State x;
    x << -12.0, 0.05, 1.0, 0.0, 0.0;
    StateCov P = StateCov::Identity() * 0.05;
    Ukf ukf(SigmaPointParams{}, 1.0, 0.5, noise);
    NaiveUkf naive(1.0, 0.5, noise);
    ukf.reset(x, P);
    naive.reset(x, P);
    ukf.predict(0.033);
    naive.predict(0.033);
    Measurement m;
    m.sensor = Sensor::Radar;
    m.z = measure(Sensor::Radar, x);
    const double s_ukf = ukf.innovation(m).S(1, 1);
    const double s_naive = naive.innovation(m).S(1, 1);
    EXPECT_LT(s_ukf, 0.01);   // (0.03 rad)² noise + ~(0.25 m / 12 m)² spread
    EXPECT_GT(s_naive, 1.0);   // dominated by ±pi residuals
}

TEST(Ukf, StaysPositiveDefiniteUnderRepeatedUpdates) {
    NoiseParams noise;
    std::mt19937 rng(3);
    std::normal_distribution<double> n(0.0, 0.15);
    Ukf ukf(SigmaPointParams{}, 5.0, 3.0, noise);
    State x;
    x << 10.0, 1.0, 3.0, 0.0, 0.0;
    ukf.reset(x, StateCov::Identity());
    for (int k = 0; k < 2000; ++k) {
        ukf.predict(1.0 / 30.0);
        ukf.update(lidarAt(10.0 + 0.1 * k + n(rng), 1.0 + n(rng), k / 30.0));
        Measurement r;
        r.sensor = Sensor::Radar;
        r.z = measure(Sensor::Radar, ukf.state());
        ukf.update(r);
    }
    EXPECT_EQ(ukf.covariance().llt().info(), Eigen::Success);
    EXPECT_LT((ukf.covariance() - ukf.covariance().transpose()).norm(), 1e-12);
}

TEST(Covariance, RepairClampsNegativeEigenvalues) {
    Eigen::MatrixXd P(2, 2);
    P << 1.0, 2.0, 2.0, 1.0;  // eigenvalues 3 and -1
    EXPECT_TRUE(conditionCovariance(P));
    EXPECT_EQ(P.llt().info(), Eigen::Success);
    Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(2, 2);
    EXPECT_FALSE(conditionCovariance(Q));
}

TEST(Imm, TransitionMatrixIsStochasticAndIdentityAtZeroStep) {
    FilterParams p;
    p.imm.models = {{0.5, 0.05}, {5.0, 3.0}, {2.0, 1.0}};
    p.imm.initial_probabilities = {0.5, 0.3, 0.2};
    Imm imm(p, NoiseParams{});
    for (const double dt : {0.0, 0.05, 1.0, 100.0}) {
        const Eigen::MatrixXd T = imm.transition(dt);
        for (int i = 0; i < 3; ++i) EXPECT_NEAR(T.row(i).sum(), 1.0, 1e-12);
    }
    EXPECT_LT((imm.transition(0.0) - Eigen::MatrixXd::Identity(3, 3)).norm(), 1e-12);
    EXPECT_NEAR(imm.transition(100.0)(0, 1), 1.0 / 3.0, 1e-9);
}

TEST(Imm, ManoeuvreRaisesTheManoeuvringModelProbability) {
    FilterParams p;  // models: cruising, manoeuvring
    Imm imm(p, NoiseParams{});
    State x;
    x << 0.0, 0.0, 10.0, 0.0, 0.0;
    StateCov P = StateCov::Identity() * 0.01;
    imm.reset(x, P);
    const double dt = 0.1;
    // Straight driving: the cruising model wins.
    for (int k = 1; k <= 30; ++k) {
        imm.predict(dt);
        imm.update(lidarAt(10.0 * dt * k, 0.0));
    }
    EXPECT_GT(imm.modeProbabilities()(0), 0.8);
    // Hard braking at 6 m/s²: the manoeuvring model takes over.
    double px = 30.0, v = 10.0;
    for (int k = 1; k <= 10; ++k) {
        px += v * dt - 0.5 * 6.0 * dt * dt;
        v -= 6.0 * dt;
        imm.predict(dt);
        imm.update(lidarAt(px, 0.0));
    }
    EXPECT_GT(imm.modeProbabilities()(1), 0.5);
}

TEST(Imm, MixStatesUsesCircularYaw) {
    State a = State::Zero(), b = State::Zero();
    a(kYaw) = M_PI - 0.1;
    b(kYaw) = -M_PI + 0.1;
    State m;
    StateCov C;
    mixStates({a, b}, {StateCov::Identity() * 0.01, StateCov::Identity() * 0.01},
              Eigen::Vector2d(0.5, 0.5), m, C);
    EXPECT_NEAR(std::abs(m(kYaw)), M_PI, 1e-9);
    EXPECT_NEAR(C(kYaw, kYaw), 0.01 + 0.01, 1e-9);  // spread of ±0.1 around pi
}
