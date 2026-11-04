#include "fusion/experiment.h"
#include "fusion/tracking/hungarian.h"
#include "fusion/tracking/tracker.h"

#include <Eigen/Cholesky>
#include <Eigen/LU>
#include <gtest/gtest.h>

#include <algorithm>
#include <numeric>
#include <random>

using namespace fusion;

namespace {

double bruteForce(const std::vector<std::vector<double>>& cost) {
    const size_t rows = cost.size(), cols = cost[0].size();
    std::vector<int> perm(std::max(rows, cols));
    std::iota(perm.begin(), perm.end(), 0);
    double best = 1e18;
    do {
        double s = 0.0;
        for (size_t r = 0; r < rows; ++r)
            if (perm[r] < static_cast<int>(cols)) s += cost[r][perm[r]];
            else if (rows <= cols) s += 1e9;  // a row must be assigned when possible
        best = std::min(best, s);
    } while (std::next_permutation(perm.begin(), perm.end()));
    return best;
}

Measurement lidarAt(double x, double y, double t, int truth = 0) {
    Measurement m;
    m.sensor = Sensor::Lidar;
    m.time = t;
    m.z = Eigen::Vector2d(x, y);
    m.truth_id = truth;
    return m;
}

}  // namespace

TEST(Hungarian, MatchesBruteForceOnRandomMatrices) {
    std::mt19937 rng(1);
    std::uniform_real_distribution<double> u(0.0, 10.0);
    for (const auto& [rows, cols] : std::vector<std::pair<int, int>>{{4, 4}, {3, 6}, {6, 3}, {1, 4}}) {
        for (int trial = 0; trial < 20; ++trial) {
            std::vector<std::vector<double>> cost(rows, std::vector<double>(cols));
            for (auto& row : cost)
                for (double& c : row) c = u(rng);
            const auto a = solveAssignment(cost);
            double total = 0.0;
            for (int r = 0; r < rows; ++r)
                if (a[r] >= 0) total += cost[r][a[r]];
            EXPECT_NEAR(total, bruteForce(cost), 1e-9);
        }
    }
}

TEST(Initialisation, TwoPointRecoversVelocityAndHeading) {
    Config c = defaultConfig();
    c.tracker.init.lateral_speed_std = 0.0;
    MultiTargetTracker tracker(c);
    State x;
    StateCov P;
    tracker.initialiseTwoPoint(lidarAt(0.0, 0.0, 0.0), lidarAt(-1.0, 1.0, 0.5), x, P);
    EXPECT_NEAR(x(0), -1.0, 1e-12);
    EXPECT_NEAR(x(2), std::sqrt(2.0) / 0.5, 1e-12);
    EXPECT_NEAR(x(kYaw), 3.0 * M_PI / 4.0, 1e-12);
    // Velocity variance per axis is 2σ²/dt².
    const double sv = 2.0 * 0.15 * 0.15 / 0.25;
    EXPECT_NEAR(P(2, 2), sv, 1e-12);
    EXPECT_EQ(P.llt().info(), Eigen::Success);
}

TEST(Initialisation, LateralPriorShrinksSidewaysVelocity) {
    Config c = defaultConfig();
    MultiTargetTracker tracker(c);
    State x;
    StateCov P;
    // Noise-like 0.3 m sideways jump over 0.2 s.
    tracker.initialiseTwoPoint(lidarAt(0.0, 0.0, 0.0), lidarAt(2.0, 0.3, 0.2), x, P);
    const double vy = x(2) * std::sin(x(kYaw));
    EXPECT_LT(std::abs(vy), 0.3 / 0.2 * 0.5);
}

TEST(Tracker, ConfirmsCoastsAndDeletes) {
    Config c = defaultConfig();
    c.tracker.confirm_hits = 3;
    c.tracker.max_coast = 0.5;
    MultiTargetTracker tracker(c);
    double t = 0.0;
    for (int k = 0; k < 10; ++k, t += 0.1) tracker.process(t, Sensor::Lidar, {lidarAt(5.0 + 1.0 * t, 0.0, t)});
    ASSERT_EQ(tracker.tracks().size(), 1u);
    EXPECT_TRUE(tracker.tracks()[0].confirmed);
    const int id = tracker.tracks()[0].id;
    for (int k = 0; k < 4; ++k, t += 0.1) tracker.process(t, Sensor::Lidar, {});
    ASSERT_EQ(tracker.tracks().size(), 1u);  // coasting
    EXPECT_EQ(tracker.tracks()[0].id, id);
    for (int k = 0; k < 4; ++k, t += 0.1) tracker.process(t, Sensor::Lidar, {});
    EXPECT_TRUE(tracker.tracks().empty());
}

TEST(Tracker, SingleClutterDetectionsNeverConfirm) {
    Config c = defaultConfig();
    MultiTargetTracker tracker(c);
    std::mt19937 rng(5);
    std::uniform_real_distribution<double> u(-20.0, 20.0);
    for (int k = 0; k < 100; ++k) {
        const double t = 0.1 * k;
        tracker.process(t, Sensor::Lidar, {lidarAt(u(rng), u(rng), t, -1)});
        for (const Track& tr : tracker.tracks()) EXPECT_FALSE(tr.confirmed);
    }
}

TEST(Tracker, GnnMatchesOracleWithoutClutter) {
    Config gnn = defaultConfig();
    Config oracle = gnn;
    oracle.tracker.association = "oracle";
    for (uint64_t seed = 1; seed <= 3; ++seed) {
        const RunMetrics a = runSimulation(gnn, seed), b = runSimulation(oracle, seed);
        EXPECT_EQ(a.id_switches, 0);
        EXPECT_EQ(a.false_tracks, 0.0);
        // Positions agree closely. Velocities can differ a little: during
        // car3's 4.5 rad/s swerve at 3.2 s a few radar detections fall
        // outside the 99.9 % gate, which the oracle applies anyway.
        EXPECT_LT((a.rmse - b.rmse).head<2>().cwiseAbs().maxCoeff(), 0.02);
        EXPECT_LT((a.rmse - b.rmse).tail<2>().cwiseAbs().maxCoeff(), 0.1);
    }
}

TEST(Tracker, KeepsIdentitiesThroughClutterAndMisses) {
    Config c = defaultConfig();
    c.sim.lidar.rate = 10.0;
    c.sim.radar.rate = 20.0;
    c.sim.radar.offset = 0.025;
    c.sim.step = 0.005;
    c.eval.rate = 20.0;
    c.sim.lidar.detection_probability = 0.9;
    c.sim.radar.detection_probability = 0.9;
    c.sim.radar.clutter_rate = 5.0;
    c.sim.lidar.clutter_rate = 1.0;
    int switches = 0;
    double false_tracks = 0.0;
    for (uint64_t seed = 1; seed <= 5; ++seed) {
        const RunMetrics m = runSimulation(c, seed);
        switches += m.id_switches;
        false_tracks += m.false_tracks;
        EXPECT_LT(m.ospa, 0.5);
    }
    EXPECT_LE(switches, 2);
    EXPECT_LT(false_tracks / 5.0, 0.1);
}
