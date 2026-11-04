// End-to-end checks of headline numbers in docs/DESIGN.md, on a few seeds.

#include "fusion/config.h"
#include "fusion/experiment.h"

#include <gtest/gtest.h>

using namespace fusion;

TEST(Regression, RecommendedConfigIsAccurateAndConsistent) {
    const Config c = loadConfig(FUSION_SOURCE_DIR "/config/default.yaml");
    for (uint64_t seed = 1; seed <= 10; ++seed) {
        const RunMetrics m = runSimulation(c, seed);
        EXPECT_GT(m.nees_position, 1.0) << "seed " << seed;
        EXPECT_LT(m.nees_position, 4.0) << "seed " << seed;
        EXPECT_LT(m.rmse(2), 0.25) << "seed " << seed;
        EXPECT_LT(m.velocity_p95, 1.0) << "seed " << seed;
        EXPECT_EQ(m.id_switches, 0) << "seed " << seed;
    }
}

TEST(Regression, PipelineBeatsTheNaiveTrackerOnVelocity) {
    const Config naive = loadConfig(FUSION_SOURCE_DIR "/config/naive.yaml");
    Config ours = loadConfig(FUSION_SOURCE_DIR "/config/default.yaml");
    ours.tracker.association = "oracle";
    double vn = 0, vo = 0;
    for (uint64_t seed = 1; seed <= 10; ++seed) {
        vn += runSimulation(naive, seed).rmse.tail<2>().norm();
        vo += runSimulation(ours, seed).rmse.tail<2>().norm();
    }
    EXPECT_LT(vo, 0.85 * vn);
}

TEST(Regression, RobustUkfSurvivesWhereTheNaiveOneDiverges) {
    const Config robust =
        loadConfig(FUSION_SOURCE_DIR "/config/experiments/4_radar_rate/radar_10hz_ukf.yaml");
    const Config naive =
        loadConfig(FUSION_SOURCE_DIR "/config/experiments/4_radar_rate/radar_10hz_naive_ukf.yaml");
    const RunMetrics r = runSimulation(robust, 1), n = runSimulation(naive, 1);
    EXPECT_LT(r.rmse(2), 1.0);
    EXPECT_GT(n.rmse(2), 10.0);
}
