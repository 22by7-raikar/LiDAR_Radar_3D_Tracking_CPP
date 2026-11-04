#include "fusion/config.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>

using namespace fusion;

namespace {

std::string writeTemp(const std::string& name, const std::string& text) {
    const auto path = std::filesystem::temp_directory_path() / name;
    std::ofstream(path) << text;
    return path.string();
}

}  // namespace

TEST(Config, DefaultFileMatchesBuiltInDefaults) {
    const Config file = loadConfig(FUSION_SOURCE_DIR "/config/default.yaml");
    const Config built = defaultConfig();
    EXPECT_EQ(file.filter.type, built.filter.type);
    EXPECT_EQ(file.filter.accel_noise, built.filter.accel_noise);
    EXPECT_EQ(file.filter.imm.models.size(), built.filter.imm.models.size());
    EXPECT_EQ(file.filter.imm.models[1].yaw_accel_noise, built.filter.imm.models[1].yaw_accel_noise);
    EXPECT_EQ(file.tracker.gate_chi2_radar, built.tracker.gate_chi2_radar);
    EXPECT_EQ(file.tracker.init.min_dt, built.tracker.init.min_dt);
    EXPECT_EQ(file.sim.lidar.scan.cluster_cell, built.sim.lidar.scan.cluster_cell);
    EXPECT_NEAR(file.sim.step, built.sim.step, 1e-12);
    EXPECT_EQ(file.sim.fov, built.sim.fov);
    EXPECT_EQ(file.sim.vehicles.size(), 4u);
    EXPECT_EQ(file.sim.ego.velocity, built.sim.ego.velocity);
    const Config spelled = loadConfig(FUSION_SOURCE_DIR "/config/scenarios/highway.yaml");
    ASSERT_EQ(spelled.sim.vehicles.size(), built.sim.vehicles.size());
    for (size_t i = 0; i < built.sim.vehicles.size(); ++i) {
        EXPECT_EQ(spelled.sim.vehicles[i].name, built.sim.vehicles[i].name);
        EXPECT_EQ(spelled.sim.vehicles[i].x, built.sim.vehicles[i].x);
        EXPECT_EQ(spelled.sim.vehicles[i].instructions.size(),
                  built.sim.vehicles[i].instructions.size());
    }
}

TEST(Config, EveryShippedFileLoads) {
    for (const auto& entry :
         std::filesystem::recursive_directory_iterator(FUSION_SOURCE_DIR "/config")) {
        if (entry.path().extension() != ".yaml") continue;
        EXPECT_NO_THROW(loadConfig(entry.path().string())) << entry.path();
    }
}

TEST(Config, UnknownKeysAreRejected) {
    EXPECT_THROW(loadConfig(writeTemp("fusion_bad1.yaml", "filter:\n  typo: 1\n")),
                 std::runtime_error);
    EXPECT_THROW(loadConfig(writeTemp("fusion_bad2.yaml", "filters:\n  type: ukf\n")),
                 std::runtime_error);
    EXPECT_THROW(loadConfig(writeTemp("fusion_bad3.yaml", "filter:\n  type: kalman\n")),
                 std::runtime_error);
}

TEST(Config, IncludesApplyInOrder) {
    writeTemp("fusion_a.yaml", "filter:\n  type: ekf\n  accel_noise: 2.0\n");
    writeTemp("fusion_b.yaml", "filter:\n  accel_noise: 3.0\n");
    const Config c = loadConfig(
        writeTemp("fusion_c.yaml", "include: [fusion_a.yaml, fusion_b.yaml]\nfilter:\n  yaw_accel_noise: 0.7\n"));
    EXPECT_EQ(c.filter.type, "ekf");
    EXPECT_EQ(c.filter.accel_noise, 3.0);
    EXPECT_EQ(c.filter.yaw_accel_noise, 0.7);
}

TEST(Config, VehiclesReplaceTheDefaultScenario) {
    const Config c = loadConfig(FUSION_SOURCE_DIR "/config/scenarios/cut_in.yaml");
    ASSERT_EQ(c.sim.vehicles.size(), 3u);
    EXPECT_EQ(c.sim.vehicles[1].name, "cutter");
    EXPECT_EQ(c.sim.vehicles[1].instructions.size(), 4u);
}
