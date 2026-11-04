#include "fusion/sim/lidar_scanner.h"
#include "fusion/sim/simulator.h"

#include <gtest/gtest.h>

#include <cmath>

using namespace fusion;

namespace {

VehicleParams car(double x, double y, double yaw = 0.0) {
    VehicleParams p;
    p.name = "car";
    p.x = x;
    p.y = y;
    p.angle = yaw;
    return p;
}

}  // namespace

TEST(Vehicle, FollowsItsScript) {
    VehicleParams p = car(0.0, 0.0);
    p.velocity = 2.0;
    p.instructions = {{1.0, 1.0, 0.0}};
    Vehicle v(p, 0);
    const double dt = 0.01;
    for (int k = 0; k < 200; ++k) v.step(k * dt, dt);
    // 1 s at 2 m/s, then 1 s accelerating at 1 m/s² (midpoint rule: exact).
    EXPECT_NEAR(v.state().x, 2.0 + 2.0 + 0.5, 1e-9);
    EXPECT_NEAR(v.state().v, 3.0, 1e-9);
}

TEST(Simulator, SensorsFireAtTheirOwnRates) {
    Config c = defaultConfig();
    c.sim.step = 0.005;
    c.sim.duration = 2.0;
    c.sim.lidar.rate = 10.0;
    c.sim.radar.rate = 20.0;
    c.sim.radar.offset = 0.025;
    Simulator sim(c, 1);
    int lidar = 0, radar = 0;
    while (!sim.done()) {
        for (const auto& f : sim.sense()) (f.sensor == Sensor::Lidar ? lidar : radar)++;
        sim.advance();
    }
    EXPECT_EQ(lidar, 21);  // t = 0, 0.1, ..., 2.0
    EXPECT_EQ(radar, 40);  // t = 0.025, 0.075, ..., 1.975
}

TEST(Simulator, NoiseIsIndependentAcrossVehicles) {
    // Two vehicles at the same place must get different noise draws.
    Config c = defaultConfig();
    c.sim.vehicles = {car(10.0, 0.0), car(10.0, 0.0)};
    Simulator sim(c, 7);
    const auto frames = sim.sense();
    ASSERT_EQ(frames[0].detections.size(), 2u);
    EXPECT_GT((frames[0].detections[0].z - frames[0].detections[1].z).norm(), 1e-6);
}

TEST(Simulator, DetectionsAreInTheEgoFrame) {
    // A car 20 m ahead of a moving, rotated ego vehicle, at the same speed.
    Config c = defaultConfig();
    c.noise.lidar_std = 1e-9;
    c.noise.radar_range_std = c.noise.radar_bearing_std = c.noise.radar_rate_std = 1e-9;
    c.sim.ego = car(100.0, 50.0, 0.5);
    c.sim.ego.velocity = 20.0;
    VehicleParams other = car(100.0 + 20.0 * std::cos(0.5), 50.0 + 20.0 * std::sin(0.5), 0.5);
    other.velocity = 23.0;
    c.sim.vehicles = {other};
    Simulator sim(c, 1);
    const auto frames = sim.sense();
    ASSERT_EQ(frames.size(), 2u);
    const Measurement& lidar = frames[0].detections.at(0);
    const Measurement& radar = frames[1].detections.at(0);
    EXPECT_NEAR(lidar.z(0), 20.0, 1e-6);
    EXPECT_NEAR(lidar.z(1), 0.0, 1e-6);
    EXPECT_NEAR(radar.z(2), 3.0, 1e-6);  // closing speed is relative
    EXPECT_NEAR(lidar.ego.x, 100.0, 1e-12);
    EXPECT_NEAR(lidar.ego.yaw, 0.5, 1e-12);
}

TEST(Simulator, ClutterRateMatchesPoisson) {
    Config c = defaultConfig();
    c.sim.vehicles.clear();
    c.sim.radar.clutter_rate = 4.0;
    c.sim.lidar.enabled = false;
    Simulator sim(c, 3);
    double total = 0;
    int frames = 0;
    while (!sim.done()) {
        for (const auto& f : sim.sense()) {
            total += static_cast<double>(f.detections.size());
            ++frames;
        }
        sim.advance();
    }
    EXPECT_NEAR(total / frames, 4.0, 0.4);
}

TEST(LidarScanner, BoxDetectorFindsTheCentre) {
    ScanParams sp;
    LidarScanner scanner(sp);
    std::mt19937_64 rng(1);
    // Seen from behind, from the side, at an angle, and in another lane.
    for (const auto& [x, y, yaw] : std::vector<std::tuple<double, double, double>>{
             {20.0, 0.0, 0.0}, {0.0, 6.0, 0.0}, {12.0, 4.0, 0.4}, {-15.0, -4.0, 0.0}}) {
        const std::vector<Vehicle> v{Vehicle(car(x, y, yaw), 0)};
        const auto dets = scanner.detect(scanner.scan(v, EgoPose{}, rng));
        ASSERT_EQ(dets.size(), 1u) << x << "," << y;
        EXPECT_LT((dets[0].center - Eigen::Vector2d(x, y)).norm(), 0.25) << x << "," << y;
        // The centroid sits on the visible faces, well off the centre.
        EXPECT_GT((dets[0].centroid - Eigen::Vector2d(x, y)).norm(), 0.4) << x << "," << y;
    }
}

TEST(LidarScanner, NearVehicleOccludesFarOne) {
    ScanParams sp;
    LidarScanner scanner(sp);
    std::mt19937_64 rng(1);
    const std::vector<Vehicle> v{Vehicle(car(8.0, 0.0), 0), Vehicle(car(20.0, 0.0), 1)};
    int far = 0;
    for (const auto& p : scanner.scan(v, EgoPose{}, rng)) far += p.vehicle == 1;
    // Only the top of the far car's cabin shows over the near one.
    std::mt19937_64 rng2(1);
    int alone = 0;
    for (const auto& p : scanner.scan({Vehicle(car(20.0, 0.0), 1)}, EgoPose{}, rng2))
        alone += p.vehicle == 1;
    EXPECT_LT(far, alone / 2);
}

TEST(LidarScanner, AdjacentLanesStaySeparate) {
    ScanParams sp;
    LidarScanner scanner(sp);
    std::mt19937_64 rng(1);
    const std::vector<Vehicle> v{Vehicle(car(15.0, 0.0), 0), Vehicle(car(15.0, 3.7), 1)};
    EXPECT_EQ(scanner.detect(scanner.scan(v, EgoPose{}, rng)).size(), 2u);
}
