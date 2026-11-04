/// @file
/// @brief Highway simulation: traffic, sensor timing, noise, misses and clutter.
#pragma once

#include "fusion/config.h"
#include "fusion/sim/lidar_scanner.h"
#include "fusion/sim/vehicle.h"
#include "fusion/types.h"

#include <cstdint>
#include <random>
#include <vector>

namespace fusion {

/// @brief Everything one sensor reported at one time.
struct SensorFrame {
    double time = 0.0;                     ///< Time stamp [s].
    Sensor sensor = Sensor::Lidar;         ///< Which sensor.
    EgoPose ego;                           ///< Ego pose at #time (perfect odometry).
    std::vector<Measurement> detections;   ///< Detections, in random order.
    std::vector<ScanPoint> points;         ///< Lidar returns (scan model only).
    std::vector<ScanDetection> objects;    ///< Lidar scan detections (scan model only).
};

/// @brief Steps the traffic and produces sensor frames on schedule.
///
/// Everything moves in the world frame, including the ego vehicle that
/// carries both sensors; detections are produced in the ego frame. Each
/// sensor has its own rate and offset, so lidar and radar can be
/// asynchronous. Every random draw comes from generators seeded once per run
/// (one per sensor), so results are reproducible and the noise of different
/// vehicles and sensors is independent.
class Simulator {
public:
    /// @param config Scenario, sensors and noise.
    /// @param seed Seed for all random draws.
    Simulator(const Config& config, uint64_t seed);

    /// @brief Current time [s].
    double time() const { return tick_ * config_.sim.step; }

    /// @brief True when the configured duration has been simulated.
    bool done() const;

    /// @brief Frames due at the current time (possibly none, possibly both sensors).
    std::vector<SensorFrame> sense();

    /// @brief Advances the traffic by one step.
    void advance();

    /// @brief The traffic.
    const std::vector<Vehicle>& vehicles() const { return vehicles_; }

    /// @brief The ego vehicle.
    const Vehicle& ego() const { return ego_; }

    /// @brief Current ego pose and velocity.
    EgoPose egoPose() const;

    /// @brief True if a vehicle is inside the field of view (SimParams::fov, ego frame).
    bool visible(const Vehicle& v) const;

private:
    SensorFrame lidarFrame();
    SensorFrame radarFrame();

    Config config_;
    Vehicle ego_;
    std::vector<Vehicle> vehicles_;
    LidarScanner scanner_;
    long tick_ = 0;
    double next_lidar_, next_radar_;
    std::mt19937_64 lidar_rng_, radar_rng_;
};

}  // namespace fusion
