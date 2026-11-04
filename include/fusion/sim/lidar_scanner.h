/// @file
/// @brief Ray-cast LiDAR scans of the traffic and a detector that turns them into position measurements.
#pragma once

#include "fusion/config.h"
#include "fusion/sim/vehicle.h"
#include "fusion/types.h"

#include <Eigen/Core>

#include <random>
#include <vector>

namespace fusion {

/// @brief One LiDAR return.
struct ScanPoint {
    Eigen::Vector3d p;  ///< Position in the sensor (ego) frame [m].
    int vehicle = -1;   ///< Vehicle that was hit (ground truth, for evaluation only).
};

/// @brief One object found in a scan.
struct ScanDetection {
    Eigen::Vector2d centroid;  ///< Mean of the returns [m].
    Eigen::Vector2d center;    ///< Centre of the completed box [m].
    double yaw = 0.0;          ///< Box heading, modulo pi/2 [rad].
    Eigen::Vector2d extent;    ///< Observed extent along the two box axes [m].
    int points = 0;            ///< Number of returns.
    bool cabin_only = false;   ///< Only the cabin was visible (lower body hidden by another vehicle).
    int vehicle = -1;          ///< Vehicle with most returns in the cluster (evaluation only).
};

/// @brief Simulated spinning LiDAR over a flat road.
///
/// Rays are intersected exactly with each vehicle's two boxes (body above
/// the ground clearance, and a narrower cabin) and with the road plane. Road returns
/// are not emitted: the detector sees perfect ground removal, so its errors
/// come only from partial visibility, occlusion and range noise.
class LidarScanner {
public:
    /// @param params Sensor geometry and detector settings.
    explicit LidarScanner(const ScanParams& params);

    /// @brief Casts all rays that can reach a vehicle and returns the hits.
    /// @param vehicles Traffic, in the world frame.
    /// @param ego Pose of the vehicle carrying the sensor.
    /// @param rng Random source for the range noise.
    std::vector<ScanPoint> scan(const std::vector<Vehicle>& vehicles, const EgoPose& ego,
                                std::mt19937_64& rng) const;

    /// @brief Clusters returns on a 2-D grid and fits a box to each cluster.
    ///
    /// The box is an L-shape fit (variance criterion, 1° steps) to the nearest
    /// return in each azimuth bin. A side seen shorter than the nominal vehicle
    /// size is extended away from the sensor, because the hidden part of a car
    /// is always on the far side. When only the cabin is visible (no returns
    /// below the belt line, typically over a nearer vehicle), the cabin's box
    /// is completed instead and shifted to the body centre.
    std::vector<ScanDetection> detect(const std::vector<ScanPoint>& points) const;

private:
    ScanParams params_;
};

}  // namespace fusion
