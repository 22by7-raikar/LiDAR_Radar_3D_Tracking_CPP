/// @file
/// @brief Accuracy, consistency and multi-target tracking metrics.
#pragma once

#include "fusion/config.h"
#include "fusion/sim/vehicle.h"
#include "fusion/tracking/tracker.h"

#include <Eigen/Core>

#include <map>
#include <vector>

namespace fusion {

/// @brief Summary of one simulated run.
struct RunMetrics {
    /// RMSE of [px, py, vx, vy] over all scored estimates from EvalParams::start on.
    Eigen::Vector4d rmse = Eigen::Vector4d::Zero();
    /// 95th percentile of the velocity error magnitude, from EvalParams::start on [m/s].
    double velocity_p95 = 0.0;
    /// Mean time from a vehicle's first appearance in the field of view to its
    /// first confirmed, matched track [s]; vehicles never tracked count as the
    /// rest of the run.
    double latency = 0.0;
    double nees_position = 0.0;     ///< Mean position NEES; 2 for a consistent filter.
    double nis_lidar = 0.0;         ///< Mean lidar NIS; 2 for a consistent filter.
    double nis_radar = 0.0;         ///< Mean radar NIS; 3 for a consistent filter.
    double nis_lidar_in95 = 0.0;    ///< Share of lidar NIS below the χ²₂ 95 % quantile (expect 0.95).
    double nis_radar_in95 = 0.0;    ///< Share of radar NIS below the χ²₃ 95 % quantile (expect 0.95).
    double ospa = 0.0;              ///< Mean OSPA distance over the scored times [m].
    int id_switches = 0;            ///< Times a vehicle's matched track identity changed.
    double false_tracks = 0.0;      ///< Mean confirmed tracks not near any vehicle, per scored time.
    double missed = 0.0;            ///< Share of (vehicle, time) pairs without a matched track.
    double us_per_detection = 0.0;  ///< Tracker time per processed detection [µs].
    int detections = 0;             ///< Detections processed.
};

/// @brief Error of one estimate against the truth, for logs and figures.
struct ScoredEstimate {
    double time = 0.0;      ///< Scoring time [s].
    int vehicle = -1;       ///< Vehicle.
    int track = -1;         ///< Matched track.
    Eigen::Vector4d error;  ///< Estimate − truth in [px, py, vx, vy].
    double nees = 0.0;      ///< Position NEES.
};

/// @brief Scores tracker output against the simulated truth over a run.
///
/// Confirmed tracks are predicted to the scoring time and matched to vehicles:
/// by label for the oracle tracker, otherwise by an optimal assignment on
/// position distance with EvalParams::match_distance as the limit.
class Evaluator {
public:
    /// @param config Evaluation and tracker settings.
    explicit Evaluator(const Config& config);

    /// @brief Scores the tracks at @p time.
    /// @param time Scoring time [s].
    /// @param vehicles Traffic at @p time.
    /// @param visible Which vehicles can be seen (missing tracks for the others are not penalised).
    /// @param tracker Tracker after processing all detections up to @p time.
    /// @return The matched estimates.
    std::vector<ScoredEstimate> score(double time, const std::vector<Vehicle>& vehicles,
                                      const std::vector<bool>& visible,
                                      const MultiTargetTracker& tracker);

    /// @brief Adds the tracker's latest filter updates to the NIS statistics.
    void addUpdates(const std::vector<UpdateRecord>& updates);

    /// @brief Summary so far.
    RunMetrics summary() const;

private:
    Config config_;
    Eigen::Vector4d sq_ = Eigen::Vector4d::Zero();
    long n_ = 0;
    std::vector<double> speed_errors_;
    std::map<int, double> first_seen_, first_tracked_;
    double last_time_ = 0.0;
    double nees_ = 0.0;
    double nis_[2] = {0, 0};
    long nis_n_[2] = {0, 0}, nis_in_[2] = {0, 0};
    double ospa_ = 0.0, false_ = 0.0;
    long steps_ = 0, truth_steps_ = 0, missed_ = 0;
    int switches_ = 0;
    std::map<int, int> last_track_;
};

/// @brief OSPA distance (order 2) between two point sets.
/// @param truth Reference positions.
/// @param estimates Estimated positions.
/// @param cutoff Cut-off c; also the penalty per missing or extra point [m].
double ospa(const std::vector<Eigen::Vector2d>& truth,
            const std::vector<Eigen::Vector2d>& estimates, double cutoff);

}  // namespace fusion
