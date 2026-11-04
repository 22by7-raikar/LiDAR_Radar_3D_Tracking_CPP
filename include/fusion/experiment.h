/// @file
/// @brief Runs a simulation through the tracker and scores it.
#pragma once

#include "fusion/config.h"
#include "fusion/eval/metrics.h"
#include "fusion/sim/simulator.h"

#include <cstdint>
#include <vector>

namespace fusion {

/// @brief One estimate as logged for plots.
struct LoggedEstimate {
    double time = 0.0;              ///< Time [s].
    int track = -1;                 ///< Track identity.
    State x;                        ///< State estimate.
    StateCov P;                     ///< Covariance.
    Eigen::VectorXd mode_probabilities;  ///< IMM mode probabilities (empty for other filters).
};

/// @brief True state of one vehicle at one time.
struct LoggedTruth {
    double time = 0.0;  ///< Time [s].
    int vehicle = -1;   ///< Vehicle.
    VehicleState s;     ///< State.
};

/// @brief Everything recorded during a run, for the CSV outputs and the figures.
struct RunLog {
    std::vector<LoggedTruth> truth;           ///< Truth at each scoring time.
    std::vector<double> ego_times;            ///< Scoring times, one per entry of #ego.
    std::vector<EgoPose> ego;                 ///< Ego pose at each scoring time.
    std::vector<Measurement> measurements;    ///< All detections, in processing order.
    std::vector<LoggedEstimate> estimates;    ///< Confirmed tracks at each scoring time.
    std::vector<ScoredEstimate> scored;       ///< Matched estimates and their errors.
    std::vector<UpdateRecord> updates;        ///< Every filter update with its NIS.
    std::vector<double> update_times;         ///< Time of each entry of #updates.
    /// Lidar frames kept with their returns (scan model, every frame).
    std::vector<SensorFrame> scans;
};

/// @brief Simulates one run with @p seed, tracks it and scores it.
/// @param config Configuration.
/// @param seed Random seed.
/// @param log If not null, receives the full record of the run.
RunMetrics runSimulation(const Config& config, uint64_t seed, RunLog* log = nullptr);

}  // namespace fusion
