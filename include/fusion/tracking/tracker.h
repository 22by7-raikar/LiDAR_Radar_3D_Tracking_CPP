/// @file
/// @brief Multi-target tracker: association, initialisation and track life cycle.
#pragma once

#include "fusion/config.h"
#include "fusion/filters/filter.h"
#include "fusion/types.h"

#include <memory>
#include <vector>

namespace fusion {

/// @brief One tracked object.
struct Track {
    int id = -1;                     ///< Identity, unique within a tracker.
    std::unique_ptr<Filter> filter;  ///< Estimator; empty until initialised.
    Measurement pending;             ///< First detection, while waiting for a second (two-point initialisation).
    double time = 0.0;               ///< Time of the filter estimate [s].
    double last_update = 0.0;        ///< Time of the last associated detection [s].
    int hits = 1;                    ///< Associated detections so far.
    bool confirmed = false;          ///< Reported to the user (TrackerParams::confirm_hits reached).
    int truth_id = -1;               ///< Oracle association only: the simulated object.
    double score = 0.0;              ///< Log-likelihood ratio, track versus clutter.
    double best_score = 0.0;         ///< Highest #score so far.

    /// @brief True once the filter exists.
    bool initialised() const { return static_cast<bool>(filter); }
};

/// @brief One filter update, for consistency statistics.
struct UpdateRecord {
    int track_id = -1;       ///< Updated track.
    Sensor sensor = Sensor::Lidar;  ///< Sensor of the detection.
    double nis = 0.0;        ///< Normalised innovation squared before the update.
    int truth_id = -1;       ///< Object behind the detection (-1 for clutter).
};

/// @brief Tracks an unknown number of objects from unlabelled lidar and radar detections.
///
/// Tracks are kept in the world frame; detections arrive in the ego frame
/// with the ego pose attached (Measurement::ego), and the measurement models
/// account for the ego vehicle's position and motion.
///
/// Each call to process() handles all detections of one sensor at one time:
/// 1. Every initialised track is predicted to that time.
/// 2. **Association.** `gnn` scores each track/detection pair by the negative
///    log-likelihood d² + ln|S|, where d² is the normalised innovation, and
///    forbids pairs with d² above the sensor's χ² gate. Confirmed tracks are
///    assigned first, then tentative tracks compete for the rest; each stage
///    is solved optimally with solveAssignment(). `oracle` uses the
///    simulator's labels instead.
/// 3. **Initialisation.** Detections left over are paired with waiting first
///    detections (two-point) or start a filter directly (single).
/// 4. **Life cycle.** With `score` management each track carries a
///    log-likelihood ratio (ScoreParams): it is confirmed when the ratio
///    passes ScoreParams::confirm and deleted when it falls below
///    ScoreParams::delete_tentative (unconfirmed) or ScoreParams::delete_drop
///    under its best value (confirmed). With `hits` management a track is
///    confirmed after TrackerParams::confirm_hits detections. Either way,
///    tracks are also dropped after TrackerParams::tentative_timeout
///    (unconfirmed) or TrackerParams::max_coast (confirmed) without a detection.
class MultiTargetTracker {
public:
    /// @param config Filter, noise and tracker parameters are read from it.
    explicit MultiTargetTracker(const Config& config);

    /// @brief Processes one sensor frame.
    /// @param time Time stamp [s]; must not decrease between calls.
    /// @param sensor Sensor that produced the frame.
    /// @param detections Its detections (may be empty: the frame still counts
    ///        as a missed detection for every track in view).
    /// @param ego Ego pose at @p time, used to decide which tracks were in view.
    void process(double time, Sensor sensor, const std::vector<Measurement>& detections,
                 const EgoPose& ego = {});

    /// @brief Live tracks, including unconfirmed and uninitialised ones.
    const std::vector<Track>& tracks() const { return tracks_; }

    /// @brief Filter updates made by the last process() call.
    const std::vector<UpdateRecord>& lastUpdates() const { return updates_; }

    /// @brief Starts a filter from one detection, assuming it moves like the ego vehicle.
    void initialiseSingle(const Measurement& m, State& x, StateCov& P) const;

    /// @brief Starts a filter from two detections of one object.
    ///
    /// Position from the second detection, velocity from the difference, and a
    /// covariance that follows from the two detection covariances (including
    /// the position/velocity correlation); the velocity covariance is mapped to
    /// speed and heading by linearisation.
    void initialiseTwoPoint(const Measurement& first, const Measurement& second, State& x,
                            StateCov& P) const;

private:
    void associate(const std::vector<Measurement>& detections, std::vector<bool>& used,
                   bool confirmed_stage);
    void pairPending(double time, const std::vector<Measurement>& detections,
                     std::vector<bool>& used);
    void startFilter(Track& track, const State& x, const StateCov& P, double time);
    void applyUpdate(Track& track, const Measurement& m);
    double clutterDensity(const Measurement& m) const;
    double detectionProbability(Sensor sensor) const;

    Config config_;
    std::vector<Track> tracks_;
    std::vector<UpdateRecord> updates_;
    int next_id_ = 0;
};

}  // namespace fusion
