#include "fusion/tracking/tracker.h"

#include "fusion/models.h"
#include "fusion/tracking/hungarian.h"

#include <Eigen/Cholesky>

#include <algorithm>
#include <cmath>
#include <limits>

namespace fusion {
namespace {

constexpr double kForbidden = 1e9;  // cost of a gated-out pair

}  // namespace

MultiTargetTracker::MultiTargetTracker(const Config& config) : config_(config) {}

void MultiTargetTracker::initialiseSingle(const Measurement& m, State& x, StateCov& P) const {
    x.setZero();
    P.setIdentity();
    const Eigen::Vector2d p = detectionPosition(m);
    x.head<2>() = p;
    // No velocity is observed yet: assume the object moves like the ego vehicle.
    x(2) = std::hypot(m.ego.vx, m.ego.vy);
    x(kYaw) = m.ego.yaw;
    const double sp = m.sensor == Sensor::Lidar ? config_.noise.lidar_std
                                                : config_.noise.radar_range_std;
    P(0, 0) = P(1, 1) = sp * sp;
}

void MultiTargetTracker::initialiseTwoPoint(const Measurement& first, const Measurement& second,
                                            State& x, StateCov& P) const {
    const double dt = second.time - first.time;
    const Eigen::Vector2d p1 = detectionPosition(first), p2 = detectionPosition(second);
    const Eigen::Matrix2d C1 = detectionPositionCov(first, config_.noise);
    const Eigen::Matrix2d C2 = detectionPositionCov(second, config_.noise);
    // Joint estimate of [p2, vel]; p2 appears in both, hence the cross term.
    Eigen::Vector4d s;
    s << p2, (p2 - p1) / dt;
    Eigen::Matrix4d C;
    C.topLeftCorner<2, 2>() = C2;
    C.topRightCorner<2, 2>() = C2 / dt;
    C.bottomLeftCorner<2, 2>() = C2 / dt;
    C.bottomRightCorner<2, 2>() = (C1 + C2) / (dt * dt);

    // Road-aligned prior: the velocity component across the ego heading is
    // ~ N(0, σ²), fused as a scalar Kalman update.
    const double lat = config_.tracker.init.lateral_speed_std;
    if (lat > 0.0) {
        const Eigen::Vector4d h(0.0, 0.0, -std::sin(second.ego.yaw), std::cos(second.ego.yaw));
        const double S = h.dot(C * h) + lat * lat;
        const Eigen::Vector4d K = C * h / S;
        s -= K * h.dot(s);
        C -= K * h.transpose() * C;
        C = 0.5 * (C + C.transpose());
    }
    const Eigen::Vector2d vel = s.tail<2>();

    const double speed = vel.norm();
    const double yaw = std::atan2(vel.y(), vel.x());
    // Linearised map from (vx, vy) to (v, yaw); the floor keeps it finite for
    // objects that barely move, whose heading is then simply very uncertain.
    const double v_lin = std::max(speed, 0.5);
    Eigen::Matrix4d J = Eigen::Matrix4d::Identity();
    J(2, 2) = std::cos(yaw);
    J(2, 3) = std::sin(yaw);
    J(3, 2) = -std::sin(yaw) / v_lin;
    J(3, 3) = std::cos(yaw) / v_lin;
    const Eigen::Matrix4d Cs = J * C * J.transpose();

    x << s(0), s(1), speed, yaw, 0.0;
    P.setZero();
    P.topLeftCorner<4, 4>() = Cs;
    P(kYaw, kYaw) = std::min(P(kYaw, kYaw), M_PI * M_PI / 4.0);
    P(4, 4) = config_.tracker.init.yaw_rate_std * config_.tracker.init.yaw_rate_std;
}

void MultiTargetTracker::startFilter(Track& track, const State& x, const StateCov& P,
                                     double time) {
    track.filter = makeFilter(config_.filter, config_.noise);
    track.filter->reset(x, P);
    track.time = time;
    track.last_update = time;
}

double MultiTargetTracker::detectionProbability(Sensor sensor) const {
    return sensor == Sensor::Lidar ? config_.tracker.score.lidar_detection_probability
                                   : config_.tracker.score.radar_detection_probability;
}

double MultiTargetTracker::clutterDensity(const Measurement& m) const {
    const ScoreParams& s = config_.tracker.score;
    if (m.sensor == Sensor::Lidar) return s.lidar_clutter_density;
    // A density uniform over the ground maps to range·density in (range,
    // bearing), spread over the range-rate span.
    return s.radar_clutter_density * std::max(m.z(0), 1.0) / s.radar_rate_span;
}

void MultiTargetTracker::applyUpdate(Track& track, const Measurement& m) {
    const Innovation inn = track.filter->update(m);
    updates_.push_back({track.id, m.sensor, inn.nis(), m.truth_id});
    track.last_update = m.time;
    ++track.hits;
    track.score += std::log(detectionProbability(m.sensor)) + inn.logLikelihood() -
                   std::log(clutterDensity(m));
    track.best_score = std::max(track.best_score, track.score);
}

void MultiTargetTracker::associate(const std::vector<Measurement>& detections,
                                   std::vector<bool>& used, bool confirmed_stage) {
    std::vector<int> rows;
    for (int i = 0; i < static_cast<int>(tracks_.size()); ++i)
        if (tracks_[i].initialised() && tracks_[i].confirmed == confirmed_stage) rows.push_back(i);
    std::vector<int> cols;
    for (int j = 0; j < static_cast<int>(detections.size()); ++j)
        if (!used[j]) cols.push_back(j);
    if (rows.empty() || cols.empty()) return;

    std::vector<std::vector<double>> cost(rows.size(), std::vector<double>(cols.size()));
    for (size_t r = 0; r < rows.size(); ++r) {
        const Filter& f = *tracks_[rows[r]].filter;
        for (size_t c = 0; c < cols.size(); ++c) {
            const Measurement& m = detections[cols[c]];
            const Innovation inn = f.innovation(m);
            const auto ldlt = inn.S.ldlt();
            const double d2 = inn.y.dot(ldlt.solve(inn.y));
            const double gate = m.sensor == Sensor::Lidar ? config_.tracker.gate_chi2_lidar
                                                          : config_.tracker.gate_chi2_radar;
            const double cost_rc = d2 + ldlt.vectorD().array().log().sum();
            // A non-finite cost (diverged filter) must never reach the solver.
            cost[r][c] = std::isfinite(cost_rc) && d2 <= gate ? cost_rc : kForbidden;
        }
    }
    const std::vector<int> assignment = solveAssignment(cost);
    for (size_t r = 0; r < rows.size(); ++r) {
        const int c = assignment[r];
        if (c < 0 || cost[r][c] >= kForbidden) continue;
        applyUpdate(tracks_[rows[r]], detections[cols[c]]);
        used[cols[c]] = true;
    }
}

void MultiTargetTracker::pairPending(double time, const std::vector<Measurement>& detections,
                                     std::vector<bool>& used) {
    const InitParams& ip = config_.tracker.init;
    std::vector<int> rows;
    for (int i = 0; i < static_cast<int>(tracks_.size()); ++i)
        if (!tracks_[i].initialised()) rows.push_back(i);
    std::vector<int> cols;
    for (int j = 0; j < static_cast<int>(detections.size()); ++j)
        if (!used[j]) cols.push_back(j);
    if (rows.empty() || cols.empty()) return;

    std::vector<std::vector<double>> cost(rows.size(), std::vector<double>(cols.size()));
    for (size_t r = 0; r < rows.size(); ++r) {
        const Measurement& first = tracks_[rows[r]].pending;
        const Eigen::Vector2d p1 = detectionPosition(first);
        const Eigen::Matrix2d C1 = detectionPositionCov(first, config_.noise);
        for (size_t c = 0; c < cols.size(); ++c) {
            const Measurement& m = detections[cols[c]];
            const double dist = (detectionPosition(m) - p1).norm();
            const double sigma =
                std::sqrt((C1 + detectionPositionCov(m, config_.noise)).trace());
            const bool ok = config_.tracker.association == "oracle"
                                ? m.truth_id == tracks_[rows[r]].truth_id
                                : dist <= ip.max_speed * (time - first.time) + 3.0 * sigma;
            cost[r][c] = ok ? dist : kForbidden;
        }
    }
    const std::vector<int> assignment = solveAssignment(cost);
    for (size_t r = 0; r < rows.size(); ++r) {
        const int c = assignment[r];
        if (c < 0 || cost[r][c] >= kForbidden) continue;
        Track& track = tracks_[rows[r]];
        const Measurement& m = detections[cols[c]];
        used[cols[c]] = true;
        if (m.time - track.pending.time < ip.min_dt - 1e-9) {
            // Too close in time for a usable velocity (often the other sensor
            // seeing the same object): absorb it and keep waiting.
            track.last_update = m.time;
            continue;
        }
        State x;
        StateCov P;
        initialiseTwoPoint(track.pending, m, x, P);
        startFilter(track, x, P, m.time);
        track.hits = 2;
    }
}

void MultiTargetTracker::process(double time, Sensor sensor,
                                 const std::vector<Measurement>& detections, const EgoPose& ego) {
    updates_.clear();
    const bool oracle = config_.tracker.association == "oracle";

    for (Track& t : tracks_) {
        if (!t.initialised() || time <= t.time) continue;
        t.filter->predict(time - t.time);
        t.time = time;
    }
    // A filter whose estimate has become non-finite cannot recover: drop it.
    tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(),
                                 [](const Track& t) {
                                     return t.initialised() && (!t.filter->state().allFinite() ||
                                                                !t.filter->covariance().allFinite());
                                 }),
                  tracks_.end());

    std::vector<bool> used(detections.size(), false);
    if (oracle) {
        // Labels decide; clutter (truth_id < 0) is dropped.
        for (size_t j = 0; j < detections.size(); ++j) {
            if (detections[j].truth_id < 0) {
                used[j] = true;
                continue;
            }
            for (Track& t : tracks_)
                if (t.initialised() && t.truth_id == detections[j].truth_id) {
                    applyUpdate(t, detections[j]);
                    used[j] = true;
                    break;
                }
        }
    } else {
        associate(detections, used, true);
        associate(detections, used, false);
    }

    if (config_.tracker.init.method == "two_point") pairPending(time, detections, used);

    for (size_t j = 0; j < detections.size(); ++j) {
        if (used[j]) continue;
        const Measurement& m = detections[j];
        Track t;
        t.id = next_id_++;
        t.truth_id = oracle ? m.truth_id : -1;
        t.time = t.last_update = m.time;
        if (config_.tracker.init.method == "two_point") {
            t.pending = m;
        } else {
            State x;
            StateCov P;
            initialiseSingle(m, x, P);
            startFilter(t, x, P, m.time);
        }
        tracks_.push_back(std::move(t));
    }

    // A frame from this sensor without a detection for a track that should
    // have been seen counts against it.
    {
        const auto& fov = config_.sim.fov;
        const double miss = std::log(1.0 - detectionProbability(sensor));
        for (Track& t : tracks_) {
            if (!t.initialised() || t.last_update == time) continue;
            const Eigen::VectorXd p = measure(Sensor::Lidar, t.filter->state(), ego);
            if (p(0) < fov[0] || p(0) > fov[1] || p(1) < fov[2] || p(1) > fov[3]) continue;
            t.score += miss;
        }
    }

    const bool by_score = config_.tracker.management == "score";
    for (Track& t : tracks_) {
        if (!t.initialised() || t.confirmed) continue;
        if (oracle || (by_score ? t.score >= config_.tracker.score.confirm
                                : t.hits >= config_.tracker.confirm_hits))
            t.confirmed = true;
    }

    tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(),
                                 [&](const Track& t) {
                                     if (oracle) return false;
                                     const double idle = time - t.last_update;
                                     if (idle > (t.confirmed ? config_.tracker.max_coast
                                                             : config_.tracker.tentative_timeout))
                                         return true;
                                     if (!by_score || !t.initialised()) return false;
                                     return t.confirmed
                                                ? t.score < t.best_score - config_.tracker.score.delete_drop
                                                : t.score < config_.tracker.score.delete_tentative;
                                 }),
                  tracks_.end());
}

}  // namespace fusion
