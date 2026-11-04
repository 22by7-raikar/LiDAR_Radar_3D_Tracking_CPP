#include "fusion/eval/metrics.h"

#include "fusion/tracking/hungarian.h"

#include <Eigen/Cholesky>

#include <algorithm>
#include <cmath>

namespace fusion {
namespace {

constexpr double kChi2_95[2] = {5.991, 7.815};  // 2 and 3 degrees of freedom

struct Estimate {
    int track;
    State x;
    StateCov P;
};

}  // namespace

double ospa(const std::vector<Eigen::Vector2d>& truth,
            const std::vector<Eigen::Vector2d>& estimates, double cutoff) {
    const auto& small = truth.size() <= estimates.size() ? truth : estimates;
    const auto& large = truth.size() <= estimates.size() ? estimates : truth;
    if (large.empty()) return 0.0;
    double total = cutoff * cutoff * static_cast<double>(large.size() - small.size());
    if (!small.empty()) {
        std::vector<std::vector<double>> cost(small.size(), std::vector<double>(large.size()));
        for (size_t i = 0; i < small.size(); ++i)
            for (size_t j = 0; j < large.size(); ++j) {
                const double d = std::min((small[i] - large[j]).norm(), cutoff);
                cost[i][j] = d * d;
            }
        const auto a = solveAssignment(cost);
        for (size_t i = 0; i < small.size(); ++i) total += cost[i][a[i]];
    }
    return std::sqrt(total / static_cast<double>(large.size()));
}

Evaluator::Evaluator(const Config& config) : config_(config) {}

std::vector<ScoredEstimate> Evaluator::score(double time, const std::vector<Vehicle>& vehicles,
                                             const std::vector<bool>& visible,
                                             const MultiTargetTracker& tracker) {
    std::vector<Estimate> estimates;
    for (const Track& t : tracker.tracks()) {
        if (!t.initialised() || !t.confirmed) continue;
        auto f = t.filter->clone();
        f->predict(time - t.time);
        estimates.push_back({t.id, f->state(), f->covariance()});
    }

    // Match vehicles to estimates.
    std::vector<int> match(vehicles.size(), -1);
    if (config_.tracker.association == "oracle") {
        for (size_t v = 0; v < vehicles.size(); ++v)
            for (size_t e = 0; e < estimates.size(); ++e) {
                const Track* t = nullptr;
                for (const Track& tr : tracker.tracks())
                    if (tr.id == estimates[e].track) t = &tr;
                if (t && t->truth_id == vehicles[v].id()) match[v] = static_cast<int>(e);
            }
    } else if (!estimates.empty()) {
        std::vector<std::vector<double>> cost(vehicles.size(),
                                              std::vector<double>(estimates.size()));
        for (size_t v = 0; v < vehicles.size(); ++v) {
            const VehicleState s = vehicles[v].state();
            for (size_t e = 0; e < estimates.size(); ++e) {
                const double d = std::hypot(estimates[e].x(0) - s.x, estimates[e].x(1) - s.y);
                cost[v][e] = d <= config_.eval.match_distance ? d : 1e9;
            }
        }
        const auto a = solveAssignment(cost);
        for (size_t v = 0; v < vehicles.size(); ++v)
            if (a[v] >= 0 && cost[v][a[v]] < 1e9) match[v] = a[v];
    }

    std::vector<ScoredEstimate> out;
    std::vector<Eigen::Vector2d> truth_pos, est_pos;
    std::vector<bool> est_used(estimates.size(), false), est_hidden(estimates.size(), false);
    const bool after_start = time + 1e-9 >= config_.eval.start;
    for (size_t v = 0; v < vehicles.size(); ++v) {
        const VehicleState s = vehicles[v].state();
        if (visible[v]) {
            first_seen_.emplace(vehicles[v].id(), time);
            truth_pos.emplace_back(s.x, s.y);
            if (after_start) ++truth_steps_;
        }
        if (match[v] < 0) {
            if (visible[v] && after_start) ++missed_;
            continue;
        }
        const Estimate& e = estimates[match[v]];
        est_used[match[v]] = true;
        if (!visible[v]) {
            // Still tracked after leaving the field of view: neither scored
            // nor counted as a false track.
            est_hidden[match[v]] = true;
            continue;
        }
        const Eigen::Vector2d vel = velocityXY(e.x);
        ScoredEstimate se;
        se.time = time;
        se.vehicle = vehicles[v].id();
        se.track = e.track;
        se.error << e.x(0) - s.x, e.x(1) - s.y, vel.x() - s.velocity().x(),
            vel.y() - s.velocity().y();
        const Eigen::Vector2d dp = se.error.head<2>();
        se.nees = dp.dot(e.P.topLeftCorner<2, 2>().ldlt().solve(dp));
        out.push_back(se);

        first_tracked_.emplace(se.vehicle, time);
        const Eigen::Vector4d sq = se.error.cwiseAbs2();
        if (after_start) {
            sq_ += sq;
            ++n_;
            nees_ += se.nees;
            speed_errors_.push_back(se.error.tail<2>().norm());
        }
        auto it = last_track_.find(se.vehicle);
        if (it != last_track_.end() && it->second != se.track) ++switches_;
        last_track_[se.vehicle] = se.track;
    }
    for (size_t e = 0; e < estimates.size(); ++e)
        if (!est_hidden[e]) est_pos.push_back(estimates[e].x.head<2>());

    last_time_ = time;
    if (after_start) {
        ospa_ += ospa(truth_pos, est_pos, config_.eval.ospa_cutoff);
        false_ += static_cast<double>(std::count(est_used.begin(), est_used.end(), false));
        ++steps_;
    }
    return out;
}

void Evaluator::addUpdates(const std::vector<UpdateRecord>& updates) {
    for (const UpdateRecord& u : updates) {
        if (u.truth_id < 0) continue;  // clutter accepted by a track: not a model check
        const int k = u.sensor == Sensor::Lidar ? 0 : 1;
        nis_[k] += u.nis;
        ++nis_n_[k];
        if (u.nis <= kChi2_95[k]) ++nis_in_[k];
    }
}

RunMetrics Evaluator::summary() const {
    RunMetrics m;
    if (n_ > 0) {
        m.rmse = (sq_ / static_cast<double>(n_)).cwiseSqrt();
        m.nees_position = nees_ / static_cast<double>(n_);
    }
    if (!speed_errors_.empty()) {
        std::vector<double> e = speed_errors_;
        const size_t k = static_cast<size_t>(0.95 * static_cast<double>(e.size() - 1));
        std::nth_element(e.begin(), e.begin() + static_cast<long>(k), e.end());
        m.velocity_p95 = e[k];
    }
    if (!first_seen_.empty()) {
        double total = 0.0;
        for (const auto& [id, seen] : first_seen_) {
            const auto it = first_tracked_.find(id);
            total += (it != first_tracked_.end() ? it->second : last_time_) - seen;
        }
        m.latency = total / static_cast<double>(first_seen_.size());
    }
    if (nis_n_[0]) {
        m.nis_lidar = nis_[0] / static_cast<double>(nis_n_[0]);
        m.nis_lidar_in95 = static_cast<double>(nis_in_[0]) / static_cast<double>(nis_n_[0]);
    }
    if (nis_n_[1]) {
        m.nis_radar = nis_[1] / static_cast<double>(nis_n_[1]);
        m.nis_radar_in95 = static_cast<double>(nis_in_[1]) / static_cast<double>(nis_n_[1]);
    }
    if (steps_) {
        m.ospa = ospa_ / static_cast<double>(steps_);
        m.false_tracks = false_ / static_cast<double>(steps_);
    }
    if (truth_steps_) m.missed = static_cast<double>(missed_) / static_cast<double>(truth_steps_);
    m.id_switches = switches_;
    return m;
}

}  // namespace fusion
