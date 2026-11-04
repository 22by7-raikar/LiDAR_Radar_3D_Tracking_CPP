#include "fusion/experiment.h"

#include "fusion/filters/imm.h"
#include "fusion/tracking/tracker.h"

#include <chrono>
#include <cmath>

namespace fusion {

RunMetrics runSimulation(const Config& config, uint64_t seed, RunLog* log) {
    Simulator sim(config, seed);
    MultiTargetTracker tracker(config);
    Evaluator evaluator(config);
    const double eval_period = 1.0 / config.eval.rate;
    double next_eval = 0.0;
    double tracker_seconds = 0.0;
    int detections = 0;

    while (!sim.done()) {
        const double t = sim.time();
        for (SensorFrame& frame : sim.sense()) {
            const auto start = std::chrono::steady_clock::now();
            tracker.process(frame.time, frame.sensor, frame.detections, frame.ego);
            tracker_seconds +=
                std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            detections += static_cast<int>(frame.detections.size());
            evaluator.addUpdates(tracker.lastUpdates());
            if (log) {
                log->measurements.insert(log->measurements.end(), frame.detections.begin(),
                                         frame.detections.end());
                for (const UpdateRecord& u : tracker.lastUpdates()) {
                    log->updates.push_back(u);
                    log->update_times.push_back(frame.time);
                }
                if (!frame.points.empty()) log->scans.push_back(std::move(frame));
            }
        }
        if (t + 1e-9 >= next_eval) {
            next_eval += eval_period;
            std::vector<bool> visible;
            for (const Vehicle& v : sim.vehicles()) visible.push_back(sim.visible(v));
            const auto scored = evaluator.score(t, sim.vehicles(), visible, tracker);
            if (log) {
                log->scored.insert(log->scored.end(), scored.begin(), scored.end());
                for (const Vehicle& v : sim.vehicles()) log->truth.push_back({t, v.id(), v.state()});
                log->ego_times.push_back(t);
                log->ego.push_back(sim.egoPose());
                for (const Track& tr : tracker.tracks()) {
                    if (!tr.initialised() || !tr.confirmed) continue;
                    auto f = tr.filter->clone();
                    f->predict(t - tr.time);
                    LoggedEstimate e{t, tr.id, f->state(), f->covariance(), {}};
                    if (const auto* imm = dynamic_cast<const Imm*>(f.get()))
                        e.mode_probabilities = imm->modeProbabilities();
                    log->estimates.push_back(e);
                }
            }
        }
        sim.advance();
    }

    RunMetrics m = evaluator.summary();
    m.detections = detections;
    m.us_per_detection = detections ? 1e6 * tracker_seconds / detections : 0.0;
    return m;
}

}  // namespace fusion
