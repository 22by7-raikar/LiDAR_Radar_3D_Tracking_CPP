#include "fusion/sim/simulator.h"

#include <algorithm>
#include <cmath>

namespace fusion {
namespace {

// Tolerance for "the sensor is due on this tick".
constexpr double kTimeEps = 1e-9;

// Position and velocity of a vehicle in the ego frame.
void relative(const VehicleState& s, const EgoPose& ego, Eigen::Vector2d& p, Eigen::Vector2d& v) {
    const double c = std::cos(ego.yaw), sn = std::sin(ego.yaw);
    Eigen::Matrix2d Rt;
    Rt << c, sn, -sn, c;
    p = Rt * Eigen::Vector2d(s.x - ego.x, s.y - ego.y);
    v = Rt * (s.velocity() - Eigen::Vector2d(ego.vx, ego.vy));
}

}  // namespace

Simulator::Simulator(const Config& config, uint64_t seed)
    : config_(config),
      ego_(config.sim.ego, -2),
      scanner_(config.sim.lidar.scan),
      next_lidar_(config.sim.lidar.offset),
      next_radar_(config.sim.radar.offset),
      lidar_rng_(seed * 2 + 1),
      radar_rng_(seed * 2 + 2) {
    for (size_t i = 0; i < config.sim.vehicles.size(); ++i)
        vehicles_.emplace_back(config.sim.vehicles[i], static_cast<int>(i));
}

bool Simulator::done() const { return time() > config_.sim.duration + kTimeEps; }

EgoPose Simulator::egoPose() const {
    const VehicleState s = ego_.state();
    const Eigen::Vector2d v = s.velocity();
    return {s.x, s.y, s.yaw, v.x(), v.y()};
}

bool Simulator::visible(const Vehicle& v) const {
    const auto& f = config_.sim.fov;
    Eigen::Vector2d p, vel;
    relative(v.state(), egoPose(), p, vel);
    return p.x() >= f[0] && p.x() <= f[1] && p.y() >= f[2] && p.y() <= f[3];
}

std::vector<SensorFrame> Simulator::sense() {
    std::vector<SensorFrame> frames;
    const double t = time();
    // Lidar first when both sensors share a time stamp.
    if (config_.sim.lidar.enabled && t + kTimeEps >= next_lidar_) {
        frames.push_back(lidarFrame());
        while (next_lidar_ <= t + kTimeEps) next_lidar_ += 1.0 / config_.sim.lidar.rate;
    }
    if (config_.sim.radar.enabled && t + kTimeEps >= next_radar_) {
        frames.push_back(radarFrame());
        while (next_radar_ <= t + kTimeEps) next_radar_ += 1.0 / config_.sim.radar.rate;
    }
    return frames;
}

void Simulator::advance() {
    const double t = time();
    ego_.step(t, config_.sim.step);
    for (Vehicle& v : vehicles_) v.step(t, config_.sim.step);
    ++tick_;
}

SensorFrame Simulator::lidarFrame() {
    SensorFrame frame;
    frame.time = time();
    frame.sensor = Sensor::Lidar;
    frame.ego = egoPose();
    const auto& p = config_.sim.lidar;
    const auto& fov = config_.sim.fov;
    std::uniform_real_distribution<double> unit(0.0, 1.0);

    if (p.model == "scan") {
        frame.points = scanner_.scan(vehicles_, frame.ego, lidar_rng_);
        frame.objects = scanner_.detect(frame.points);
        for (const ScanDetection& d : frame.objects) {
            if (unit(lidar_rng_) > p.detection_probability) continue;
            Measurement m;
            m.sensor = Sensor::Lidar;
            m.time = frame.time;
            m.ego = frame.ego;
            m.z = p.scan.detector == "box" ? d.center : d.centroid;
            m.truth_id = d.vehicle;
            frame.detections.push_back(m);
        }
    } else {
        std::normal_distribution<double> noise(0.0, config_.noise.lidar_std);
        for (const Vehicle& v : vehicles_) {
            if (!visible(v) || unit(lidar_rng_) > p.detection_probability) continue;
            Eigen::Vector2d rp, rv;
            relative(v.state(), frame.ego, rp, rv);
            Measurement m;
            m.sensor = Sensor::Lidar;
            m.time = frame.time;
            m.ego = frame.ego;
            m.z = Eigen::Vector2d(rp.x() + noise(lidar_rng_), rp.y() + noise(lidar_rng_));
            m.truth_id = v.id();
            frame.detections.push_back(m);
        }
    }

    std::poisson_distribution<int> clutter(std::max(p.clutter_rate, 1e-12));
    const int false_alarms = p.clutter_rate > 0.0 ? clutter(lidar_rng_) : 0;
    for (int k = 0; k < false_alarms; ++k) {
        Measurement m;
        m.sensor = Sensor::Lidar;
        m.time = frame.time;
        m.ego = frame.ego;
        m.z = Eigen::Vector2d(fov[0] + unit(lidar_rng_) * (fov[1] - fov[0]),
                              fov[2] + unit(lidar_rng_) * (fov[3] - fov[2]));
        frame.detections.push_back(m);
    }
    std::shuffle(frame.detections.begin(), frame.detections.end(), lidar_rng_);
    return frame;
}

SensorFrame Simulator::radarFrame() {
    SensorFrame frame;
    frame.time = time();
    frame.sensor = Sensor::Radar;
    frame.ego = egoPose();
    const auto& p = config_.sim.radar;
    const auto& n = config_.noise;
    const auto& fov = config_.sim.fov;
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    std::normal_distribution<double> nr(0.0, n.radar_range_std), nb(0.0, n.radar_bearing_std),
        nd(0.0, n.radar_rate_std);

    for (const Vehicle& v : vehicles_) {
        if (!visible(v) || unit(radar_rng_) > p.detection_probability) continue;
        Eigen::Vector2d rp, rv;
        relative(v.state(), frame.ego, rp, rv);
        const double rho = rp.norm();
        const double phi = std::atan2(rp.y(), rp.x());
        const double rho_dot = rp.dot(rv) / std::max(rho, 1e-6);
        Measurement m;
        m.sensor = Sensor::Radar;
        m.time = frame.time;
        m.ego = frame.ego;
        m.z = Eigen::Vector3d(rho + nr(radar_rng_), wrapAngle(phi + nb(radar_rng_)),
                              rho_dot + nd(radar_rng_));
        m.truth_id = v.id();
        frame.detections.push_back(m);
    }

    std::poisson_distribution<int> clutter(std::max(p.clutter_rate, 1e-12));
    const int false_alarms = p.clutter_rate > 0.0 ? clutter(radar_rng_) : 0;
    for (int k = 0; k < false_alarms; ++k) {
        const double x = fov[0] + unit(radar_rng_) * (fov[1] - fov[0]);
        const double y = fov[2] + unit(radar_rng_) * (fov[3] - fov[2]);
        Measurement m;
        m.sensor = Sensor::Radar;
        m.time = frame.time;
        m.ego = frame.ego;
        m.z = Eigen::Vector3d(std::hypot(x, y), std::atan2(y, x), -15.0 + 30.0 * unit(radar_rng_));
        frame.detections.push_back(m);
    }
    std::shuffle(frame.detections.begin(), frame.detections.end(), radar_rng_);
    return frame;
}

}  // namespace fusion
