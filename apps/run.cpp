// Runs one simulation and writes everything it produced as CSV files.

#include "args.h"

#include "fusion/config.h"
#include "fusion/experiment.h"
#include "fusion/models.h"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>

using namespace fusion;

namespace {

std::ofstream open(const std::filesystem::path& dir, const std::string& name,
                   const std::string& header) {
    std::ofstream f(dir / name);
    f << std::setprecision(9) << header << "\n";
    return f;
}

void printSummary(const RunMetrics& m) {
    std::cout << std::fixed << std::setprecision(3)
              << "RMSE [px py vx vy]:        " << m.rmse.transpose() << "\n"
              << "velocity error p95 [m/s]:  " << m.velocity_p95 << "\n"
              << "track start latency [s]:   " << m.latency << "\n"
              << "position NEES (2 ideal):   " << m.nees_position << "\n"
              << "lidar NIS (2), in 95%:     " << m.nis_lidar << ", " << m.nis_lidar_in95 << "\n"
              << "radar NIS (3), in 95%:     " << m.nis_radar << ", " << m.nis_radar_in95 << "\n"
              << "OSPA [m]:                  " << m.ospa << "\n"
              << "ID switches:               " << m.id_switches << "\n"
              << "false tracks per step:     " << m.false_tracks << "\n"
              << "missed share:              " << m.missed << "\n"
              << "tracker µs per detection:  " << m.us_per_detection << "\n";
}

}  // namespace

int main(int argc, char** argv) {
    const Args args(argc, argv, {{"config", "config/default.yaml"}, {"seed", "1"}, {"out", ""}},
                    "usage: fusion_run [--config FILE] [--seed N] [--out DIR]");
    const Config config = loadConfig(args.str("config"));
    const auto seed = static_cast<uint64_t>(std::stoull(args.str("seed")));

    RunLog log;
    const RunMetrics m = runSimulation(config, seed, args.str("out").empty() ? nullptr : &log);
    printSummary(m);
    if (args.str("out").empty()) return 0;

    const std::filesystem::path dir(args.str("out"));
    std::filesystem::create_directories(dir);
    {
        auto f = open(dir, "truth.csv", "time,vehicle,x,y,v,yaw,yaw_rate");
        for (const auto& t : log.truth)
            f << t.time << "," << t.vehicle << "," << t.s.x << "," << t.s.y << "," << t.s.v << ","
              << t.s.yaw << "," << t.s.yaw_rate << "\n";
    }
    {
        auto f = open(dir, "ego.csv", "time,x,y,yaw,vx,vy");
        for (size_t i = 0; i < log.ego.size(); ++i)
            f << log.ego_times[i] << "," << log.ego[i].x << "," << log.ego[i].y << ","
              << log.ego[i].yaw << "," << log.ego[i].vx << "," << log.ego[i].vy << "\n";
    }
    {
        // z0..z2 in the ego frame as measured; wx, wy the world position.
        auto f = open(dir, "measurements.csv", "time,sensor,z0,z1,z2,truth,wx,wy");
        for (const auto& m : log.measurements) {
            const Eigen::Vector2d w = detectionPosition(m);
            f << m.time << "," << sensorName(m.sensor) << "," << m.z(0) << "," << m.z(1) << ","
              << (m.z.size() > 2 ? m.z(2) : 0.0) << "," << m.truth_id << "," << w.x() << ","
              << w.y() << "\n";
        }
    }
    {
        auto f = open(dir, "estimates.csv",
                      "time,track,x,y,v,yaw,yaw_rate,pxx,pxy,pyy,mu0,mu1,mu2");
        for (const auto& e : log.estimates) {
            f << e.time << "," << e.track;
            for (int i = 0; i < 5; ++i) f << "," << e.x(i);
            f << "," << e.P(0, 0) << "," << e.P(0, 1) << "," << e.P(1, 1);
            for (int i = 0; i < 3; ++i)
                f << "," << (i < e.mode_probabilities.size() ? e.mode_probabilities(i) : 0.0);
            f << "\n";
        }
    }
    {
        auto f = open(dir, "scored.csv", "time,vehicle,track,ex,ey,evx,evy,nees");
        for (const auto& s : log.scored)
            f << s.time << "," << s.vehicle << "," << s.track << "," << s.error(0) << ","
              << s.error(1) << "," << s.error(2) << "," << s.error(3) << "," << s.nees << "\n";
    }
    {
        auto f = open(dir, "updates.csv", "time,track,sensor,nis,truth");
        for (size_t i = 0; i < log.updates.size(); ++i)
            f << log.update_times[i] << "," << log.updates[i].track_id << ","
              << sensorName(log.updates[i].sensor) << "," << log.updates[i].nis << ","
              << log.updates[i].truth_id << "\n";
    }
    if (!log.scans.empty()) {
        auto p = open(dir, "scan_points.csv", "time,x,y,z,vehicle");
        auto o = open(dir, "scan_objects.csv",
                      "time,centroid_x,centroid_y,center_x,center_y,yaw,extent0,extent1,points,vehicle");
        for (const auto& frame : log.scans) {
            for (const auto& q : frame.points)
                p << frame.time << "," << q.p.x() << "," << q.p.y() << "," << q.p.z() << ","
                  << q.vehicle << "\n";
            for (const auto& d : frame.objects)
                o << frame.time << "," << d.centroid.x() << "," << d.centroid.y() << ","
                  << d.center.x() << "," << d.center.y() << "," << d.yaw << "," << d.extent(0)
                  << "," << d.extent(1) << "," << d.points << "," << d.vehicle << "\n";
        }
    }
    std::cout << "wrote " << dir.string() << "\n";
    return 0;
}
