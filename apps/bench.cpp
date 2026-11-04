// Monte Carlo comparison of configurations: runs every config file on the
// same seeds and writes per-run metrics plus a summary table.
//
//   fusion_bench [--runs N] [--threads N] [--out FILE.csv] CONFIG.yaml...

#include "fusion/config.h"
#include "fusion/experiment.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

using namespace fusion;

namespace {

struct Stat {
    std::vector<double> v;
    double mean() const {
        double s = 0;
        for (double x : v) s += x;
        return v.empty() ? 0.0 : s / static_cast<double>(v.size());
    }
};

[[noreturn]] void usage() {
    std::cerr << "usage: fusion_bench [--runs N] [--threads N] [--out FILE.csv] CONFIG.yaml...\n";
    std::exit(2);
}

}  // namespace

int main(int argc, char** argv) {
    int runs = 100;
    int threads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    std::string out;
    std::vector<std::string> configs;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if ((a == "--runs" || a == "--threads" || a == "--out") && i + 1 < argc) {
            const std::string v = argv[++i];
            if (a == "--runs") runs = std::stoi(v);
            else if (a == "--threads") threads = std::stoi(v);
            else out = v;
        } else if (a.rfind("--", 0) == 0) {
            usage();
        } else {
            configs.push_back(a);
        }
    }
    if (configs.empty()) usage();

    std::ofstream csv;
    if (!out.empty()) {
        if (std::filesystem::path(out).has_parent_path())
            std::filesystem::create_directories(std::filesystem::path(out).parent_path());
        csv.open(out);
        csv << "config,seed,rmse_px,rmse_py,rmse_vx,rmse_vy,velocity_p95,latency,nees_position,"
               "nis_lidar,nis_radar,nis_lidar_in95,nis_radar_in95,ospa,id_switches,false_tracks,"
               "missed,us_per_detection\n";
    }

    std::cout << std::left << std::setw(28) << "config" << std::right << std::fixed
              << std::setprecision(3) << std::setw(7) << "px" << std::setw(7) << "py"
              << std::setw(7) << "vx" << std::setw(7) << "vy" << std::setw(7) << "p95" << std::setw(7) << "lat"
              << std::setw(7) << "NEES" << std::setw(7) << "NISl" << std::setw(7) << "NISr"
              << std::setw(7) << "OSPA" << std::setw(7) << "IDsw" << std::setw(7) << "false"
              << std::setw(7) << "miss" << std::setw(8) << "us/det" << "\n";

    for (const std::string& path : configs) {
        const Config config = loadConfig(path);
        std::vector<RunMetrics> results(runs);
        std::atomic<int> next{0};
        std::vector<std::thread> pool;
        for (int t = 0; t < threads; ++t)
            pool.emplace_back([&] {
                for (int i = next++; i < runs; i = next++)
                    results[i] = runSimulation(config, static_cast<uint64_t>(i + 1));
            });
        for (auto& th : pool) th.join();

        Stat r[4], p95, lat, nees, nisl, nisr, ospa, sw, fa, miss, us;
        const std::string name = std::filesystem::path(path).stem().string();
        for (int i = 0; i < runs; ++i) {
            const RunMetrics& m = results[i];
            for (int k = 0; k < 4; ++k) r[k].v.push_back(m.rmse(k) * m.rmse(k));
            p95.v.push_back(m.velocity_p95);
            lat.v.push_back(m.latency);
            nees.v.push_back(m.nees_position);
            nisl.v.push_back(m.nis_lidar);
            nisr.v.push_back(m.nis_radar);
            ospa.v.push_back(m.ospa);
            sw.v.push_back(m.id_switches);
            fa.v.push_back(m.false_tracks);
            miss.v.push_back(m.missed);
            us.v.push_back(m.us_per_detection);
            if (csv.is_open()) {
                csv << name << "," << i + 1;
                for (int k = 0; k < 4; ++k) csv << "," << m.rmse(k);
                csv << "," << m.velocity_p95 << "," << m.latency << "," << m.nees_position << "," << m.nis_lidar << ","
                    << m.nis_radar << "," << m.nis_lidar_in95 << "," << m.nis_radar_in95 << ","
                    << m.ospa << "," << m.id_switches << "," << m.false_tracks << "," << m.missed
                    << "," << m.us_per_detection << "\n";
            }
        }
        // RMSE over all runs pools the squared errors of the runs.
        std::cout << std::left << std::setw(28) << name << std::right;
        for (auto& s : r) std::cout << std::setw(7) << std::sqrt(s.mean());
        std::cout << std::setw(7) << p95.mean() << std::setw(7) << lat.mean() << std::setw(7)
                  << nees.mean() << std::setw(7)
                  << nisl.mean() << std::setw(7) << nisr.mean() << std::setw(7) << ospa.mean()
                  << std::setw(7) << sw.mean() << std::setw(7) << fa.mean() << std::setw(7)
                  << miss.mean() << std::setw(8) << std::setprecision(1) << us.mean()
                  << std::setprecision(3) << "\n";
    }
    return 0;
}
