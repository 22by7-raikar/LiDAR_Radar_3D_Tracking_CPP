#include "fusion/sim/lidar_scanner.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <unordered_map>

namespace fusion {
namespace {

constexpr double kDeg = M_PI / 180.0;

// Ray parameter of the first intersection with an axis-aligned box, or
// infinity (slab method).
double rayBox(const Eigen::Vector3d& o, const Eigen::Vector3d& d, const Eigen::Vector3d& lo,
              const Eigen::Vector3d& hi) {
    double t0 = 0.0, t1 = std::numeric_limits<double>::infinity();
    for (int k = 0; k < 3; ++k) {
        if (std::abs(d(k)) < 1e-12) {
            if (o(k) < lo(k) || o(k) > hi(k)) return std::numeric_limits<double>::infinity();
            continue;
        }
        double a = (lo(k) - o(k)) / d(k), b = (hi(k) - o(k)) / d(k);
        if (a > b) std::swap(a, b);
        t0 = std::max(t0, a);
        t1 = std::min(t1, b);
        if (t0 > t1) return std::numeric_limits<double>::infinity();
    }
    return t0;
}

// Pose of a vehicle in the sensor (ego) frame.
struct LocalPose {
    double x, y, yaw;
};

LocalPose localPose(const Vehicle& v, const EgoPose& ego) {
    const VehicleState s = v.state();
    const double c = std::cos(ego.yaw), sn = std::sin(ego.yaw);
    const double dx = s.x - ego.x, dy = s.y - ego.y;
    return {c * dx + sn * dy, -sn * dx + c * dy, s.yaw - ego.yaw};
}

// First hit of a ray with a vehicle: a body box above the ground clearance
// and a narrower cabin box set back from the bonnet.
double rayVehicle(const Eigen::Vector3d& o, const Eigen::Vector3d& d, const LocalPose& pose,
                  const VehicleParams& p) {
    const double c = std::cos(-pose.yaw), sn = std::sin(-pose.yaw);
    auto local = [&](const Eigen::Vector3d& w, bool point) {
        const double x = point ? w.x() - pose.x : w.x(), y = point ? w.y() - pose.y : w.y();
        return Eigen::Vector3d(c * x - sn * y, sn * x + c * y, w.z());
    };
    const Eigen::Vector3d ol = local(o, true), dl = local(d, false);
    const double L = p.length, W = p.width, H = p.height;
    const double clearance = 0.15 * H, belt = 0.6 * H;
    const double body = rayBox(ol, dl, {-L / 2, -W / 2, clearance}, {L / 2, W / 2, belt});
    const double cabin =
        rayBox(ol, dl, {-0.35 * L, -0.45 * W, belt}, {0.2 * L, 0.45 * W, H});
    return std::min(body, cabin);
}

}  // namespace

LidarScanner::LidarScanner(const ScanParams& params) : params_(params) {}

std::vector<ScanPoint> LidarScanner::scan(const std::vector<Vehicle>& vehicles,
                                          const EgoPose& ego, std::mt19937_64& rng) const {
    std::vector<LocalPose> poses;
    for (const Vehicle& v : vehicles) poses.push_back(localPose(v, ego));

    const int columns = static_cast<int>(std::lround(360.0 / params_.azimuth_step_deg));
    const double step = 2.0 * M_PI / columns;

    // Only azimuths that can reach a vehicle are cast.
    std::vector<char> active(columns, 0);
    for (size_t k = 0; k < vehicles.size(); ++k) {
        const LocalPose& s = poses[k];
        const double r = 0.5 * std::hypot(vehicles[k].params().length, vehicles[k].params().width);
        const double d = std::hypot(s.x, s.y);
        if (d <= r) {
            std::fill(active.begin(), active.end(), 1);
            break;
        }
        const double centre = std::atan2(s.y, s.x), half = std::asin(r / d);
        const int a = static_cast<int>(std::floor((centre - half) / step));
        const int b = static_cast<int>(std::ceil((centre + half) / step));
        for (int i = a; i <= b; ++i) active[((i % columns) + columns) % columns] = 1;
    }

    std::normal_distribution<double> range_noise(0.0, params_.range_std);
    const Eigen::Vector3d origin(0.0, 0.0, params_.mount_height);
    const double layer_step = params_.fov_deg / params_.layers;
    std::vector<ScanPoint> points;
    for (int col = 0; col < columns; ++col) {
        if (!active[col]) continue;
        const double az = col * step;
        for (int layer = 0; layer < params_.layers; ++layer) {
            const double el = (params_.lowest_deg + layer * layer_step) * kDeg;
            const Eigen::Vector3d dir(std::cos(el) * std::cos(az), std::cos(el) * std::sin(az),
                                      std::sin(el));
            const double ground = dir.z() < 0.0 ? params_.mount_height / -dir.z()
                                                : std::numeric_limits<double>::infinity();
            double best = std::numeric_limits<double>::infinity();
            int hit = -1;
            for (size_t k = 0; k < vehicles.size(); ++k) {
                const double t = rayVehicle(origin, dir, poses[k], vehicles[k].params());
                if (t < best) {
                    best = t;
                    hit = static_cast<int>(k);
                }
            }
            if (hit < 0 || best >= ground || best > params_.max_range) continue;
            points.push_back({origin + (best + range_noise(rng)) * dir, vehicles[hit].id()});
        }
    }
    return points;
}

std::vector<ScanDetection> LidarScanner::detect(const std::vector<ScanPoint>& points) const {
    // Connected components of occupied grid cells (8-neighbourhood).
    const double cell = params_.cluster_cell;
    auto key = [](int32_t i, int32_t j) {
        return (static_cast<uint64_t>(static_cast<uint32_t>(i)) << 32) | static_cast<uint32_t>(j);
    };
    std::unordered_map<uint64_t, std::vector<int>> grid;
    for (int i = 0; i < static_cast<int>(points.size()); ++i)
        grid[key(static_cast<int32_t>(std::floor(points[i].p.x() / cell)),
                 static_cast<int32_t>(std::floor(points[i].p.y() / cell)))]
            .push_back(i);

    std::unordered_map<uint64_t, int> label;
    std::vector<std::vector<int>> clusters;
    for (const auto& [k0, members0] : grid) {
        if (label.count(k0)) continue;
        const int id = static_cast<int>(clusters.size());
        clusters.emplace_back();
        std::vector<uint64_t> stack{k0};
        label[k0] = id;
        while (!stack.empty()) {
            const uint64_t k = stack.back();
            stack.pop_back();
            const auto& members = grid.at(k);
            clusters[id].insert(clusters[id].end(), members.begin(), members.end());
            const auto ci = static_cast<int32_t>(static_cast<uint32_t>(k >> 32));
            const auto cj = static_cast<int32_t>(static_cast<uint32_t>(k & 0xffffffffu));
            for (int32_t di = -1; di <= 1; ++di)
                for (int32_t dj = -1; dj <= 1; ++dj) {
                    const uint64_t n = key(ci + di, cj + dj);
                    if (!grid.count(n) || label.count(n)) continue;
                    label[n] = id;
                    stack.push_back(n);
                }
        }
    }

    // A vehicle seen from behind can fall apart into its rear faces and a
    // single scan line on the roof, separated by more than a car length's
    // worth of empty grid. Merge clusters while their union still fits in a
    // vehicle footprint (smallest enclosing rectangle over 5° steps).
    if (params_.merge_margin >= 0.0) {
        auto fits = [&](const std::vector<int>& a, const std::vector<int>& b) {
            const double len = params_.nominal_length + params_.merge_margin;
            const double wid = params_.nominal_width + params_.merge_margin;
            const size_t stride_a = std::max<size_t>(1, a.size() / 200);
            const size_t stride_b = std::max<size_t>(1, b.size() / 200);
            for (double deg = 0.0; deg < 90.0; deg += 5.0) {
                const Eigen::Vector2d e1(std::cos(deg * kDeg), std::sin(deg * kDeg));
                const Eigen::Vector2d e2(-e1.y(), e1.x());
                double lo1 = 1e18, hi1 = -1e18, lo2 = 1e18, hi2 = -1e18;
                auto add = [&](const std::vector<int>& c, size_t stride) {
                    for (size_t i = 0; i < c.size(); i += stride) {
                        const Eigen::Vector2d q = points[c[i]].p.head<2>();
                        lo1 = std::min(lo1, q.dot(e1));
                        hi1 = std::max(hi1, q.dot(e1));
                        lo2 = std::min(lo2, q.dot(e2));
                        hi2 = std::max(hi2, q.dot(e2));
                    }
                };
                add(a, stride_a);
                add(b, stride_b);
                const double big = std::max(hi1 - lo1, hi2 - lo2);
                const double small = std::min(hi1 - lo1, hi2 - lo2);
                if (big <= len && small <= wid) return true;
            }
            return false;
        };
        for (bool merged = true; merged;) {
            merged = false;
            for (size_t i = 0; i < clusters.size() && !merged; ++i)
                for (size_t j = i + 1; j < clusters.size() && !merged; ++j)
                    if (fits(clusters[i], clusters[j])) {
                        clusters[i].insert(clusters[i].end(), clusters[j].begin(),
                                           clusters[j].end());
                        clusters.erase(clusters.begin() + static_cast<long>(j));
                        merged = true;
                    }
        }
    }

    std::vector<ScanDetection> out;
    for (const auto& cl : clusters) {
        if (static_cast<int>(cl.size()) < params_.min_points) continue;
        ScanDetection det;
        det.points = static_cast<int>(cl.size());
        std::vector<Eigen::Vector2d> xy;
        std::map<int, int> votes;
        Eigen::Vector2d sum = Eigen::Vector2d::Zero();
        for (int i : cl) {
            xy.push_back(points[i].p.head<2>());
            sum += xy.back();
            ++votes[points[i].vehicle];
        }
        det.centroid = sum / static_cast<double>(cl.size());
        det.vehicle = std::max_element(votes.begin(), votes.end(), [](auto& a, auto& b) {
                          return a.second < b.second;
                      })->first;

        // Fit to the visible outline only: in each 0.25° azimuth bin keep the
        // nearest return. Returns from the cabin and roof lie inside the body's
        // footprint and would otherwise tilt the fitted rectangle.
        {
            std::map<long, Eigen::Vector2d> nearest;
            for (const auto& q : xy) {
                const long bin = std::lround(std::atan2(q.y(), q.x()) / (0.25 * kDeg));
                auto it = nearest.find(bin);
                if (it == nearest.end() || q.squaredNorm() < it->second.squaredNorm())
                    nearest[bin] = q;
            }
            if (nearest.size() >= 3) {
                xy.clear();
                for (const auto& [bin, q] : nearest) xy.push_back(q);
            }
        }

        // L-shape fit, variance criterion: assign each point to the nearer of
        // its two closest box edges and minimise the spread of those distances.
        double best_score = -std::numeric_limits<double>::infinity(), best_theta = 0.0;
        for (double deg = 0.0; deg < 90.0; deg += 1.0) {
            const double th = deg * kDeg;
            const Eigen::Vector2d e1(std::cos(th), std::sin(th)), e2(-e1.y(), e1.x());
            double min1 = 1e18, max1 = -1e18, min2 = 1e18, max2 = -1e18;
            for (const auto& q : xy) {
                min1 = std::min(min1, q.dot(e1));
                max1 = std::max(max1, q.dot(e1));
                min2 = std::min(min2, q.dot(e2));
                max2 = std::max(max2, q.dot(e2));
            }
            double s1 = 0, ss1 = 0, s2 = 0, ss2 = 0;
            int n1 = 0, n2 = 0;
            for (const auto& q : xy) {
                const double d1 = std::min(max1 - q.dot(e1), q.dot(e1) - min1);
                const double d2 = std::min(max2 - q.dot(e2), q.dot(e2) - min2);
                if (d1 < d2) {
                    s1 += d1, ss1 += d1 * d1, ++n1;
                } else {
                    s2 += d2, ss2 += d2 * d2, ++n2;
                }
            }
            const double v1 = n1 ? ss1 / n1 - (s1 / n1) * (s1 / n1) : 0.0;
            const double v2 = n2 ? ss2 / n2 - (s2 / n2) * (s2 / n2) : 0.0;
            if (-(v1 + v2) > best_score) {
                best_score = -(v1 + v2);
                best_theta = th;
            }
        }
        const Eigen::Vector2d e1(std::cos(best_theta), std::sin(best_theta)), e2(-e1.y(), e1.x());
        double min1 = 1e18, max1 = -1e18, min2 = 1e18, max2 = -1e18;
        for (const auto& q : xy) {
            min1 = std::min(min1, q.dot(e1));
            max1 = std::max(max1, q.dot(e1));
            min2 = std::min(min2, q.dot(e2));
            max2 = std::max(max2, q.dot(e2));
        }
        det.yaw = best_theta;
        det.extent = {max1 - min1, max2 - min2};
        Eigen::Vector2d c = e1 * 0.5 * (min1 + max1) + e2 * 0.5 * (min2 + max2);

        // Which axis is the vehicle's length: the one seen longer than a car
        // is wide; if only one face is visible, the axis closer to the line
        // of sight (the visible face is then the rear or the front).
        const Eigen::Vector2d sight = c.normalized();
        int length_axis;
        if (std::max(det.extent(0), det.extent(1)) > params_.nominal_width + 0.3)
            length_axis = det.extent(0) >= det.extent(1) ? 0 : 1;
        else
            length_axis = std::abs(sight.dot(e1)) >= std::abs(sight.dot(e2)) ? 0 : 1;
        // Seen over another vehicle, only the cabin may be visible: no returns
        // below the belt line. Complete the cabin's box instead of the body's,
        // then move from the cabin centre to the body centre. The cabin sits
        // 0.075 L behind the body centre; "behind" assumes the vehicle travels
        // in the ego vehicle's direction, as traffic on a highway does.
        double min_z = 1e18;
        for (int i : cl) min_z = std::min(min_z, points[i].p.z());
        const double belt = 0.6 * params_.nominal_height;
        det.cabin_only = min_z > belt + 0.05;
        const double len = det.cabin_only ? 0.55 * params_.nominal_length : params_.nominal_length;
        const double wid = det.cabin_only ? 0.9 * params_.nominal_width : params_.nominal_width;
        const Eigen::Vector2d axes[2] = {e1, e2};
        for (int k = 0; k < 2; ++k) {
            const double nominal = k == length_axis ? len : wid;
            if (det.extent(k) >= nominal) continue;
            const double away = c.dot(axes[k]) >= 0.0 ? 1.0 : -1.0;
            c += away * 0.5 * (nominal - det.extent(k)) * axes[k];
        }
        if (det.cabin_only) {
            const Eigen::Vector2d forward =
                axes[length_axis].x() >= 0.0 ? axes[length_axis] : Eigen::Vector2d(-axes[length_axis]);
            c += 0.075 * params_.nominal_length * forward;
        }
        det.center = c;
        out.push_back(det);
    }
    return out;
}

}  // namespace fusion
