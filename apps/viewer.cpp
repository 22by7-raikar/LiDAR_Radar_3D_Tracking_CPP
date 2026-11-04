// Interactive 3D view of a simulated run: road, traffic, lidar and radar
// detections, scan returns, and the tracker's estimates with their predicted
// paths.
//
// Keys: space pause/resume, n single step, r restart, p toggle predictions,
//       q quit.
//
// Off-screen use (e.g. under xvfb-run): --screenshot FILE --at T saves one
// image at time T and exits; --record DIR saves every --every-th tick.

#include "args.h"

#include "fusion/config.h"
#include "fusion/models.h"
#include "fusion/sim/simulator.h"
#include "fusion/tracking/tracker.h"

#include <pcl/visualization/pcl_visualizer.h>

#include <chrono>
#include <cmath>
#include <iomanip>
#include <memory>
#include <sstream>
#include <thread>

using namespace fusion;

namespace {

struct Rgb {
    double r, g, b;
};

constexpr Rgb kRoad{0.22, 0.23, 0.26}, kLane{0.92, 0.92, 0.92}, kEgo{0.16, 0.62, 0.56},
    kCar{0.42, 0.55, 0.75}, kLidar{0.82, 0.29, 0.36}, kRadar{0.6, 0.37, 0.75},
    kTrack{0.95, 0.62, 0.15}, kClutter{0.55, 0.58, 0.62}, kPoints{0.72, 0.85, 0.95};

class Viewer {
public:
    Viewer(const Config& config, uint64_t seed)
        : config_(config), seed_(seed), viewer_(new pcl::visualization::PCLVisualizer("fusion")) {
        viewer_->setBackgroundColor(0.05, 0.06, 0.07);
        viewer_->setShowFPS(false);
        viewer_->initCameraParameters();
        viewer_->setCameraPosition(-21, -8, 11, 16, 1, 0, 0, 0, 1);
        viewer_->registerKeyboardCallback(
            [this](const pcl::visualization::KeyboardEvent& e) { onKey(e); });
        restart();
    }

    // Steps without waiting, saving images as requested, then returns.
    void capture(const std::string& screenshot, double at, const std::string& record, int every) {
        viewer_->setSize(1280, 720);
        long tick_count = 0;
        while (!sim_->done()) {
            const double t = sim_->time();
            tick();
            viewer_->spinOnce(1);
            if (!record.empty() && tick_count % every == 0) {
                std::ostringstream name;
                name << record << "/frame_" << std::setw(5) << std::setfill('0') << tick_count << ".png";
                viewer_->saveScreenshot(name.str());
            }
            ++tick_count;
            if (!screenshot.empty() && t + 1e-9 >= at) {
                viewer_->saveScreenshot(screenshot);
                return;
            }
        }
    }

    void run() {
        const auto period = std::chrono::duration<double>(config_.sim.step);
        while (!viewer_->wasStopped()) {
            const auto start = std::chrono::steady_clock::now();
            if ((!paused_ || step_once_) && !sim_->done()) {
                tick();
                step_once_ = false;
            }
            viewer_->spinOnce(1);
            std::this_thread::sleep_until(start + period);
        }
    }

private:
    void restart() {
        sim_ = std::make_unique<Simulator>(config_, seed_);
        tracker_ = std::make_unique<MultiTargetTracker>(config_);
        last_lidar_.clear();
        last_radar_.clear();
        points_.clear();
    }

    void onKey(const pcl::visualization::KeyboardEvent& e) {
        if (!e.keyDown()) return;
        const std::string k = e.getKeySym();
        if (k == "space") paused_ = !paused_;
        else if (k == "n") step_once_ = paused_ = true;
        else if (k == "r") restart();
        else if (k == "p") predictions_ = !predictions_;
    }

    void tick() {
        for (SensorFrame& f : sim_->sense()) {
            tracker_->process(f.time, f.sensor, f.detections, f.ego);
            (f.sensor == Sensor::Lidar ? last_lidar_ : last_radar_) = f.detections;
            if (f.sensor == Sensor::Lidar && !f.points.empty()) points_ = std::move(f.points);
        }
        draw();
        sim_->advance();
    }

    void box(const std::string& id, const Eigen::Vector3f& c, float yaw, const Eigen::Vector3f& size,
             Rgb color, bool wire) {
        const Eigen::Quaternionf q(Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitZ()));
        viewer_->addCube(c, q, size.x(), size.y(), size.z(), id);
        viewer_->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, color.r,
                                             color.g, color.b, id);
        viewer_->setShapeRenderingProperties(
            pcl::visualization::PCL_VISUALIZER_REPRESENTATION,
            wire ? pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME
                 : pcl::visualization::PCL_VISUALIZER_REPRESENTATION_SURFACE,
            id);
        viewer_->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_SHADING,
                                             pcl::visualization::PCL_VISUALIZER_SHADING_FLAT, id);
    }

    // Body above the ground clearance and a narrower cabin, matching the
    // shape the lidar simulation ray-casts against. Position in the ego frame.
    void car(const std::string& id, double x, double y, double yaw, const VehicleParams& p, Rgb c) {
        const float L = static_cast<float>(p.length), W = static_cast<float>(p.width),
                    H = static_cast<float>(p.height);
        const float fx = static_cast<float>(x), fy = static_cast<float>(y);
        const float clearance = 0.15f * H, belt = 0.6f * H;
        const float cabin_shift = -0.075f * L;  // cabin spans -0.35 L … 0.2 L
        box(id + "_body", {fx, fy, 0.5f * (clearance + belt)}, static_cast<float>(yaw),
            {L, W, belt - clearance}, c, false);
        const float cx = fx + cabin_shift * std::cos(static_cast<float>(yaw));
        const float cy = fy + cabin_shift * std::sin(static_cast<float>(yaw));
        box(id + "_cabin", {cx, cy, 0.5f * (belt + H)}, static_cast<float>(yaw),
            {0.55f * L, 0.9f * W, H - belt}, {0.6 * c.r + 0.1, 0.6 * c.g + 0.1, 0.6 * c.b + 0.1},
            false);
    }

    // World → ego frame (the view follows the ego vehicle).
    Eigen::Vector3d toEgo(double x, double y, double yaw) const {
        const EgoPose e = sim_->egoPose();
        const double c = std::cos(e.yaw), s = std::sin(e.yaw);
        const double dx = x - e.x, dy = y - e.y;
        return {c * dx + s * dy, -s * dx + c * dy, yaw - e.yaw};
    }

    void draw() {
        viewer_->removeAllShapes();
        viewer_->removeAllPointClouds();

        // Three-lane road; dashed lane markings scroll with the ego vehicle.
        viewer_->addCube(-40, 80, -5.55, 5.55, -0.05, 0, kRoad.r, kRoad.g, kRoad.b, "road");
        viewer_->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION,
                                             pcl::visualization::PCL_VISUALIZER_REPRESENTATION_SURFACE,
                                             "road");
        for (const double y : {-5.55, 5.55}) {
            const std::string id = "edge" + std::to_string(static_cast<int>(y * 10));
            viewer_->addLine(pcl::PointXYZ(-40, static_cast<float>(y), 0.02f),
                             pcl::PointXYZ(80, static_cast<float>(y), 0.02f), 0.9, 0.9, 0.9, id);
            viewer_->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, id);
        }
        const double ego_x = sim_->egoPose().x;
        const double phase = std::fmod(ego_x, 12.0);
        int dash = 0;
        for (const double y : {-1.85, 1.85})
            for (double x = -40.0 - phase; x < 80.0; x += 12.0) {
                const std::string id = "dash" + std::to_string(dash++);
                viewer_->addLine(pcl::PointXYZ(static_cast<float>(x), static_cast<float>(y), 0.02f),
                                 pcl::PointXYZ(static_cast<float>(x + 3.0), static_cast<float>(y), 0.02f),
                                 kLane.r, kLane.g, kLane.b, id);
                viewer_->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, id);
            }

        car("ego", 0, 0, 0, config_.sim.ego, kEgo);
        for (const Vehicle& v : sim_->vehicles()) {
            const VehicleState s = v.state();
            const Eigen::Vector3d e = toEgo(s.x, s.y, s.yaw);
            car("car" + std::to_string(v.id()), e.x(), e.y(), e.z(), v.params(), kCar);
        }

        if (!points_.empty()) {
            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
            for (const auto& p : points_)
                cloud->push_back(pcl::PointXYZ(static_cast<float>(p.p.x()),
                                               static_cast<float>(p.p.y()),
                                               static_cast<float>(p.p.z()) + 0.02f));
            viewer_->addPointCloud<pcl::PointXYZ>(cloud, "scan");
            viewer_->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR,
                                                      kPoints.r, kPoints.g, kPoints.b, "scan");
            viewer_->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
                                                      2, "scan");
        }

        int n = 0;
        for (const Measurement& m : last_lidar_) {
            const Rgb c = m.truth_id < 0 ? kClutter : kLidar;
            viewer_->addSphere(pcl::PointXYZ(static_cast<float>(m.z(0)), static_cast<float>(m.z(1)), 2.2f),
                               0.3, c.r, c.g, c.b, "lidar" + std::to_string(n++));
        }
        for (const Measurement& m : last_radar_) {
            const Eigen::Vector2d p(m.z(0) * std::cos(m.z(1)), m.z(0) * std::sin(m.z(1)));
            const Eigen::Vector2d tip = p + m.z(2) * p.normalized();
            const Rgb c = m.truth_id < 0 ? kClutter : kRadar;
            const std::string id = "radar" + std::to_string(n++);
            viewer_->addLine(pcl::PointXYZ(0, 0, 1.9f), pcl::PointXYZ(p.x(), p.y(), 1.9f), c.r, c.g,
                             c.b, id);
            viewer_->addArrow(pcl::PointXYZ(tip.x(), tip.y(), 1.9f), pcl::PointXYZ(p.x(), p.y(), 1.9f),
                              c.r, c.g, c.b, false, id + "_rate");
        }

        for (const Track& t : tracker_->tracks()) {
            if (!t.initialised() || !t.confirmed) continue;
            auto f = t.filter->clone();
            f->predict(sim_->time() - t.time);
            const State x = f->state();
            const std::string id = "track" + std::to_string(t.id);
            const Eigen::Vector3d e = toEgo(x(0), x(1), x(kYaw));
            viewer_->addSphere(pcl::PointXYZ(e.x(), e.y(), 2.6), 0.4, kTrack.r, kTrack.g, kTrack.b, id);
            // Velocity relative to the ego vehicle, 1 s long.
            const EgoPose ego = sim_->egoPose();
            const Eigen::Vector2d vw = velocityXY(x) - Eigen::Vector2d(ego.vx, ego.vy);
            const double c = std::cos(ego.yaw), sn = std::sin(ego.yaw);
            const Eigen::Vector2d v(c * vw.x() + sn * vw.y(), -sn * vw.x() + c * vw.y());
            viewer_->addArrow(pcl::PointXYZ(e.x() + v.x(), e.y() + v.y(), 2.6),
                              pcl::PointXYZ(e.x(), e.y(), 2.6), kTrack.r, kTrack.g, kTrack.b, false,
                              id + "_v");
            viewer_->addText3D(std::to_string(t.id), pcl::PointXYZ(e.x(), e.y(), 3.3), 0.6, 1, 1, 1,
                               id + "_label");
            if (predictions_)
                for (int k = 1; k <= 6; ++k) {
                    f->predict(0.25);
                    const State p = f->state();
                    const std::string pid = id + "_p" + std::to_string(k);
                    // Predicted position relative to where the ego vehicle will be.
                    const EgoPose eg = sim_->egoPose();
                    const double dt = 0.25 * k;
                    const Eigen::Vector3d e2 = toEgo(p(0) - eg.vx * dt, p(1) - eg.vy * dt, 0.0);
                    viewer_->addSphere(pcl::PointXYZ(e2.x(), e2.y(), 2.6), 0.25, kTrack.r, kTrack.g,
                                       kTrack.b, pid);
                    viewer_->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY,
                                                         1.0 - 0.13 * k, pid);
                }
        }

        std::ostringstream text;
        text << std::fixed << std::setprecision(2) << "t = " << sim_->time() << " s   filter "
             << config_.filter.type << "   association " << config_.tracker.association
             << (paused_ ? "   [paused]" : "");
        viewer_->addText(text.str(), 20, 20, 16, 1, 1, 1, "status");
    }

    Config config_;
    uint64_t seed_;
    pcl::visualization::PCLVisualizer::Ptr viewer_;
    std::unique_ptr<Simulator> sim_;
    std::unique_ptr<MultiTargetTracker> tracker_;
    std::vector<Measurement> last_lidar_, last_radar_;
    std::vector<ScanPoint> points_;
    bool paused_ = false, step_once_ = false, predictions_ = true;
};

}  // namespace

int main(int argc, char** argv) {
    const Args args(argc, argv,
                    {{"config", "config/default.yaml"}, {"seed", "1"}, {"screenshot", ""},
                     {"at", "5"}, {"record", ""}, {"every", "3"}},
                    "usage: fusion_viewer [--config FILE] [--seed N]\n"
                    "                     [--screenshot FILE --at T] [--record DIR --every N]\n"
                    "keys: space pause, n step, r restart, p predictions, q quit");
    Viewer viewer(loadConfig(args.str("config")), std::stoull(args.str("seed")));
    if (!args.str("screenshot").empty() || !args.str("record").empty())
        viewer.capture(args.str("screenshot"), std::stod(args.str("at")), args.str("record"),
                       args.integer("every"));
    else
        viewer.run();
    return 0;
}
