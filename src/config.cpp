#include "fusion/config.h"

#include <yaml-cpp/yaml.h>

#include <filesystem>
#include <set>
#include <stdexcept>

namespace fusion {
namespace {

// Reads a YAML map into existing defaults and rejects keys nobody asked for,
// so a typo in a config file fails loudly instead of being silently ignored.
class Section {
public:
    // `node` may be absent (missing section) or null ("key:" with no value).
    Section(const YAML::Node& node, std::string path)
        : node_(node), present_(node.IsDefined() && !node.IsNull()), path_(std::move(path)) {
        if (present_ && !node_.IsMap())
            throw std::runtime_error("config: '" + path_ + "' must be a map");
    }

    ~Section() noexcept(false) {
        if (!present_ || std::uncaught_exceptions() > 0) return;
        for (const auto& kv : node_) {
            const auto key = kv.first.as<std::string>();
            if (!used_.count(key))
                throw std::runtime_error("config: unknown key '" + qualified(key) + "'");
        }
    }

    template <typename T>
    void get(const std::string& key, T& value) {
        used_.insert(key);
        const YAML::Node v = lookup(key);
        if (!v.IsDefined() || v.IsNull()) return;
        try {
            value = v.as<T>();
        } catch (const YAML::Exception&) {
            throw std::runtime_error("config: bad value for '" + qualified(key) + "'");
        }
    }

    template <size_t N>
    void get(const std::string& key, std::array<double, N>& value) {
        std::vector<double> v(value.begin(), value.end());
        get(key, v);
        if (v.size() != N)
            throw std::runtime_error("config: '" + qualified(key) + "' needs " +
                                     std::to_string(N) + " values");
        std::copy(v.begin(), v.end(), value.begin());
    }

    Section child(const std::string& key) {
        used_.insert(key);
        return Section(lookup(key), qualified(key));
    }

    // Raw node of a key that is parsed by the caller (lists of maps).
    YAML::Node raw(const std::string& key) {
        used_.insert(key);
        return lookup(key);
    }

    std::string qualified(const std::string& key) const {
        return path_.empty() ? key : path_ + "." + key;
    }

private:
    // Looks up through a const node: the non-const operator[] would insert the key.
    YAML::Node lookup(const std::string& key) const {
        if (!present_) return YAML::Node(YAML::NodeType::Undefined);
        const YAML::Node& self = node_;
        return self[key];
    }

    YAML::Node node_;
    bool present_;
    std::string path_;
    std::set<std::string> used_;
};

void readScan(Section s, ScanParams& p) {
    s.get("mount_height", p.mount_height);
    s.get("layers", p.layers);
    s.get("lowest_deg", p.lowest_deg);
    s.get("fov_deg", p.fov_deg);
    s.get("azimuth_step_deg", p.azimuth_step_deg);
    s.get("range_std", p.range_std);
    s.get("max_range", p.max_range);
    s.get("cluster_cell", p.cluster_cell);
    s.get("merge_margin", p.merge_margin);
    s.get("min_points", p.min_points);
    s.get("detector", p.detector);
    s.get("nominal_length", p.nominal_length);
    s.get("nominal_width", p.nominal_width);
    s.get("nominal_height", p.nominal_height);
    if (p.detector != "box" && p.detector != "centroid")
        throw std::runtime_error("config: scan.detector must be box or centroid");
}

void readSensor(Section s, SensorSimParams& p, bool lidar) {
    s.get("enabled", p.enabled);
    s.get("rate", p.rate);
    s.get("offset", p.offset);
    s.get("detection_probability", p.detection_probability);
    s.get("clutter_rate", p.clutter_rate);
    if (lidar) {
        s.get("model", p.model);
        readScan(s.child("scan"), p.scan);
        if (p.model != "point" && p.model != "scan")
            throw std::runtime_error("config: lidar.model must be point or scan");
    }
    if (p.rate <= 0.0) throw std::runtime_error("config: sensor rate must be positive");
}

VehicleParams readVehicle(const YAML::Node& node, const std::string& path) {
    VehicleParams v;
    Section s(node, path);
    s.get("name", v.name);
    s.get("x", v.x);
    s.get("y", v.y);
    s.get("velocity", v.velocity);
    s.get("angle", v.angle);
    s.get("length", v.length);
    s.get("width", v.width);
    s.get("height", v.height);
    s.get("wheelbase", v.wheelbase);
    // Instructions are [time, acceleration, steering] triples.
    std::vector<std::vector<double>> rows;
    s.get("instructions", rows);
    for (const auto& r : rows) {
        if (r.size() != 3)
            throw std::runtime_error("config: " + path +
                                     ".instructions entries are [time, acceleration, steering]");
        v.instructions.push_back({r[0], r[1], r[2]});
    }
    return v;
}

void readInto(const YAML::Node& root, Config& c) {
    Section top(root, "");
    top.raw("include");  // handled by loadInto

    {
        Section s = top.child("noise");
        s.get("lidar_std", c.noise.lidar_std);
        s.get("radar_range_std", c.noise.radar_range_std);
        s.get("radar_bearing_std", c.noise.radar_bearing_std);
        s.get("radar_rate_std", c.noise.radar_rate_std);
    }
    {
        Section s = top.child("filter");
        auto& f = c.filter;
        s.get("type", f.type);
        s.get("accel_noise", f.accel_noise);
        s.get("yaw_accel_noise", f.yaw_accel_noise);
        {
            Section g = s.child("sigma");
            g.get("scheme", f.sigma.scheme);
            g.get("alpha", f.sigma.alpha);
            g.get("beta", f.sigma.beta);
            g.get("kappa", f.sigma.kappa);
        }
        {
            Section g = s.child("imm");
            const YAML::Node models = g.raw("models");
            if (models.IsDefined() && !models.IsNull()) {
                if (!models.IsSequence())
                    throw std::runtime_error("config: filter.imm.models must be a list");
                f.imm.models.clear();
                for (size_t i = 0; i < models.size(); ++i) {
                    Section m(models[i], "filter.imm.models[" + std::to_string(i) + "]");
                    ImmModel model;
                    m.get("accel_noise", model.accel_noise);
                    m.get("yaw_accel_noise", model.yaw_accel_noise);
                    f.imm.models.push_back(model);
                }
            }
            g.get("switch_rate", f.imm.switch_rate);
            g.get("initial_probabilities", f.imm.initial_probabilities);
        }
        if (f.type != "ukf" && f.type != "ekf" && f.type != "imm" && f.type != "naive_ukf")
            throw std::runtime_error("config: filter.type must be imm, ukf, ekf or naive_ukf");
        if (f.sigma.scheme != "julier" && f.sigma.scheme != "merwe")
            throw std::runtime_error("config: filter.sigma.scheme must be julier or merwe");
        if (f.imm.models.empty() || f.imm.models.size() != f.imm.initial_probabilities.size())
            throw std::runtime_error(
                "config: filter.imm needs one initial probability per model");
    }
    {
        Section s = top.child("tracker");
        auto& t = c.tracker;
        s.get("association", t.association);
        s.get("gate_chi2_lidar", t.gate_chi2_lidar);
        s.get("gate_chi2_radar", t.gate_chi2_radar);
        s.get("management", t.management);
        s.get("confirm_hits", t.confirm_hits);
        s.get("tentative_timeout", t.tentative_timeout);
        s.get("max_coast", t.max_coast);
        {
            Section g = s.child("score");
            g.get("lidar_detection_probability", t.score.lidar_detection_probability);
            g.get("radar_detection_probability", t.score.radar_detection_probability);
            g.get("lidar_clutter_density", t.score.lidar_clutter_density);
            g.get("radar_clutter_density", t.score.radar_clutter_density);
            g.get("radar_rate_span", t.score.radar_rate_span);
            g.get("confirm", t.score.confirm);
            g.get("delete_tentative", t.score.delete_tentative);
            g.get("delete_drop", t.score.delete_drop);
        }
        if (t.management != "score" && t.management != "hits")
            throw std::runtime_error("config: tracker.management must be score or hits");
        Section i = s.child("init");
        i.get("method", t.init.method);
        i.get("max_speed", t.init.max_speed);
        i.get("min_dt", t.init.min_dt);
        i.get("yaw_rate_std", t.init.yaw_rate_std);
        i.get("lateral_speed_std", t.init.lateral_speed_std);
        if (t.association != "gnn" && t.association != "oracle")
            throw std::runtime_error("config: tracker.association must be gnn or oracle");
        if (t.init.method != "two_point" && t.init.method != "single")
            throw std::runtime_error("config: tracker.init.method must be two_point or single");
    }
    {
        Section s = top.child("sim");
        auto& m = c.sim;
        s.get("duration", m.duration);
        s.get("step", m.step);
        s.get("fov", m.fov);
        {
            const YAML::Node ego = s.raw("ego");
            if (ego.IsDefined() && !ego.IsNull()) m.ego = readVehicle(ego, "sim.ego");
        }
        readSensor(s.child("lidar"), m.lidar, true);
        readSensor(s.child("radar"), m.radar, false);
        const YAML::Node vehicles = s.raw("vehicles");
        if (vehicles.IsDefined() && !vehicles.IsNull()) {
            if (!vehicles.IsSequence())
                throw std::runtime_error("config: sim.vehicles must be a list");
            m.vehicles.clear();
            for (size_t i = 0; i < vehicles.size(); ++i)
                m.vehicles.push_back(
                    readVehicle(vehicles[i], "sim.vehicles[" + std::to_string(i) + "]"));
        }
        if (m.step <= 0.0) throw std::runtime_error("config: sim.step must be positive");
    }
    {
        Section s = top.child("eval");
        s.get("rate", c.eval.rate);
        s.get("start", c.eval.start);
        s.get("match_distance", c.eval.match_distance);
        s.get("ospa_cutoff", c.eval.ospa_cutoff);
    }
}

// Applies a file, after the files it includes, on top of `c`.
void loadInto(const std::filesystem::path& path, Config& c, int depth) {
    if (depth > 8) throw std::runtime_error("config: include chain too deep at " + path.string());
    YAML::Node root;
    try {
        root = YAML::LoadFile(path.string());
    } catch (const YAML::Exception& e) {
        throw std::runtime_error("config: cannot read " + path.string() + ": " + e.what());
    }
    if (root.IsNull()) return;
    if (!root.IsMap()) throw std::runtime_error("config: " + path.string() + " must be a map");
    if (const YAML::Node inc = root["include"]) {
        std::vector<std::string> files;
        if (inc.IsSequence())
            files = inc.as<std::vector<std::string>>();
        else
            files.push_back(inc.as<std::string>());
        for (const auto& f : files) loadInto(path.parent_path() / f, c, depth + 1);
    }
    readInto(root, c);
}

}  // namespace

std::vector<VehicleParams> highwayScenario() {
    // Lanes at y = 3.7 (left), 0 (ego lane) and -3.7 (right); the ego vehicle
    // drives at 25 m/s in the centre, with a follower directly behind it
    // (radar bearing ±pi). A lane change is a steering pulse of
    // ±0.0123 rad for 1 s each way, which moves a car at 29 m/s by one lane.
    VehicleParams overtaker{"overtaker", -18.0, 3.7, 29.0, 0.0, 4.6, 1.9, 1.5, 2.8,
                            {{6.5, 0.0, -0.0123}, {7.5, 0.0, 0.0123}, {8.5, -2.0, 0.0}}};
    VehicleParams lead{"lead", 40.0, 0.0, 25.0, 0.0, 4.6, 1.9, 1.5, 2.8,
                       {{2.0, -3.0, 0.0}, {3.5, 3.0, 0.0}, {5.0, 0.0, 0.0}}};
    VehicleParams slow{"slow", 20.0, -3.7, 21.0, 0.0, 4.6, 1.9, 1.5, 2.8,
                       {{3.0, 1.5, 0.0}, {6.0, 0.0, 0.0}}};
    VehicleParams follower{"follower", -15.0, 0.0, 25.5, 0.0, 4.6, 1.9, 1.5, 2.8,
                           {{4.0, -0.5, 0.0}, {6.0, 0.0, 0.0}}};
    return {overtaker, lead, slow, follower};
}

Config defaultConfig() {
    Config c;
    c.sim.vehicles = highwayScenario();
    return c;
}

Config loadConfig(const std::string& path) {
    Config c = defaultConfig();
    loadInto(path, c, 0);
    return c;
}

}  // namespace fusion
