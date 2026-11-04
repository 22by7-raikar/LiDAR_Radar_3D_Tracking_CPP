/// @file
/// @brief All tunable parameters and the YAML loader.
///
/// The defaults are the recommended configuration on the built-in highway
/// scenario. `config/default.yaml` lists every field with the same names;
/// `config/naive.yaml` (a conventional single-UKF tracker, the comparison
/// point) and `config/realistic.yaml` (hard sensing) are the other reference
/// configurations used in `docs/DESIGN.md`.
#pragma once

#include <array>
#include <string>
#include <vector>

namespace fusion {

/// @brief Measurement noise standard deviations, used by the simulator and the filters.
struct NoiseParams {
    double lidar_std = 0.15;          ///< Lidar position noise per axis [m].
    double radar_range_std = 0.3;     ///< Radar range noise [m].
    double radar_bearing_std = 0.03;  ///< Radar bearing noise [rad].
    double radar_rate_std = 0.3;      ///< Radar range-rate noise [m/s].
};

/// @brief Sigma-point placement of the unscented transform.
struct SigmaPointParams {
    /// `julier`: lambda = 3 - n; the centre weight is negative for n > 3. `merwe`: scaled points with
    /// lambda = alpha²(n + kappa) - n and a separate covariance weight
    /// for the centre point.
    std::string scheme = "merwe";
    double alpha = 1.0;  ///< Spread of the scaled points (`merwe` only).
    double beta = 2.0;   ///< Prior-knowledge term; 2 is optimal for Gaussians (`merwe` only).
    double kappa = 0.0;  ///< Secondary scaling (`merwe` only).
};

/// @brief One motion model of the IMM filter: CTRV with its own process noise.
struct ImmModel {
    double accel_noise = 0.5;      ///< Longitudinal acceleration noise [m/s²].
    double yaw_accel_noise = 0.1;  ///< Yaw acceleration noise [rad/s²].
};

/// @brief Interacting multiple model parameters.
struct ImmParams {
    /// Models; the defaults are a cruising and a manoeuvring CTRV model.
    std::vector<ImmModel> models{{0.3, 0.02}, {3.0, 0.5}};
    /// Rate at which the target switches from one model to another [1/s]. The
    /// transition matrix for a step dt is exp(dt·Λ), so a zero-length step
    /// (two sensors at one time stamp) mixes nothing.
    double switch_rate = 0.1;
    std::vector<double> initial_probabilities{0.8, 0.2};  ///< Mode probabilities of a new track.
};

/// @brief Filter used by each track.
struct FilterParams {
    /// `imm` (IMM of UKFs), `ukf`, `ekf`, or `naive_ukf` (a textbook UKF
    /// without the robustness measures of `ukf`, kept for comparison).
    std::string type = "imm";
    double accel_noise = 3.0;      ///< Longitudinal acceleration noise [m/s²] (`ukf`, `ekf`, `naive_ukf`).
    double yaw_accel_noise = 0.5;  ///< Yaw acceleration noise [rad/s²] (`ukf`, `ekf`, `naive_ukf`).
    SigmaPointParams sigma;        ///< Unscented transform (`ukf`, `imm`).
    ImmParams imm;                 ///< Used when #type is `imm`.
};

/// @brief Track initialisation.
struct InitParams {
    /// `two_point`: wait for a second detection and start with position,
    /// speed and heading from the two, with matching covariance.
    /// `single`: start from the first detection, assuming the object moves
    /// like the ego vehicle (its speed and heading), with unit variances on
    /// speed, heading and turn rate.
    std::string method = "two_point";
    double max_speed = 40.0;      ///< Two detections further apart than max_speed·dt + gate are not paired [m/s].
    double min_dt = 0.2;          ///< Second detections closer in time than this are absorbed, not paired [s].
    double yaw_rate_std = 0.3;    ///< Initial yaw-rate standard deviation [rad/s].
    /// Prior on the lateral speed at initialisation (across the ego vehicle's
    /// heading): traffic moves along the road, so v_lat ~ N(0, σ²) is fused
    /// with the two-point velocity. 0 disables the prior [m/s].
    double lateral_speed_std = 1.0;
};

/// @brief Log-likelihood-ratio track score (Blackman and Popoli).
///
/// A detection associated with a track adds ln(Pd·N(y; 0, S) / β), where β
/// is the clutter density in measurement space; a sensor frame in which a
/// track inside the field of view gets nothing adds ln(1 − Pd).
struct ScoreParams {
    double lidar_detection_probability = 0.9;  ///< Pd assumed for lidar.
    double radar_detection_probability = 0.9;  ///< Pd assumed for radar.
    double lidar_clutter_density = 1e-3;       ///< False lidar detections per m² per frame.
    double radar_clutter_density = 3e-3;       ///< False radar detections per m² per frame.
    double radar_rate_span = 30.0;             ///< Spread of clutter range rates [m/s].
    double confirm = 10.0;                     ///< Score at which a track is confirmed.
    double delete_tentative = -5.0;            ///< Unconfirmed tracks below this are dropped.
    double delete_drop = 15.0;                 ///< Confirmed tracks this far below their best score are dropped.
};

/// @brief Multi-target tracker.
struct TrackerParams {
    /// `gnn`: global nearest neighbour with gating and the Hungarian algorithm.
    /// `oracle`: the simulator's labels decide which track a detection belongs
    /// to and clutter is dropped (an upper bound for the tracker).
    std::string association = "gnn";
    double gate_chi2_lidar = 13.82;  ///< Gate on the normalised innovation, 99.9 % for 2 DoF.
    double gate_chi2_radar = 16.27;  ///< Gate on the normalised innovation, 99.9 % for 3 DoF.
    /// `score`: sequential log-likelihood-ratio track score (confirm and
    /// delete on thresholds, see ScoreParams). `hits`: confirm after
    /// #confirm_hits detections, delete only on timeouts.
    std::string management = "score";
    int confirm_hits = 4;            ///< `hits`: detections needed before a track is reported.
    double tentative_timeout = 0.25; ///< Unconfirmed tracks without an update for this long are dropped [s].
    double max_coast = 1.0;          ///< Confirmed tracks without an update for this long are dropped [s].
    ScoreParams score;               ///< Used when #management is `score`.
    InitParams init;                 ///< Track initialisation.
};

/// @brief LiDAR ray-casting model (`lidar.model: scan`).
struct ScanParams {
    double mount_height = 1.9;      ///< Sensor height above the road (roof mount) [m].
    int layers = 64;                ///< Number of lasers.
    double lowest_deg = -24.8;      ///< Elevation of the lowest laser [deg].
    double fov_deg = 26.8;          ///< Vertical field of view [deg].
    double azimuth_step_deg = 0.1;  ///< Horizontal resolution [deg].
    double range_std = 0.02;        ///< Range noise per return [m].
    double max_range = 120.0;       ///< Returns beyond this are dropped [m].
    double cluster_cell = 0.6;      ///< Grid cell for connected-component clustering [m].
    /// Clusters are merged while their union fits in the nominal vehicle
    /// size plus this margin; negative disables merging [m].
    double merge_margin = 0.3;
    int min_points = 5;             ///< Smaller clusters are dropped.
    /// `box`: L-shape rectangle fit, extended to the nominal vehicle size away
    /// from the sensor. `centroid`: mean of the returns.
    std::string detector = "box";
    double nominal_length = 4.6;    ///< Vehicle length used to complete partly seen boxes [m].
    double nominal_width = 1.9;     ///< Vehicle width used to complete partly seen boxes [m].
    double nominal_height = 1.5;    ///< Vehicle roof height, to recognise cabin-only views [m].
};

/// @brief One simulated sensor.
struct SensorSimParams {
    bool enabled = true;               ///< Off removes the sensor from the run.
    double rate = 30.0;                ///< Measurement rate [Hz].
    double offset = 0.0;               ///< Time of the first measurement [s].
    double detection_probability = 1.0;  ///< Probability that a vehicle is detected in a frame.
    double clutter_rate = 0.0;         ///< Mean false detections per frame (Poisson).
    std::string model = "point";       ///< Lidar only: `point` (noisy centre) or `scan` (ray casting).
    ScanParams scan;                   ///< Lidar only, used when #model is `scan`.
};

/// @brief One scripted change of a vehicle's controls.
struct Instruction {
    double time = 0.0;          ///< When it takes effect [s].
    double acceleration = 0.0;  ///< New longitudinal acceleration [m/s²].
    double steering = 0.0;      ///< New steering angle [rad].
};

/// @brief A simulated vehicle in the world frame (x along the road, y to the left).
struct VehicleParams {
    std::string name;                 ///< Label used in outputs.
    double x = 0.0;                   ///< Initial position x [m].
    double y = 0.0;                   ///< Initial position y [m].
    double velocity = 25.0;           ///< Initial speed [m/s].
    double angle = 0.0;               ///< Initial heading [rad].
    double length = 4.6;              ///< Body length [m].
    double width = 1.9;               ///< Body width [m].
    double height = 1.5;              ///< Roof height [m].
    double wheelbase = 2.8;           ///< For the kinematic bicycle model [m].
    std::vector<Instruction> instructions;  ///< Control changes, in time order.
};

/// @brief Simulation set-up.
struct SimParams {
    double duration = 10.0;      ///< Simulated time [s].
    double step = 1.0 / 30.0;    ///< Integration step; sensors fire on these ticks [s].
    SensorSimParams lidar;       ///< Lidar.
    SensorSimParams radar;       ///< Radar.
    /// Field of view, used for clutter and for which vehicles are visible:
    /// x_min, x_max, y_min, y_max in the ego frame [m].
    std::array<double, 4> fov{-30.0, 70.0, -9.0, 9.0};
    VehicleParams ego{"ego", 0.0, 0.0, 25.0, 0.0, 4.6, 1.9, 1.5, 2.8, {}};  ///< The vehicle carrying the sensors.
    std::vector<VehicleParams> vehicles;  ///< Traffic; defaults to highwayScenario().
};

/// @brief Evaluation settings.
struct EvalParams {
    double rate = 30.0;                 ///< Estimates are scored on this grid [Hz].
    double start = 1.0;                 ///< Errors are accumulated from this time on, after start-up [s].
    double match_distance = 3.0;        ///< Tracks further from every vehicle are false tracks [m].
    double ospa_cutoff = 5.0;           ///< OSPA cut-off c [m] (order p = 2).
};

/// @brief Everything.
struct Config {
    NoiseParams noise;      ///< Sensor noise.
    FilterParams filter;    ///< Per-track filter.
    TrackerParams tracker;  ///< Association and track management.
    SimParams sim;          ///< Simulation.
    EvalParams eval;        ///< Evaluation.
};

/// @brief Built-in highway scenario: three lanes, an overtaking car that cuts
/// in ahead of the ego vehicle, a lead car that brakes and recovers, a slower
/// car on the right that speeds up, and a car following directly behind.
std::vector<VehicleParams> highwayScenario();

/// @brief Built-in defaults, with highwayScenario() as traffic.
Config defaultConfig();

/// @brief Loads a YAML file on top of defaultConfig().
///
/// Keys that are not given keep their defaults; unknown keys and badly typed
/// values throw std::runtime_error, so a typo cannot be silently ignored.
/// A file may start from other files with `include: other.yaml` or
/// `include: [a.yaml, b.yaml]` (paths relative to the including file); they
/// are applied in order, then the file's own keys.
/// @param path YAML file.
Config loadConfig(const std::string& path);

}  // namespace fusion
