#include "fusion/models.h"

#include <algorithm>
#include <cmath>

namespace fusion {
namespace {

// Range below which the radar model is evaluated at this range instead, so
// the bearing and range-rate derivatives stay finite.
constexpr double kMinRange = 1e-3;

}  // namespace

const char* sensorName(Sensor sensor) { return sensor == Sensor::Lidar ? "lidar" : "radar"; }

int measurementDim(Sensor sensor) { return sensor == Sensor::Lidar ? 2 : 3; }

State ctrvPredict(const State& x, double dt, double nu_a, double nu_yawdd) {
    const double v = x(2), yaw = x(kYaw), yawd = x(4);
    State out = x;
    if (std::abs(yawd) > kStraightYawRate) {
        out(0) += v / yawd * (std::sin(yaw + yawd * dt) - std::sin(yaw));
        out(1) += v / yawd * (std::cos(yaw) - std::cos(yaw + yawd * dt));
    } else {
        out(0) += v * dt * std::cos(yaw);
        out(1) += v * dt * std::sin(yaw);
    }
    out(kYaw) += yawd * dt;
    const double half_dt2 = 0.5 * dt * dt;
    out(0) += half_dt2 * std::cos(yaw) * nu_a;
    out(1) += half_dt2 * std::sin(yaw) * nu_a;
    out(2) += dt * nu_a;
    out(kYaw) += half_dt2 * nu_yawdd;
    out(4) += dt * nu_yawdd;
    return out;
}

StateCov ctrvJacobian(const State& x, double dt) {
    const double v = x(2), yaw = x(kYaw), w = x(4);
    StateCov F = StateCov::Identity();
    const double s0 = std::sin(yaw), c0 = std::cos(yaw);
    if (std::abs(w) > kStraightYawRate) {
        const double s1 = std::sin(yaw + w * dt), c1 = std::cos(yaw + w * dt);
        F(0, 2) = (s1 - s0) / w;
        F(0, 3) = v / w * (c1 - c0);
        F(0, 4) = v * dt * c1 / w - v / (w * w) * (s1 - s0);
        F(1, 2) = (c0 - c1) / w;
        F(1, 3) = v / w * (s1 - s0);
        F(1, 4) = v * dt * s1 / w - v / (w * w) * (c0 - c1);
    } else {
        // Limits of the expressions above as w → 0.
        F(0, 2) = dt * c0;
        F(0, 3) = -v * dt * s0;
        F(0, 4) = -0.5 * v * dt * dt * s0;
        F(1, 2) = dt * s0;
        F(1, 3) = v * dt * c0;
        F(1, 4) = 0.5 * v * dt * dt * c0;
    }
    F(3, 4) = dt;
    return F;
}

Eigen::Matrix<double, 5, 2> ctrvNoiseJacobian(const State& x, double dt) {
    Eigen::Matrix<double, 5, 2> G = Eigen::Matrix<double, 5, 2>::Zero();
    const double half_dt2 = 0.5 * dt * dt;
    G(0, 0) = half_dt2 * std::cos(x(kYaw));
    G(1, 0) = half_dt2 * std::sin(x(kYaw));
    G(2, 0) = dt;
    G(3, 1) = half_dt2;
    G(4, 1) = dt;
    return G;
}

namespace {

// Rotation from the ego frame to the world frame.
Eigen::Matrix2d egoRotation(const EgoPose& ego) {
    const double c = std::cos(ego.yaw), s = std::sin(ego.yaw);
    Eigen::Matrix2d R;
    R << c, -s, s, c;
    return R;
}

}  // namespace

Eigen::VectorXd measure(Sensor sensor, const State& x, const EgoPose& ego) {
    const Eigen::Matrix2d Rt = egoRotation(ego).transpose();
    const Eigen::Vector2d p = Rt * (x.head<2>() - Eigen::Vector2d(ego.x, ego.y));
    if (sensor == Sensor::Lidar) return p;
    const Eigen::Vector2d v = Rt * (velocityXY(x) - Eigen::Vector2d(ego.vx, ego.vy));
    const double rho = std::max(p.norm(), kMinRange);
    Eigen::VectorXd z(3);
    z << rho, std::atan2(p.y(), p.x()), p.dot(v) / rho;
    return z;
}

Eigen::MatrixXd measurementJacobian(Sensor sensor, const State& x, const EgoPose& ego) {
    const Eigen::Matrix2d Rt = egoRotation(ego).transpose();
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(measurementDim(sensor), kStateDim);
    if (sensor == Sensor::Lidar) {
        H.leftCols<2>() = Rt;
        return H;
    }
    // z depends on the relative position p and velocity u in the ego frame:
    // dz/dx = dz/dp · dp/dx + dz/du · du/dx.
    const Eigen::Vector2d p = Rt * (x.head<2>() - Eigen::Vector2d(ego.x, ego.y));
    const Eigen::Vector2d u = Rt * (velocityXY(x) - Eigen::Vector2d(ego.vx, ego.vy));
    const double rho = std::max(p.norm(), kMinRange), rho2 = rho * rho;
    const double rho_dot = p.dot(u) / rho;
    Eigen::Matrix<double, 3, 2> dz_dp, dz_du;
    dz_dp << p.x() / rho, p.y() / rho,
             -p.y() / rho2, p.x() / rho2,
             u.x() / rho - rho_dot * p.x() / rho2, u.y() / rho - rho_dot * p.y() / rho2;
    dz_du << 0, 0, 0, 0, p.x() / rho, p.y() / rho;
    const double v = x(2), c = std::cos(x(kYaw)), s = std::sin(x(kYaw));
    Eigen::Matrix2d dvel;  // d(vx, vy) / d(v, yaw)
    dvel << c, -v * s, s, v * c;
    H.leftCols<2>() = dz_dp * Rt;
    H.middleCols<2>(2) = dz_du * Rt * dvel;
    return H;
}

Eigen::MatrixXd measurementNoise(Sensor sensor, const NoiseParams& noise) {
    if (sensor == Sensor::Lidar)
        return Eigen::Vector2d(noise.lidar_std * noise.lidar_std,
                               noise.lidar_std * noise.lidar_std)
            .asDiagonal();
    return Eigen::Vector3d(noise.radar_range_std * noise.radar_range_std,
                           noise.radar_bearing_std * noise.radar_bearing_std,
                           noise.radar_rate_std * noise.radar_rate_std)
        .asDiagonal();
}

Eigen::VectorXd residual(Sensor sensor, const Eigen::VectorXd& a, const Eigen::VectorXd& b) {
    Eigen::VectorXd d = a - b;
    if (sensor == Sensor::Radar) d(1) = wrapAngle(d(1));
    return d;
}

Eigen::Vector2d detectionPosition(const Measurement& m) {
    const Eigen::Vector2d local = m.sensor == Sensor::Lidar
                                      ? Eigen::Vector2d(m.z.head<2>())
                                      : Eigen::Vector2d(m.z(0) * std::cos(m.z(1)),
                                                        m.z(0) * std::sin(m.z(1)));
    return Eigen::Vector2d(m.ego.x, m.ego.y) + egoRotation(m.ego) * local;
}

Eigen::Matrix2d detectionPositionCov(const Measurement& m, const NoiseParams& noise) {
    if (m.sensor == Sensor::Lidar)
        return Eigen::Matrix2d::Identity() * noise.lidar_std * noise.lidar_std;
    const double rho = m.z(0), c = std::cos(m.z(1)), s = std::sin(m.z(1));
    Eigen::Matrix2d J;
    J << c, -rho * s, s, rho * c;
    const Eigen::Vector2d var(noise.radar_range_std * noise.radar_range_std,
                              noise.radar_bearing_std * noise.radar_bearing_std);
    const Eigen::Matrix2d R = egoRotation(m.ego);
    return R * J * var.asDiagonal() * J.transpose() * R.transpose();
}

}  // namespace fusion
