#pragma once
#include <vector>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <optional>
#include <utility>
#include <limits>

namespace frenet_detail {

// Natural cubic spline in 1D (declaration only)
class CubicSpline1D {
public:
    CubicSpline1D();
    CubicSpline1D(const std::vector<double>& x, const std::vector<double>& y);

    void build(const std::vector<double>& x, const std::vector<double>& y);

    // Evaluate S(xq)
    double operator()(double xq) const;

    // Evaluate S'(xq)
    double derivative(double xq) const;

private:
    size_t findInterval(double xq) const;

    size_t n_{0};
    std::vector<double> x_, a_, b_, c_, d_;
};

// positive modulo in [0, m)
double fmod_pos(double a, double m);

// clamp helper
double clip(double v, double lo, double hi);

} // namespace frenet_detail


class FrenetConverter {
public:
    // Constructor: inputs correspond to Python version
    FrenetConverter(const std::vector<double>& waypoints_x,
                    const std::vector<double>& waypoints_y,
                    const std::vector<double>& waypoints_psi);

    // ---- Public configuration (match Python defaults) ----
    double waypoints_distance_m = 0.1; // [m]
    int    iter_max = 3;

    // ======= Core API (batch) =======

    // Compute Frenet [s, d] for arrays of points; optional initial s estimates
    std::pair<std::vector<double>, std::vector<double>>
    get_frenet(const std::vector<double>& xs, const std::vector<double>& ys,
               const std::vector<double>* s0_opt = nullptr);

    // Compute approximate s via nearest waypoint index * waypoints_distance_m
    std::vector<double> get_approx_s(const std::vector<double>& xs,
                                     const std::vector<double>& ys) const;

    // Batch: returns indices of the closest waypoint for each query
    std::vector<int> get_closest_index(const std::vector<double>& xs,
                                       const std::vector<double>& ys) const;

    // Refine s,d by perpendicular projection (Python logic, including clipping)
    std::pair<std::vector<double>, std::vector<double>>
    get_frenet_coord(const std::vector<double>& xs,
                     const std::vector<double>& ys,
                     std::vector<double> s,
                     double eps_m = 0.01);

    // Derivative (dx/ds, dy/ds) for each s
    void get_derivative(const std::vector<double>& s,
                        std::vector<double>& dx_ds,
                        std::vector<double>& dy_ds) const;

    // Convert Frenet (s,d) to Cartesian (x,y) for each element
    std::pair<std::vector<double>, std::vector<double>>
    get_cartesian(const std::vector<double>& s, const std::vector<double>& d) const;

    // ======= Convenience single-point API (mirrors Python usage pattern) =======

    // Update & return closest waypoint index for a single query
    int get_closest_index(double x, double y);

    // Single-point get_frenet
    std::pair<double,double> get_frenet(double x, double y, std::optional<double> s0 = std::nullopt);

    // Single-point velocities in Frenet using cached closest index (like Python)
    std::pair<double,double> get_frenet_velocities(double vx, double vy, double theta) const;

    // Single-point Frenet->Cartesian
    std::pair<double,double> get_cartesian(double s, double d) const;

    // Raceline length accessor
    double raceline_length() const { return raceline_length_; }

private:
    // ===== internals =====
    void buildRaceline();

    double wrap_s(double s) const;

    int argmin_waypoint_distance(double x, double y) const;

    // Core of check_perpendicular: returns projection (onto tangent) and lateral d
    void check_perpendicular(const std::vector<double>& xs,
                             const std::vector<double>& ys,
                             const std::vector<double>& s,
                             std::vector<double>& proj_out,
                             std::vector<double>& d_out,
                             double eps_m) const;

private:
    // Inputs
    std::vector<double> waypoints_x_;
    std::vector<double> waypoints_y_;
    std::vector<double> waypoints_psi_;

    // Built state
    std::vector<double> waypoints_s_;
    double raceline_length_{0.0};
    frenet_detail::CubicSpline1D spline_x_, spline_y_;

    // Cached “closest index” like Python (single-point usage)
    std::optional<int> closest_index_scalar_{};
    // Batch cache (not used outside but mirrors Python update)
    std::vector<int> closest_index_batch_;
};
