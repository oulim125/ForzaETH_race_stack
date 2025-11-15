#include <frenet_conversion_cpp/frenet_converter_cpp.hpp>

#include <numeric> // optional

// ========================
// frenet_detail::CubicSpline1D
// ========================
namespace frenet_detail {

CubicSpline1D::CubicSpline1D() = default;

CubicSpline1D::CubicSpline1D(const std::vector<double>& x,
                             const std::vector<double>& y) {
    build(x, y);
}

void CubicSpline1D::build(const std::vector<double>& x,
                          const std::vector<double>& y) {
    if (x.size() < 2 || x.size() != y.size())
        throw std::invalid_argument("CubicSpline1D: invalid input sizes");
    for (size_t i = 1; i < x.size(); ++i) {
        if (!(x[i] > x[i-1])) {
            throw std::invalid_argument("CubicSpline1D: x must be strictly increasing");
        }
    }
    n_ = x.size();
    x_ = x;
    a_ = y; // a_i = y_i

    std::vector<double> h(n_-1);
    for (size_t i = 0; i < n_-1; ++i) h[i] = x_[i+1] - x_[i];

    std::vector<double> alpha(n_, 0.0);
    for (size_t i = 1; i < n_-1; ++i) {
        alpha[i] = (3.0/h[i])*(a_[i+1]-a_[i]) - (3.0/h[i-1])*(a_[i]-a_[i-1]);
    }

    std::vector<double> l(n_, 0.0), mu(n_, 0.0), z(n_, 0.0);
    c_.assign(n_, 0.0);
    b_.assign(n_-1, 0.0);
    d_.assign(n_-1, 0.0);

    l[0] = 1.0; z[0] = 0.0; mu[0] = 0.0;
    for (size_t i = 1; i < n_-1; ++i) {
        l[i] = 2.0*(x_[i+1]-x_[i-1]) - h[i-1]*mu[i-1];
        if (std::abs(l[i]) < 1e-14) throw std::runtime_error("CubicSpline1D: singular system");
        mu[i] = h[i]/l[i];
        z[i] = (alpha[i] - h[i-1]*z[i-1]) / l[i];
    }
    l[n_-1] = 1.0; z[n_-1] = 0.0; c_[n_-1] = 0.0;

    for (size_t j = n_-2; j < n_-1; --j) { // j goes n_-2 down to 0
        c_[j] = z[j] - mu[j]*c_[j+1];
        b_[j] = (a_[j+1]-a_[j])/h[j] - h[j]*(c_[j+1] + 2.0*c_[j])/3.0;
        d_[j] = (c_[j+1] - c_[j]) / (3.0*h[j]);
        if (j == 0) break; // size_t underflow guard
    }
}

double CubicSpline1D::operator()(double xq) const {
    size_t i = findInterval(xq);
    double dx = xq - x_[i];
    return a_[i] + b_[i]*dx + c_[i]*dx*dx + d_[i]*dx*dx*dx;
}

double CubicSpline1D::derivative(double xq) const {
    size_t i = findInterval(xq);
    double dx = xq - x_[i];
    return b_[i] + 2.0*c_[i]*dx + 3.0*d_[i]*dx*dx;
}

size_t CubicSpline1D::findInterval(double xq) const {
    if (xq <= x_.front()) return 0;
    if (xq >= x_.back())  return n_-2; // last interval
    // binary search
    size_t lo = 0, hi = n_-1;
    while (hi - lo > 1) {
        size_t mid = (lo + hi) >> 1;
        if (xq < x_[mid]) hi = mid;
        else              lo = mid;
    }
    return lo;
}

double fmod_pos(double a, double m) {
    // positive modulo in [0, m)
    double r = std::fmod(a, m);
    if (r < 0) r += m;
    return r;
}

double clip(double v, double lo, double hi) {
    return std::max(lo, std::min(hi, v));
}

} // namespace frenet_detail


// ========================
// FrenetConverter
// ========================
FrenetConverter::FrenetConverter(const std::vector<double>& waypoints_x,
                                 const std::vector<double>& waypoints_y,
                                 const std::vector<double>& waypoints_psi)
: waypoints_x_(waypoints_x),
  waypoints_y_(waypoints_y),
  waypoints_psi_(waypoints_psi)
{
    if (waypoints_x_.size() < 2 || waypoints_y_.size() != waypoints_x_.size() ||
        waypoints_psi_.size() != waypoints_x_.size()) {
        throw std::invalid_argument("FrenetConverter: invalid waypoint sizes");
    }
    buildRaceline();
}

std::pair<std::vector<double>, std::vector<double>>
FrenetConverter::get_frenet(const std::vector<double>& xs,
                            const std::vector<double>& ys,
                            const std::vector<double>* s0_opt)
{
    if (xs.size() != ys.size()) {
        throw std::invalid_argument("get_frenet: xs and ys must have same size");
    }
    // update closest indices (like Python side effect)
    closest_index_batch_ = get_closest_index(xs, ys);

    std::vector<double> s;
    if (s0_opt == nullptr) {
        s = get_approx_s(xs, ys);
    } else {
        if (s0_opt->size() != xs.size())
            throw std::invalid_argument("get_frenet: s0 size mismatch");
        s = *s0_opt;
    }
    auto sd = get_frenet_coord(xs, ys, s);
    return sd; // {s, d}
}

std::vector<double>
FrenetConverter::get_approx_s(const std::vector<double>& xs,
                              const std::vector<double>& ys) const
{
    if (xs.size() != ys.size())
        throw std::invalid_argument("get_approx_s: xs, ys size mismatch");

    std::vector<double> s(xs.size(), 0.0);
    for (size_t q = 0; q < xs.size(); ++q) {
        int idx = argmin_waypoint_distance(xs[q], ys[q]);
        s[q] = static_cast<double>(idx) * waypoints_distance_m;
    }
    return s;
}

std::vector<int>
FrenetConverter::get_closest_index(const std::vector<double>& xs,
                                   const std::vector<double>& ys) const
{
    if (xs.size() != ys.size())
        throw std::invalid_argument("get_closest_index: xs, ys size mismatch");

    std::vector<int> idxs(xs.size(), 0);
    for (size_t q = 0; q < xs.size(); ++q) {
        idxs[q] = argmin_waypoint_distance(xs[q], ys[q]);
    }
    return idxs;
}

std::pair<std::vector<double>, std::vector<double>>
FrenetConverter::get_frenet_coord(const std::vector<double>& xs,
                                  const std::vector<double>& ys,
                                  std::vector<double> s,
                                  double eps_m)
{
    (void)eps_m; // not used, kept for signature parity
    const size_t N = xs.size();
    if (ys.size() != N || s.size() != N)
        throw std::invalid_argument("get_frenet_coord: input size mismatch");

    // initial projection and d
    std::vector<double> proj, d;
    check_perpendicular(xs, ys, s, proj, d, eps_m);

    for (int it = 0; it < iter_max; ++it) {
        std::vector<double> s_cand(N), proj_cand, d_cand;
        for (size_t i = 0; i < N; ++i) {
            s_cand[i] = frenet_detail::fmod_pos(s[i] + proj[i], raceline_length_);
        }
        check_perpendicular(xs, ys, s_cand, proj_cand, d_cand, eps_m);

        const double clip_abs = waypoints_distance_m / (2.0 * static_cast<double>(iter_max));
        for (size_t i = 0; i < N; ++i) {
            double pc = frenet_detail::clip(proj_cand[i], -clip_abs, +clip_abs);
            // update when |cand| <= |current|
            if (std::abs(pc) <= std::abs(proj[i])) {
                proj[i] = pc;
                s[i]    = s_cand[i];
                d[i]    = d_cand[i];
            }
        }
    }
    return {s, d};
}

void FrenetConverter::get_derivative(const std::vector<double>& s,
                                     std::vector<double>& dx_ds,
                                     std::vector<double>& dy_ds) const
{
    const size_t N = s.size();
    dx_ds.resize(N);
    dy_ds.resize(N);
    for (size_t i = 0; i < N; ++i) {
        double si = wrap_s(s[i]);
        dx_ds[i] = spline_x_.derivative(si);
        dy_ds[i] = spline_y_.derivative(si);
    }
}

std::pair<std::vector<double>, std::vector<double>>
FrenetConverter::get_cartesian(const std::vector<double>& s,
                               const std::vector<double>& d) const
{
    const size_t N = s.size();
    if (d.size() != N) throw std::invalid_argument("get_cartesian: size mismatch");

    std::vector<double> X(N), Y(N);
    for (size_t i = 0; i < N; ++i) {
        double si = wrap_s(s[i]);
        double x  = spline_x_(si);
        double y  = spline_y_(si);
        double dx = spline_x_.derivative(si);
        double dy = spline_y_.derivative(si);
        double psi = std::atan2(dy, dx);
        x += d[i] * std::cos(psi + M_PI*0.5);
        y += d[i] * std::sin(psi + M_PI*0.5);
        X[i] = x; Y[i] = y;
    }
    return {X, Y};
}

int FrenetConverter::get_closest_index(double x, double y) {
    int idx = argmin_waypoint_distance(x, y);
    closest_index_scalar_ = idx; // cache for get_frenet_velocities
    return idx;
}

std::pair<double,double>
FrenetConverter::get_frenet(double x, double y, std::optional<double> s0) {
    get_closest_index(x, y); // match Python side-effect
    std::vector<double> X{ x }, Y{ y };
    std::vector<double> S;
    if (s0.has_value()) S = { s0.value() };
    auto sd = get_frenet(X, Y, s0.has_value() ? &S : nullptr);
    return { sd.first[0], sd.second[0] };
}

std::pair<double,double>
FrenetConverter::get_frenet_velocities(double vx, double vy, double theta) const {
    if (!closest_index_scalar_.has_value())
        throw std::runtime_error("FRENET CONVERTER: closest index is None, call get_closest_index first.");
    int i = *closest_index_scalar_;
    if (i < 0 || static_cast<size_t>(i) >= waypoints_psi_.size())
        throw std::runtime_error("FRENET CONVERTER: cached closest index out of range.");
    double delta_psi = theta - waypoints_psi_[static_cast<size_t>(i)];
    double s_dot =  vx * std::cos(delta_psi) - vy * std::sin(delta_psi);
    double d_dot =  vx * std::sin(delta_psi) + vy * std::cos(delta_psi);
    return {s_dot, d_dot};
}

std::pair<double,double>
FrenetConverter::get_cartesian(double s, double d) const {
    auto XY = get_cartesian(std::vector<double>{s}, std::vector<double>{d});
    return {XY.first[0], XY.second[0]};
}

void FrenetConverter::buildRaceline() {
    // cumulative arc-length waypoints_s_
    waypoints_s_.resize(waypoints_x_.size());
    waypoints_s_[0] = 0.0;
    for (size_t i = 1; i < waypoints_x_.size(); ++i) {
        double dx = waypoints_x_[i] - waypoints_x_[i-1];
        double dy = waypoints_y_[i] - waypoints_y_[i-1];
        waypoints_s_[i] = waypoints_s_[i-1] + std::hypot(dx, dy);
    }
    raceline_length_ = waypoints_s_.back();

    // Build splines x(s), y(s)
    spline_x_.build(waypoints_s_, waypoints_x_);
    spline_y_.build(waypoints_s_, waypoints_y_);
}

double FrenetConverter::wrap_s(double s) const {
    return frenet_detail::fmod_pos(s, raceline_length_);
}

int FrenetConverter::argmin_waypoint_distance(double x, double y) const {
    double best = std::numeric_limits<double>::infinity();
    int best_i = 0;
    for (size_t i = 0; i < waypoints_x_.size(); ++i) {
        double dx = x - waypoints_x_[i];
        double dy = y - waypoints_y_[i];
        double d2 = dx*dx + dy*dy;
        if (d2 < best) { best = d2; best_i = static_cast<int>(i); }
    }
    return best_i;
}

void FrenetConverter::check_perpendicular(const std::vector<double>& xs,
                                          const std::vector<double>& ys,
                                          const std::vector<double>& s,
                                          std::vector<double>& proj_out,
                                          std::vector<double>& d_out,
                                          double /*eps_m*/) const
{
    const size_t N = s.size();
    proj_out.resize(N);
    d_out.resize(N);

    // Tangent components at s: normalized
    std::vector<double> dx_ds, dy_ds;
    get_derivative(s, dx_ds, dy_ds);

    for (size_t i = 0; i < N; ++i) {
        double si = wrap_s(s[i]);

        double tx = dx_ds[i];
        double ty = dy_ds[i];
        double tnorm = std::hypot(tx, ty);
        if (tnorm < 1e-12) {
            throw std::runtime_error("FrenetConverter: zero tangent norm at s");
        }
        tx /= tnorm; ty /= tnorm;

        double x_trk = spline_x_(si);
        double y_trk = spline_y_(si);
        double vx = xs[i] - x_trk;
        double vy = ys[i] - y_trk;

        // projection of point_to_track on tangent
        double proj = tx*vx + ty*vy;

        // perp unit vector (rotated +90deg): [-ty, tx]
        double px = -ty;
        double py =  tx;
        double d   = px*vx + py*vy;

        proj_out[i] = proj;
        d_out[i]    = d;
    }
    // Note: Python returns check_perpendicular=None for speed; we do the same (not used).
}
