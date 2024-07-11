#pragma once

#include "libfive/render/brep/dc/probabilistic_quadrics.hpp"
#include "libfive/render/brep/dc/dc_flags.hpp"

#include <Eigen/Dense>

#include <functional>
#include <array>
#include <iostream>

namespace libfive {

struct Tracker {

    static Tracker & instance() {
        static Tracker instance;
        return instance;
    }

    std::function<void(std::array<double, 9>, std::array<double, 3>, std::array<double, 3>)> add_quadric;
    std::function<void(std::array<double, 3>, std::array<double, 3>)> add_point_and_normal;

    std::function<void(Eigen::AlignedBox<float, 3>)> add_cell;

    double sigma_n = 1e-3;
    bool use_probabilistic_quadrics = false;

};

template <typename Scalar, int N>
bool stable_normalize(Eigen::Matrix<Scalar, N, 1>& x, double &norm) {
    Scalar w = x.cwiseAbs().maxCoeff();
    if (w == Scalar(0)) {
        norm = 0;
        return false;
    }
    Scalar z = (x / w).squaredNorm();
    if (z > Scalar(0)) {
        x /= std::sqrt(z) * w;
        norm = std::sqrt(z) * w;
        return true;
    }
    norm = 0;
    return false;
}

template<typename Scalar, unsigned N>
struct Quadric;

template <typename Scalar>
struct Quadric<Scalar, 3>
{
    using QuadricT = pq::quadric<
          pq::math<Scalar,
                   Eigen::Matrix<Scalar, 3, 1>,
                   Eigen::Matrix<Scalar, 3, 1>,
                   Eigen::Matrix<Scalar, 3, 3>>>;

    static Quadric plane_quadric(
        Eigen::Matrix<Scalar, 3, 1> pos,
        Eigen::Matrix<Scalar, 3, 1> deriv,
        Scalar value)
    {
        Quadric fallback;

#if !LIBFIVE_UNNORMALIZED_DERIVS
        // If the point has an invalid normal, then skip it
        if (!deriv.array().isFinite().all()) {
            return fallback;
        }
        // Find normalized derivatives and distance value
        double norm = 0;
        if (!stable_normalize(deriv, norm)) {
            return fallback;
        }
        value /= norm;
#endif

        // fallback.q = QuadricT::point_quadric(pos);
        // fallback.q.c += value;
        // return fallback;

        Quadric ret;

        // double sigma_n = 1e-2; // + 1e3 * std::abs(value);
        // std::cout << pos.transpose() << std::endl;
        double sigma_n = Tracker::instance().sigma_n;

        // ret.q = QuadricT::point_quadric(pos);
        ret.q = QuadricT::probabilistic_plane_quadric(
                         pos,
                         deriv,
                         0,
                         sigma_n,
                         value);

        return ret;
    }

    Eigen::Matrix<Scalar, 3, 1> minimizer(Eigen::Matrix<Scalar, 3, 1> center) const {
        constexpr int N = 3;
        typedef Eigen::Matrix<double, N, 1> Vec;

        Eigen::LDLT<Eigen::Matrix3<Scalar>> solver;
        // // Eigen::FullPivLU<Eigen::Matrix3<Scalar>> solver;
        // // Eigen::JacobiSVD<Eigen::Matrix3<Scalar>, Eigen::ComputeThinU | Eigen::ComputeThinV> solver;
        solver.compute(q.A());
        return solver.solve(q.b() - q.A() * center) + center;
        // return center;
        // return q.minimizer();

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, N, N>> es(q.A());

        // We need to find the pseudo-inverse of AtA.
        auto eigenvalues = es.eigenvalues().real();

        // Truncate near-singular eigenvalues in the SVD's diagonal matrix
        Eigen::Matrix<double, N, N> D = Eigen::Matrix<double, N, N>::Zero();

        // Pick a cutoff depending on whether the derivatives were normalized
        // before loading them into the AtA matrix
    #if LIBFIVE_UNNORMALIZED_DERIVS
        auto highest_val = eigenvalues.template lpNorm<Eigen::Infinity>();
        // We scale EIGENVALUE_CUTOFF to highestVal.  Additionally, we need to
        // use a significantly lower cutoff threshold (here set to the square
        // of the normalized-derivatives threshold), since when derivatives
        // are not normalized a cutoff of .1 can cause one feature to be
        // entirely ignored if its derivative is fairly small in comparison to
        // another feature.  (The same can happen with any cutoff, but it is
        // much less likely this way, and it should still be high enough to
        // avoid wild vertices due to noisy normals under most circumstances,
        // at least enough that they will be a small minority of the situations
        // in which dual contouring's need to allow out-of-box vertices causes
        // issues.)

        // If highestVal is extremely small, it's almost certainly due to noise or
        // is 0; in the former case, scaling our cutoff to it will still result in
        // a garbage result, and in the latter it'll produce a diagonal matrix full
        // of infinities.  So we instead use an infinite cutoff to force D to be
        // set to zero, resulting in the mass point being used as our vertex, which
        // is the best we can do without good gradients.
        constexpr double EIGENVALUE_CUTOFF_2 = EIGENVALUE_CUTOFF * EIGENVALUE_CUTOFF;
        const double cutoff = (highest_val > 1e-20)
            ? highest_val * EIGENVALUE_CUTOFF_2
            : std::numeric_limits<double>::infinity();
    #else
        const double cutoff = EIGENVALUE_CUTOFF;
    #endif

        for (unsigned i = 0; i < N; ++i) {
            D.diagonal()[i] = (fabs(eigenvalues[i]) < cutoff)
                ? 0 : (1 / eigenvalues[i]);
        }

        // SVD matrices
        auto U = es.eigenvectors().real().eval(); // = V

        // Pseudo-inverse of A
        auto AtAp = (U * D * U.transpose()).eval();

        // Solve for vertex position (minimizing distance to center)
        Vec v = AtAp * (q.b() - (q.A() * center)) + center;

        return v;
    }

    Scalar eval(Eigen::Matrix<Scalar, 3, 1> x) const {
        return q(x);
    }

    void reset() {
        q = QuadricT();
    }

    Quadric& operator+=(const Quadric<Scalar, 3>& rhs)
    {
        q += rhs.q;
        return *this;
    }

    QuadricT q;

    std::array<double, 9> coeffs() const {
        std::array<double, 9> x;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                x[i * 3 + j] = q.A()(i, j);
            }
        }
        return x;
    }
};

template <typename Scalar, unsigned N>
struct Quadric
{
    static Quadric plane_quadric(
        Eigen::Matrix<Scalar, N, 1> pos,
        Eigen::Matrix<Scalar, N, 1> deriv,
        Scalar value)
    {
#if !LIBFIVE_UNNORMALIZED_DERIVS
        // If the point has an invalid normal, then skip it
        if (!deriv.array().isFinite().all()) {
            return {};
        }
        // Find normalized derivatives and distance value
        double norm = 0;
        if (!stable_normalize(deriv, norm)) {
            return {};
        }
        value /= norm;
#endif
        double d = deriv.dot(pos);
        Quadric ret;
        ret.A = deriv * deriv.transpose();
        ret.b = deriv * d;
        ret.c = d * d + value;
        return ret;
    }

    Eigen::Matrix<Scalar, N, 1> minimizer(Eigen::Matrix<Scalar, N, 1> center) const {
        // llt may be be faster but less numerically robust
        Eigen::LDLT<Eigen::Matrix<Scalar, N, N>> ldlt;
        ldlt.compute(A);
        return ldlt.solve(b);
    }

    Scalar eval(Eigen::Matrix<Scalar, N, 1> x) const {
        assert(x.cols() == 1);
        return x.transpose() * A * x - Scalar(2) * b.dot(x) + c;
    }

    void reset() {
        A.setZero();
        b.setZero();
        c = 0;
    }

    Quadric& operator+=(const Quadric<Scalar, N>& rhs)
    {
        A += rhs.A;
        b += rhs.b;
        c += rhs.c;
        return *this;
    }

    Eigen::Matrix<Scalar, N, N> A = Eigen::Matrix<Scalar, N, N>::Zero();
    Eigen::Matrix<Scalar, N, 1> b = Eigen::Matrix<Scalar, N, 1>::Zero();
    Scalar c = Scalar(0);
};

}
