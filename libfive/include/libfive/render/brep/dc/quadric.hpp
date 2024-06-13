#pragma once

#include "libfive/render/brep/dc/probabilistic_quadrics.hpp"
#include "libfive/render/brep/dc/dc_flags.hpp"

#include <Eigen/Dense>

#include <iostream>

namespace libfive {

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

        // double sigma_n = 1e-2 + 1e3 * std::abs(value);
        // std::cout << sigma_n << std::endl;
        double sigma_n = 1e-2;

        // ret.q = QuadricT::point_quadric(pos);
        ret.q = QuadricT::probabilistic_plane_quadric(
                         pos,
                         deriv,
                         0,
                         sigma_n);
        ret.q.c += value;

        return ret;
    }

    auto minimizer() const { return q.minimizer(); }

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

    auto minimizer() const { return ldlt.solve(b); }

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

    // llt may be be faster but less numerically robust
    Eigen::LDLT<Eigen::Matrix<Scalar, N, N>> ldlt;
};

}
