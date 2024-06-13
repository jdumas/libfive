/*
libfive: a CAD kernel for modeling with implicit functions
Copyright (C) 2018  Matt Keeter

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this file,
You can obtain one at http://mozilla.org/MPL/2.0/.
*/
#pragma once

#include <Eigen/Eigen>

#include "libfive/render/brep/dc/dc_flags.hpp"
#include "libfive/render/brep/default_new_delete.hpp"

#include "libfive/render/brep/dc/quadric.hpp"

namespace libfive {

template <unsigned N>
struct Intersection {
    Intersection()
    {
        reset();
    }

    void reset() {
        quadric.reset();
    }

    void push(Eigen::Matrix<double, N, 1> pos,
              Eigen::Matrix<double, N, 1> deriv,
              double value)
    {
        quadric += Quadric<double, N>::plane_quadric(pos, deriv, value);
    }

    Quadric<double, N> quadric;

    ALIGNED_OPERATOR_NEW_AND_DELETE(Intersection)
};

}   // namespace libfive
