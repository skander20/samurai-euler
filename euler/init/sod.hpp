// Copyright 2025 the samurai team
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <numbers>

#include <samurai/bc.hpp>

#include "../variables.hpp"
#include "registry.hpp"

namespace test_case::sod
{
    double theta = std::numbers::pi / 4.;
    double x0    = 0.5;
    double y0    = 0.5;

    PrimState<2> left_state{
        1.,
        1.,
        xt::xtensor_fixed<double, xt::xshape<2>>{0., 0.}
    };

    PrimState<2> right_state{
        0.125,
        0.1,
        xt::xtensor_fixed<double, xt::xshape<2>>{0., 0.}
    };

    auto init_fn = [](auto& u, auto& cell)
    {
        auto x = cell.center();

        const double x_theta = std::tan(theta) * (x[0] - x0);
        const double y_theta = x[1] - y0;

        if (x_theta < y_theta)
        {
            u[cell] = prim2cons<2>(left_state);
        }
        else
        {
            u[cell] = prim2cons<2>(right_state);
        }
    };

    void bc_fn(auto& u, double /*t*/)
    {
        samurai::make_bc<samurai::Neumann<1>>(u, 0., 0., 0., 0.);
    }

    template <std::size_t dim>
    auto box_fn()
    {
        xt::xtensor_fixed<double, xt::xshape<dim>> min_corner = {0., 0.};
        xt::xtensor_fixed<double, xt::xshape<dim>> max_corner = {1., 1.};

        return samurai::Box<double, dim>(min_corner, max_corner);
    }

}

REGISTER_TEST_CASE(sod, test_case::sod::box_fn, test_case::sod::init_fn, test_case::sod::bc_fn)
