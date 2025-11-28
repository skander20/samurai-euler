// Copyright 2025 the samurai team
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <numbers>

#include <samurai/bc.hpp>

#include "../variables.hpp"
#include "registry.hpp"

namespace test_case::riemann_2d_config_3
{
    double x0 = 0.5;
    double y0 = 0.5;

    PrimState<2> quad1_state{
        1.5,
        1.5,
        xt::xtensor_fixed<double, xt::xshape<2>>{0., 0.}
    };

    PrimState<2> quad2_state{
        0.5323,
        0.3,
        xt::xtensor_fixed<double, xt::xshape<2>>{1.206, 0.}
    };

    PrimState<2> quad3_state{
        0.138,
        0.29,
        xt::xtensor_fixed<double, xt::xshape<2>>{1.206, 1.206}
    };

    PrimState<2> quad4_state{
        0.5323,
        0.3,
        xt::xtensor_fixed<double, xt::xshape<2>>{0, 1.206}
    };

    auto init_fn = [](auto& u, auto& cell)
    {
        auto x = cell.center();

        if (x[0] >= x0 && x[1] >= y0)
        {
            u[cell] = prim2cons<2>(quad1_state);
        }
        else if (x[0] < x0 && x[1] >= y0)
        {
            u[cell] = prim2cons<2>(quad2_state);
        }
        else if (x[0] < x0 && x[1] < y0)
        {
            u[cell] = prim2cons<2>(quad3_state);
        }
        else // (x[0] >= x0 && x[1] < y0)
        {
            u[cell] = prim2cons<2>(quad4_state);
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

REGISTER_TEST_CASE(riemann2d_config3,
                   test_case::riemann_2d_config_3::box_fn,
                   test_case::riemann_2d_config_3::init_fn,
                   test_case::riemann_2d_config_3::bc_fn)

namespace test_case::riemann_2d_config_4
{
    double x0 = 0.5;
    double y0 = 0.5;

    PrimState<2> quad1_state{
        1.1,
        1.1,
        xt::xtensor_fixed<double, xt::xshape<2>>{0., 0.}
    };

    PrimState<2> quad2_state{
        0.5065,
        0.35,
        xt::xtensor_fixed<double, xt::xshape<2>>{0.8939, 0.}
    };

    PrimState<2> quad3_state{
        1.1,
        1.1,
        xt::xtensor_fixed<double, xt::xshape<2>>{0.8939, 0.89396}
    };

    PrimState<2> quad4_state{
        0.5065,
        0.35,
        xt::xtensor_fixed<double, xt::xshape<2>>{0, 0.89396}
    };

    auto init_fn = [](auto& u, auto& cell)
    {
        auto x = cell.center();

        if (x[0] >= x0 && x[1] >= y0)
        {
            u[cell] = prim2cons<2>(quad1_state);
        }
        else if (x[0] < x0 && x[1] >= y0)
        {
            u[cell] = prim2cons<2>(quad2_state);
        }
        else if (x[0] < x0 && x[1] < y0)
        {
            u[cell] = prim2cons<2>(quad3_state);
        }
        else // (x[0] >= x0 && x[1] < y0)
        {
            u[cell] = prim2cons<2>(quad4_state);
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

REGISTER_TEST_CASE(riemann2d_config4,
                   test_case::riemann_2d_config_4::box_fn,
                   test_case::riemann_2d_config_4::init_fn,
                   test_case::riemann_2d_config_4::bc_fn)

namespace test_case::riemann_2d_config_12
{
    double x0 = 0.5;
    double y0 = 0.5;

    PrimState<2> quad1_state{
        0.5197,
        0.4,
        xt::xtensor_fixed<double, xt::xshape<2>>{0., 0.}
    };

    PrimState<2> quad2_state{
        1,
        1,
        xt::xtensor_fixed<double, xt::xshape<2>>{-0.6259, 0.}
    };

    PrimState<2> quad3_state{
        0.8,
        1,
        xt::xtensor_fixed<double, xt::xshape<2>>{-0.6259, -0.6259}
    };

    PrimState<2> quad4_state{
        1,
        1,
        xt::xtensor_fixed<double, xt::xshape<2>>{0, -0.6259}
    };

    auto init_fn = [](auto& u, auto& cell)
    {
        auto x = cell.center();

        if (x[0] >= x0 && x[1] >= y0)
        {
            u[cell] = prim2cons<2>(quad1_state);
        }
        else if (x[0] < x0 && x[1] >= y0)
        {
            u[cell] = prim2cons<2>(quad2_state);
        }
        else if (x[0] < x0 && x[1] < y0)
        {
            u[cell] = prim2cons<2>(quad3_state);
        }
        else // (x[0] >= x0 && x[1] < y0)
        {
            u[cell] = prim2cons<2>(quad4_state);
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

REGISTER_TEST_CASE(riemann2d_config12,
                   test_case::riemann_2d_config_12::box_fn,
                   test_case::riemann_2d_config_12::init_fn,
                   test_case::riemann_2d_config_12::bc_fn)
