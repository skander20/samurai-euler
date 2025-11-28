// Copyright 2025 the samurai team
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <samurai/bc.hpp>
#include <samurai/box.hpp>

#include "../eos.hpp"
#include "../variables.hpp"
#include "registry.hpp"

namespace test_case::sedov_blast
{

    double rho_ambient = 1.0;                                  // Ambient density
    double p_ambient   = 1e-5;                                 // Ambient pressure (very small)
    double E_blast     = 0.244816;                             // Blast energy
    double r_blast     = 0.1;                                  // Blast radius
    double V_blast     = std::numbers::pi * r_blast * r_blast; // 2D: area of the disk

    xt::xtensor_fixed<double, xt::xshape<2>> center{0, 0};

    auto init_fn = [](auto& u, auto& cell)
    {
        auto x = cell.center();

        double dx = x[0] - center[0];
        double dy = x[1] - center[1];
        double r  = std::sqrt(dx * dx + dy * dy);

        double rho = rho_ambient;
        double p;
        double vx = 0.;
        double vy = 0.;

        if (r < r_blast)
        {
            // Blast zone: concentrated energy
            p = (EOS::stiffened_gas::gamma - 1.0) * E_blast / V_blast;
        }
        else
        {
            // Ambient zone
            p = p_ambient;
        }

        // Variables conservatives
        using EulerConsVar            = EulerLayout<2>;
        u[cell][EulerConsVar::rho]    = rho;
        u[cell][EulerConsVar::rhoE]   = rho * (EOS::stiffened_gas::e(rho, p) + 0.5 * (vx * vx + vy * vy));
        u[cell][EulerConsVar::mom(0)] = rho * vx;
        u[cell][EulerConsVar::mom(1)] = rho * vy;
    };

    void bc_fn(auto& u, double /*t*/)
    {
        samurai::make_bc<samurai::Neumann<1>>(u, 0., 0., 0., 0.);
    }

    template <std::size_t dim>
    auto box_fn()
    {
        xt::xtensor_fixed<double, xt::xshape<dim>> min_corner = {-1., -1.};
        xt::xtensor_fixed<double, xt::xshape<dim>> max_corner = {1., 1.};

        return samurai::Box<double, dim>(min_corner, max_corner);
    }

}

REGISTER_TEST_CASE(sedov_blast, test_case::sedov_blast::box_fn, test_case::sedov_blast::init_fn, test_case::sedov_blast::bc_fn)