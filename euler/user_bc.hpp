// Copyright 2025 the samurai team
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <samurai/bc.hpp>

template <class Field>
struct Imposed : public samurai::Bc<Field>
{
    INIT_BC(Imposed, 2)

    apply_function_t get_apply_function(constant_stencil_size_t, const direction_t&) const override
    {
        return [](Field& u, const stencil_cells_t& cells, const value_t& value)
        {
            u[cells[1]] = value;
        };
    }
};