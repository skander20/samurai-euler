// Copyright 2025 the samurai team
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include "schemes/hll.hpp"
#include "schemes/hllc.hpp"
#include "schemes/rusanov.hpp"

template <class Field>
auto get_fv_scheme(const std::string& scheme)
{
    if (scheme == "rusanov")
    {
        return make_euler_rusanov<Field>();
    }
    else if (scheme == "hll")
    {
        return make_euler_hll<Field>();
    }
    else if (scheme == "hllc")
    {
        return make_euler_hllc<Field>();
    }
    else
    {
        throw std::runtime_error("Unknown scheme: " + scheme);
    }
}