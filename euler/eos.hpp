// Copyright 2025 the samurai team
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

namespace EOS
{
    struct stiffened_gas
    {
        static constexpr double gamma  = 1.4;
        static constexpr double pi_inf = 0.;
        static constexpr double q_inf  = 0.;

        static auto p(const auto& rho, const auto& e)
        {
            return (gamma - 1.0) * rho * (e - q_inf) - gamma * pi_inf;
        }

        static auto c(const auto& rho, const auto& p)
        {
            return std::sqrt(gamma * (p + pi_inf) / rho);
        }

        static auto e(const auto& rho, const auto& p)
        {
            return (p + gamma * pi_inf) / ((gamma - 1.0) * rho) + q_inf;
        }
    };
} // namespace EOS