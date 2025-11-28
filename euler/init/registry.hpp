// Copyright 2025 the samurai team
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <functional>
#include <map>
#include <string>

#include <samurai/box.hpp>

#include "../config.hpp"

namespace test_case
{
    template <class Field>
    using InitFunc = std::function<void(Field&, const typename Field::cell_t&)>;

    template <class Field>
    using BCFunc = std::function<void(Field&, double&)>;

    template <class Field>
    using BoxFunc = std::function<samurai::Box<double, Field::dim>()>;

    template <class Field>
    struct TestCase
    {
        BoxFunc<Field> box;
        InitFunc<Field> init;
        BCFunc<Field> bc;
    };

    template <class Field>
    class TestCaseRegistry
    {
      public:

        static TestCaseRegistry& instance()
        {
            static TestCaseRegistry registry;
            return registry;
        }

        void register_test_case(const std::string& name, BoxFunc<Field> box, InitFunc<Field> init, BCFunc<Field> bc)
        {
            test_cases_[name] = {box, init, bc};
        }

        const TestCase<Field>& get(const std::string& name) const
        {
            auto it = test_cases_.find(name);
            if (it == test_cases_.end())
            {
                throw std::runtime_error("Test case '" + name + "' not found");
            }
            return it->second;
        }

        std::vector<std::string> available_test_cases() const
        {
            std::vector<std::string> names;
            for (const auto& [name, _] : test_cases_)
            {
                names.push_back(name);
            }
            return names;
        }

      private:

        std::map<std::string, TestCase<Field>> test_cases_;
    };

    // Helper pour enregistrer automatiquement un cas test
    template <class Field>
    struct TestCaseRegistrar
    {
        TestCaseRegistrar(const std::string& name, BoxFunc<Field> box, InitFunc<Field> init, BCFunc<Field> bc)
        {
            TestCaseRegistry<Field>::instance().register_test_case(name, box, init, bc);
        }
    };
}

#define REGISTER_TEST_CASE(name, box_fn, init_fn, bc_fn)                                             \
    namespace                                                                                        \
    {                                                                                                \
        static const test_case::TestCaseRegistrar<config<2>::field_t> __registrar_##name##_instance{ \
            #name,                                                                                   \
            []()                                                                                     \
            {                                                                                        \
                return box_fn<config<2>::field_t::dim>();                                            \
            },                                                                                       \
            [](config<2>::field_t& u, const typename config<2>::field_t::cell_t& cell)               \
            {                                                                                        \
                init_fn(u, cell);                                                                    \
            },                                                                                       \
            [](config<2>::field_t& u, double& t)                                                     \
            {                                                                                        \
                bc_fn(u, t);                                                                         \
            }};                                                                                      \
    }
