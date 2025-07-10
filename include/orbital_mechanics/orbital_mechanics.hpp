#pragma once

/**
 * @file orbital_mechanics.hpp
 * @brief Main header file for the Orbital Mechanics library
 * 
 * This header provides access to all major components of the orbital mechanics
 * simulation library including core data structures, physics calculations,
 * numerical integrators, and I/O functionality.
 * 
 * @author Arnav Menon
 * @version 1.0.0
 */

// Core components
#include "core/constants.hpp"
#include "core/vector.hpp"
#include "core/body.hpp"
#include "core/body_impl.hpp"

// Physics calculations
#include "physics/three_body.hpp"
#include "physics/three_body_impl.hpp"

// Numerical integrators
#include "integrators/integrator.hpp"
#include "integrators/integrator_impl.hpp"

// I/O and utilities
#include "io/data_writer.hpp"
#include "io/data_writer_impl.hpp"
#include "simulation/simulation.hpp"
#include "simulation/simulation_impl.hpp"

/**
 * @namespace orbital_mechanics
 * @brief Main namespace for the orbital mechanics library
 */
namespace orbital_mechanics {

/**
 * @brief Library version information
 */
constexpr int VERSION_MAJOR = 1;
constexpr int VERSION_MINOR = 0;
constexpr int VERSION_PATCH = 0;

/**
 * @brief Gets the library version as a string
 * @return Version string in format "major.minor.patch"
 */
constexpr const char* version_string() {
    return "1.0.0";
}

}  // namespace orbital_mechanics