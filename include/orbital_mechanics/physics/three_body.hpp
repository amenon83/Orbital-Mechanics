#pragma once

#include "../core/vector.hpp"
#include "../core/body.hpp"
#include "../core/constants.hpp"
#include "../integrators/integrator.hpp"
#include <memory>
#include <vector>
#include <cmath>

namespace orbital_mechanics::physics {

/**
 * @brief Parameters for the Circular Restricted Three-Body Problem
 * 
 * Contains the configuration for a CR3BP simulation including
 * the two primary bodies and system parameters.
 */
struct CR3BPParameters {
    double primary1_mass;    // Mass of the first primary body (kg)
    double primary2_mass;    // Mass of the second primary body (kg)
    double separation;       // Distance between primaries (m)
    double angular_velocity; // Angular velocity of the rotating frame (rad/s)
    
    // Derived quantities
    double total_mass;       // Sum of primary masses
    double mu;              // Mass parameter μ = m2/(m1+m2)
    double primary1_x;      // X-coordinate of primary 1
    double primary2_x;      // X-coordinate of primary 2
    
    /**
     * @brief Constructs CR3BP parameters from primary masses and separation
     * @param m1 Mass of primary 1 (kg)
     * @param m2 Mass of primary 2 (kg)
     * @param a Separation distance (m)
     */
    CR3BPParameters(double m1, double m2, double a);
    
    /**
     * @brief Gets the position of the L4 Lagrange point
     * @return Position vector of L4 point
     */
    [[nodiscard]] core::Vector2d get_l4_position() const;
    
    /**
     * @brief Gets the position of the L5 Lagrange point
     * @return Position vector of L5 point
     */
    [[nodiscard]] core::Vector2d get_l5_position() const;
    
    /**
     * @brief Calculates the Jacobi constant for a given state
     * @param position Position vector
     * @param velocity Velocity vector
     * @return Jacobi constant value
     */
    [[nodiscard]] double jacobi_constant(const core::Vector2d& position, 
                                        const core::Vector2d& velocity) const;
};

/**
 * @brief Circular Restricted Three-Body Problem solver
 * 
 * Implements the CR3BP dynamics in a rotating reference frame
 * where the two primary bodies are stationary.
 */
class CR3BPSolver {
public:
    /**
     * @brief Constructs a CR3BP solver with given parameters
     * @param params System parameters
     */
    explicit CR3BPSolver(CR3BPParameters params);
    
    /**
     * @brief Calculates the acceleration of a test particle
     * @param position Current position of the test particle
     * @param velocity Current velocity of the test particle
     * @return Acceleration vector
     */
    [[nodiscard]] core::Vector2d calculate_acceleration(const core::Vector2d& position, 
                                                       const core::Vector2d& velocity) const;
    
    /**
     * @brief Calculates the effective potential at a given position
     * @param position Position vector
     * @return Effective potential value
     */
    [[nodiscard]] double effective_potential(const core::Vector2d& position) const;
    
    /**
     * @brief Gets the system parameters
     * @return Reference to the CR3BP parameters
     */
    [[nodiscard]] const CR3BPParameters& parameters() const noexcept { return params_; }
    
    /**
     * @brief Creates a derivative function for use with integrators
     * @return Derivative function compatible with integrators
     */
    [[nodiscard]] integrators::DerivativeFunction create_derivative_function() const;
    
private:
    CR3BPParameters params_;
    
    /**
     * @brief Calculates gravitational acceleration from both primaries
     * @param position Position of the test particle
     * @return Gravitational acceleration vector
     */
    [[nodiscard]] core::Vector2d gravitational_acceleration(const core::Vector2d& position) const;
    
    /**
     * @brief Calculates centrifugal acceleration
     * @param position Position of the test particle
     * @return Centrifugal acceleration vector
     */
    [[nodiscard]] core::Vector2d centrifugal_acceleration(const core::Vector2d& position) const;
    
    /**
     * @brief Calculates Coriolis acceleration
     * @param velocity Velocity of the test particle
     * @return Coriolis acceleration vector
     */
    [[nodiscard]] core::Vector2d coriolis_acceleration(const core::Vector2d& velocity) const;
};

/**
 * @brief Factory function to create common CR3BP systems
 */
namespace systems {
    /**
     * @brief Creates Sun-Earth CR3BP parameters
     * @return CR3BP parameters for Sun-Earth system
     */
    [[nodiscard]] CR3BPParameters sun_earth();
    
    /**
     * @brief Creates Saturn-Mimas CR3BP parameters
     * @return CR3BP parameters for Saturn-Mimas system
     */
    [[nodiscard]] CR3BPParameters saturn_mimas();
}

}  // namespace orbital_mechanics::physics