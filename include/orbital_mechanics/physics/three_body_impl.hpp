#pragma once

#include "three_body.hpp"
#include <stdexcept>

namespace orbital_mechanics::physics {

inline CR3BPParameters::CR3BPParameters(double m1, double m2, double a)
    : primary1_mass(m1), primary2_mass(m2), separation(a) {
    
    if (m1 <= 0.0 || m2 <= 0.0 || a <= 0.0) {
        throw std::invalid_argument("Masses and separation must be positive");
    }
    
    total_mass = m1 + m2;
    mu = m2 / total_mass;
    
    // Calculate positions of primaries relative to barycenter
    primary1_x = -a * mu;
    primary2_x = a * (1.0 - mu);
    
    // Calculate angular velocity (Kepler's third law)
    angular_velocity = std::sqrt(core::Constants::G * total_mass / (a * a * a));
}

inline core::Vector2d CR3BPParameters::get_l4_position() const {
    const double x = (primary1_x + primary2_x) * 0.5;
    const double y = separation * core::Constants::SQRT_3 * 0.5;
    return {x, y};
}

inline core::Vector2d CR3BPParameters::get_l5_position() const {
    const double x = (primary1_x + primary2_x) * 0.5;
    const double y = -separation * core::Constants::SQRT_3 * 0.5;
    return {x, y};
}

inline double CR3BPParameters::jacobi_constant(const core::Vector2d& position, 
                                              const core::Vector2d& velocity) const {
    const double r1 = std::sqrt((position.x - primary1_x) * (position.x - primary1_x) + 
                               position.y * position.y);
    const double r2 = std::sqrt((position.x - primary2_x) * (position.x - primary2_x) + 
                               position.y * position.y);
    
    const double kinetic = velocity.magnitude_squared();
    const double potential = -primary1_mass / r1 - primary2_mass / r2;
    const double centrifugal = -0.5 * angular_velocity * angular_velocity * 
                              (position.x * position.x + position.y * position.y);
    
    return -2.0 * (kinetic + potential + centrifugal);
}

inline CR3BPSolver::CR3BPSolver(CR3BPParameters params) : params_(std::move(params)) {}

inline core::Vector2d CR3BPSolver::calculate_acceleration(const core::Vector2d& position, 
                                                         const core::Vector2d& velocity) const {
    return gravitational_acceleration(position) + 
           centrifugal_acceleration(position) + 
           coriolis_acceleration(velocity);
}

inline double CR3BPSolver::effective_potential(const core::Vector2d& position) const {
    const double r1 = std::sqrt((position.x - params_.primary1_x) * (position.x - params_.primary1_x) + 
                               position.y * position.y);
    const double r2 = std::sqrt((position.x - params_.primary2_x) * (position.x - params_.primary2_x) + 
                               position.y * position.y);
    
    if (r1 < core::Constants::SMALL_NUMBER || r2 < core::Constants::SMALL_NUMBER) {
        return -std::numeric_limits<double>::infinity();
    }
    
    return -core::Constants::G * params_.primary1_mass / r1 - 
           core::Constants::G * params_.primary2_mass / r2 - 
           0.5 * params_.angular_velocity * params_.angular_velocity * 
           (position.x * position.x + position.y * position.y);
}

inline integrators::DerivativeFunction CR3BPSolver::create_derivative_function() const {
    return [this](const integrators::State& state, double /*mass*/, double /*time*/, void* /*user_data*/) {
        const auto acceleration = calculate_acceleration(state.position, state.velocity);
        return integrators::Derivative(state.velocity, acceleration);
    };
}

inline core::Vector2d CR3BPSolver::gravitational_acceleration(const core::Vector2d& position) const {
    // Distance vectors to primaries
    const double dx1 = position.x - params_.primary1_x;
    const double dy1 = position.y;
    const double dx2 = position.x - params_.primary2_x;
    const double dy2 = position.y;
    
    // Squared distances
    const double r1_sq = dx1 * dx1 + dy1 * dy1;
    const double r2_sq = dx2 * dx2 + dy2 * dy2;
    
    // Check for collisions
    if (r1_sq < core::Constants::SMALL_NUMBER || r2_sq < core::Constants::SMALL_NUMBER) {
        return {0.0, 0.0};
    }
    
    // Calculate acceleration components
    const double r1 = std::sqrt(r1_sq);
    const double r2 = std::sqrt(r2_sq);
    
    const double inv_r1_cubed = 1.0 / (r1_sq * r1);
    const double inv_r2_cubed = 1.0 / (r2_sq * r2);
    
    const double ax = -core::Constants::G * params_.primary1_mass * dx1 * inv_r1_cubed - 
                      core::Constants::G * params_.primary2_mass * dx2 * inv_r2_cubed;
    const double ay = -core::Constants::G * params_.primary1_mass * dy1 * inv_r1_cubed - 
                      core::Constants::G * params_.primary2_mass * dy2 * inv_r2_cubed;
    
    return {ax, ay};
}

inline core::Vector2d CR3BPSolver::centrifugal_acceleration(const core::Vector2d& position) const {
    const double omega_sq = params_.angular_velocity * params_.angular_velocity;
    return {omega_sq * position.x, omega_sq * position.y};
}

inline core::Vector2d CR3BPSolver::coriolis_acceleration(const core::Vector2d& velocity) const {
    const double two_omega = 2.0 * params_.angular_velocity;
    return {two_omega * velocity.y, -two_omega * velocity.x};
}

namespace systems {

inline CR3BPParameters sun_earth() {
    return CR3BPParameters(core::Constants::SUN_MASS, 
                          core::Constants::EARTH_MASS, 
                          core::Constants::SUN_EARTH_DISTANCE);
}

inline CR3BPParameters saturn_mimas() {
    return CR3BPParameters(core::Constants::SATURN_MASS,
                          core::Constants::MIMAS_MASS,
                          core::Constants::MIMAS_SEMI_MAJOR_AXIS);
}

}  // namespace systems
}  // namespace orbital_mechanics::physics