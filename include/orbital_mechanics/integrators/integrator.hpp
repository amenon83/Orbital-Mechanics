#pragma once

#include "../core/vector.hpp"
#include "../core/body.hpp"
#include <vector>
#include <memory>
#include <functional>

namespace orbital_mechanics::integrators {

/**
 * @brief State vector containing position and velocity
 */
struct State {
    core::Vector2d position;
    core::Vector2d velocity;
    
    State() = default;
    State(const core::Vector2d& pos, const core::Vector2d& vel) 
        : position(pos), velocity(vel) {}
};

/**
 * @brief Derivative vector containing velocity and acceleration
 */
struct Derivative {
    core::Vector2d velocity;
    core::Vector2d acceleration;
    
    Derivative() = default;
    Derivative(const core::Vector2d& vel, const core::Vector2d& acc)
        : velocity(vel), acceleration(acc) {}
};

/**
 * @brief Function signature for calculating derivatives
 * @param state Current state (position, velocity)
 * @param mass Mass of the body
 * @param time Current time
 * @param user_data Optional user data for the calculation
 * @return Derivative (velocity, acceleration)
 */
using DerivativeFunction = std::function<Derivative(const State&, double, double, void*)>;

/**
 * @brief Abstract base class for numerical integrators
 * 
 * This class provides the interface for all numerical integration schemes
 * used in orbital mechanics simulations. Derived classes implement specific
 * integration methods like Euler, Runge-Kutta, Verlet, etc.
 */
class Integrator {
public:
    virtual ~Integrator() = default;
    
    /**
     * @brief Performs one integration step
     * @param state Current state (position, velocity)
     * @param mass Mass of the body
     * @param time Current time
     * @param dt Time step
     * @param derivative_func Function to calculate derivatives
     * @param user_data Optional user data for derivative calculation
     * @return New state after integration step
     */
    virtual State integrate_step(const State& state, double mass, double time, 
                                double dt, const DerivativeFunction& derivative_func,
                                void* user_data = nullptr) = 0;
    
    /**
     * @brief Gets the name of the integrator
     * @return String name of the integrator
     */
    virtual std::string name() const = 0;
    
    /**
     * @brief Gets the order of accuracy of the integrator
     * @return Order of accuracy (e.g., 1 for Euler, 4 for RK4)
     */
    virtual int order() const = 0;
    
    /**
     * @brief Checks if the integrator is symplectic
     * @return True if the integrator preserves phase space volume
     */
    virtual bool is_symplectic() const = 0;
};

/**
 * @brief Euler integrator (1st order)
 * 
 * Simple first-order integration scheme. Fast but not very accurate
 * for long-term simulations.
 */
class EulerIntegrator : public Integrator {
public:
    State integrate_step(const State& state, double mass, double time, 
                        double dt, const DerivativeFunction& derivative_func,
                        void* user_data = nullptr) override;
    
    std::string name() const override { return "Euler"; }
    int order() const override { return 1; }
    bool is_symplectic() const override { return false; }
};

/**
 * @brief 4th-order Runge-Kutta integrator
 * 
 * Classical RK4 method with good accuracy and stability for
 * most orbital mechanics problems.
 */
class RK4Integrator : public Integrator {
public:
    State integrate_step(const State& state, double mass, double time, 
                        double dt, const DerivativeFunction& derivative_func,
                        void* user_data = nullptr) override;
    
    std::string name() const override { return "RK4"; }
    int order() const override { return 4; }
    bool is_symplectic() const override { return false; }
};

/**
 * @brief Verlet integrator (2nd order, symplectic)
 * 
 * Symplectic integrator that preserves energy well over long time periods.
 * Particularly good for conservative systems.
 */
class VerletIntegrator : public Integrator {
public:
    State integrate_step(const State& state, double mass, double time, 
                        double dt, const DerivativeFunction& derivative_func,
                        void* user_data = nullptr) override;
    
    std::string name() const override { return "Verlet"; }
    int order() const override { return 2; }
    bool is_symplectic() const override { return true; }
};

/**
 * @brief Factory function to create integrators
 * @param type Type of integrator ("euler", "rk4", "verlet")
 * @return Unique pointer to the created integrator
 */
std::unique_ptr<Integrator> create_integrator(const std::string& type);

}  // namespace orbital_mechanics::integrators