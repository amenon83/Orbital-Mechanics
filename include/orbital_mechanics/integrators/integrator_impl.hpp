#pragma once

#include "integrator.hpp"
#include <stdexcept>
#include <algorithm>
#include <cctype>

namespace orbital_mechanics::integrators {

inline State EulerIntegrator::integrate_step(const State& state, double mass, double time, 
                                            double dt, const DerivativeFunction& derivative_func,
                                            void* user_data) {
    const auto derivative = derivative_func(state, mass, time, user_data);
    
    State new_state;
    new_state.position = state.position + derivative.velocity * dt;
    new_state.velocity = state.velocity + derivative.acceleration * dt;
    
    return new_state;
}

inline State RK4Integrator::integrate_step(const State& state, double mass, double time, 
                                          double dt, const DerivativeFunction& derivative_func,
                                          void* user_data) {
    // k1 = f(t, y)
    const auto k1 = derivative_func(state, mass, time, user_data);
    
    // k2 = f(t + dt/2, y + k1*dt/2)
    State state_k2;
    state_k2.position = state.position + k1.velocity * (dt * 0.5);
    state_k2.velocity = state.velocity + k1.acceleration * (dt * 0.5);
    const auto k2 = derivative_func(state_k2, mass, time + dt * 0.5, user_data);
    
    // k3 = f(t + dt/2, y + k2*dt/2)
    State state_k3;
    state_k3.position = state.position + k2.velocity * (dt * 0.5);
    state_k3.velocity = state.velocity + k2.acceleration * (dt * 0.5);
    const auto k3 = derivative_func(state_k3, mass, time + dt * 0.5, user_data);
    
    // k4 = f(t + dt, y + k3*dt)
    State state_k4;
    state_k4.position = state.position + k3.velocity * dt;
    state_k4.velocity = state.velocity + k3.acceleration * dt;
    const auto k4 = derivative_func(state_k4, mass, time + dt, user_data);
    
    // Combine results: y_new = y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    State new_state;
    new_state.position = state.position + (dt / 6.0) * (k1.velocity + 2.0 * k2.velocity + 2.0 * k3.velocity + k4.velocity);
    new_state.velocity = state.velocity + (dt / 6.0) * (k1.acceleration + 2.0 * k2.acceleration + 2.0 * k3.acceleration + k4.acceleration);
    
    return new_state;
}

inline State VerletIntegrator::integrate_step(const State& state, double mass, double time, 
                                             double dt, const DerivativeFunction& derivative_func,
                                             void* user_data) {
    const auto derivative = derivative_func(state, mass, time, user_data);
    
    State new_state;
    // Verlet integration: x(t+dt) = x(t) + v(t)*dt + 0.5*a(t)*dt^2
    new_state.position = state.position + state.velocity * dt + 0.5 * derivative.acceleration * (dt * dt);
    
    // Calculate acceleration at new position
    State temp_state;
    temp_state.position = new_state.position;
    temp_state.velocity = state.velocity;  // Use old velocity for acceleration calculation
    
    const auto new_derivative = derivative_func(temp_state, mass, time + dt, user_data);
    
    // Velocity: v(t+dt) = v(t) + 0.5*(a(t) + a(t+dt))*dt
    new_state.velocity = state.velocity + 0.5 * (derivative.acceleration + new_derivative.acceleration) * dt;
    
    return new_state;
}

inline std::unique_ptr<Integrator> create_integrator(const std::string& type) {
    std::string lower_type = type;
    std::transform(lower_type.begin(), lower_type.end(), lower_type.begin(), ::tolower);
    
    if (lower_type == "euler") {
        return std::make_unique<EulerIntegrator>();
    } else if (lower_type == "rk4") {
        return std::make_unique<RK4Integrator>();
    } else if (lower_type == "verlet") {
        return std::make_unique<VerletIntegrator>();
    } else {
        throw std::invalid_argument("Unknown integrator type: " + type);
    }
}

}  // namespace orbital_mechanics::integrators