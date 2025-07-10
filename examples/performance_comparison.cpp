/**
 * @file performance_comparison.cpp
 * @brief Performance comparison between different integrators
 * 
 * This example demonstrates:
 * - Comparing different numerical integrators
 * - Measuring accuracy and performance
 * - Analyzing energy conservation
 * 
 * @author Arnav Menon
 */

#include <orbital_mechanics/orbital_mechanics.hpp>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <cmath>

using namespace orbital_mechanics;

struct IntegratorResult {
    std::string name;
    double elapsed_time;
    double final_energy;
    double energy_error;
    core::Vector2d final_position;
    core::Vector2d final_velocity;
    double position_error;
};

// Analytical solution for simple harmonic oscillator
core::Vector2d analytical_position(double t, const core::Vector2d& initial_pos, const core::Vector2d& initial_vel) {
    return core::Vector2d(
        initial_pos.x * std::cos(t) + initial_vel.x * std::sin(t),
        initial_pos.y * std::cos(t) + initial_vel.y * std::sin(t)
    );
}

core::Vector2d analytical_velocity(double t, const core::Vector2d& initial_pos, const core::Vector2d& initial_vel) {
    return core::Vector2d(
        -initial_pos.x * std::sin(t) + initial_vel.x * std::cos(t),
        -initial_pos.y * std::sin(t) + initial_vel.y * std::cos(t)
    );
}

IntegratorResult test_integrator(const std::string& integrator_type, 
                               double time_step, double total_time,
                               const core::Vector2d& initial_pos,
                               const core::Vector2d& initial_vel) {
    
    auto integrator = integrators::create_integrator(integrator_type);
    
    // Simple harmonic oscillator derivative function
    auto derivative_func = [](const integrators::State& state, double /*mass*/, double /*time*/, void* /*user_data*/) {
        return integrators::Derivative(state.velocity, core::Vector2d(-state.position.x, -state.position.y));
    };
    
    integrators::State state(initial_pos, initial_vel);
    double initial_energy = 0.5 * (state.velocity.magnitude_squared() + state.position.magnitude_squared());
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Integration loop
    double time = 0.0;
    while (time < total_time) {
        state = integrator->integrate_step(state, 1.0, time, time_step, derivative_func);
        time += time_step;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration<double>(end_time - start_time);
    
    double final_energy = 0.5 * (state.velocity.magnitude_squared() + state.position.magnitude_squared());
    double energy_error = std::abs(final_energy - initial_energy) / initial_energy;
    
    // Calculate position error compared to analytical solution
    auto analytical_pos = analytical_position(total_time, initial_pos, initial_vel);
    double position_error = (state.position - analytical_pos).magnitude();
    
    return {
        integrator->name(),
        elapsed.count(),
        final_energy,
        energy_error,
        state.position,
        state.velocity,
        position_error
    };
}

int main() {
    std::cout << "Orbital Mechanics Library - Performance Comparison\n";
    std::cout << "Version: " << version_string() << "\n";
    std::cout << "==================================================\n\n";
    
    try {
        // Test parameters
        const double time_step = 0.01;
        const double total_time = 10.0;  // 10 seconds
        const core::Vector2d initial_pos(1.0, 0.0);
        const core::Vector2d initial_vel(0.0, 1.0);
        
        std::cout << "Test Parameters:\n";
        std::cout << "  Problem: Simple harmonic oscillator (x'' = -x)\n";
        std::cout << "  Initial position: " << initial_pos << "\n";
        std::cout << "  Initial velocity: " << initial_vel << "\n";
        std::cout << "  Time step: " << time_step << " s\n";
        std::cout << "  Total time: " << total_time << " s\n";
        std::cout << "  Number of steps: " << static_cast<int>(total_time / time_step) << "\n\n";
        
        // Test different integrators
        std::vector<std::string> integrator_types = {"euler", "rk4", "verlet"};
        std::vector<IntegratorResult> results;
        
        for (const auto& type : integrator_types) {
            std::cout << "Testing " << type << " integrator...\n";
            auto result = test_integrator(type, time_step, total_time, initial_pos, initial_vel);
            results.push_back(result);
        }
        
        std::cout << "\nResults:\n";
        std::cout << "=======\n\n";
        
        // Print results table
        std::cout << std::left << std::setw(10) << "Integrator"
                  << std::setw(12) << "Time (s)"
                  << std::setw(15) << "Energy Error"
                  << std::setw(15) << "Position Error"
                  << std::setw(20) << "Final Position"
                  << std::setw(20) << "Final Velocity" << "\n";
        std::cout << std::string(90, '-') << "\n";
        
        for (const auto& result : results) {
            std::cout << std::left << std::setw(10) << result.name
                      << std::setw(12) << std::fixed << std::setprecision(4) << result.elapsed_time
                      << std::setw(15) << std::scientific << std::setprecision(2) << result.energy_error
                      << std::setw(15) << std::scientific << std::setprecision(2) << result.position_error
                      << std::setw(20) << result.final_position
                      << std::setw(20) << result.final_velocity << "\n";
        }
        
        std::cout << "\nAnalytical solution:\n";
        auto analytical_pos = analytical_position(total_time, initial_pos, initial_vel);
        auto analytical_vel = analytical_velocity(total_time, initial_pos, initial_vel);
        std::cout << "  Final position: " << analytical_pos << "\n";
        std::cout << "  Final velocity: " << analytical_vel << "\n";
        
        // Performance analysis
        std::cout << "\nPerformance Analysis:\n";
        std::cout << "====================\n";
        
        double fastest_time = results[0].elapsed_time;
        for (const auto& result : results) {
            if (result.elapsed_time < fastest_time) {
                fastest_time = result.elapsed_time;
            }
        }
        
        for (const auto& result : results) {
            double relative_speed = fastest_time / result.elapsed_time;
            double steps_per_second = (total_time / time_step) / result.elapsed_time;
            
            std::cout << result.name << ":\n";
            std::cout << "  Relative speed: " << std::fixed << std::setprecision(2) << relative_speed << "x\n";
            std::cout << "  Steps per second: " << std::scientific << std::setprecision(1) << steps_per_second << "\n";
            std::cout << "  Energy conservation: " << (result.energy_error < 1e-10 ? "Excellent" : 
                        result.energy_error < 1e-6 ? "Good" : 
                        result.energy_error < 1e-3 ? "Fair" : "Poor") << "\n";
            std::cout << "  Position accuracy: " << (result.position_error < 1e-10 ? "Excellent" : 
                        result.position_error < 1e-6 ? "Good" : 
                        result.position_error < 1e-3 ? "Fair" : "Poor") << "\n\n";
        }
        
        // Recommendations
        std::cout << "Recommendations:\n";
        std::cout << "================\n";
        std::cout << "- For high accuracy requirements: Use RK4 integrator\n";
        std::cout << "- For long-term energy conservation: Use Verlet integrator\n";
        std::cout << "- For maximum speed (low accuracy): Use Euler integrator\n";
        std::cout << "- For general orbital mechanics: RK4 provides the best balance\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    
    return EXIT_SUCCESS;
}