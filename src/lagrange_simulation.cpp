/**
 * @file lagrange_simulation.cpp
 * @brief Main executable for Lagrange point simulation
 * 
 * This program simulates the motion of a test particle near the L4 Lagrange point
 * of the Sun-Earth system using the modern orbital mechanics library.
 * 
 * @author Arnav Menon
 * @version 1.0.0
 */

#include <orbital_mechanics/orbital_mechanics.hpp>
#include <iostream>
#include <iomanip>
#include <chrono>

using namespace orbital_mechanics;

/**
 * @brief Custom progress callback for the simulation
 */
void progress_callback(size_t step, size_t total_steps, double elapsed_time, double current_time) {
    const int percent = static_cast<int>((100.0 * step) / total_steps);
    const double current_time_days = current_time / core::Constants::DAY_TO_SECONDS;
    
    std::cout << "\rProgress: " << std::setw(3) << percent << "% (" 
              << std::setw(8) << step << "/" << total_steps 
              << "), Simulation Time: " << std::fixed << std::setprecision(1) 
              << std::setw(8) << current_time_days << " days, "
              << "Elapsed: " << std::setprecision(1) << elapsed_time << "s" << std::flush;
}

int main(int argc, char* argv[]) {
    std::cout << "Orbital Mechanics Library - Lagrange Point Simulation\n";
    std::cout << "Version: " << version_string() << "\n";
    std::cout << "======================================================\n\n";
    
    try {
        // Simulation configuration
        simulation::SimulationConfig config;
        config.time_step = 3600.0;  // 1 hour
        config.total_time = 365.25 * core::Constants::DAY_TO_SECONDS;  // 1 year
        config.output_interval = 24;  // Output every 24 hours
        config.integrator_type = "rk4";
        config.output_format = "text";
        config.output_filename = "lagrange_output.txt";
        config.write_velocities = true;
        config.console_update_interval = 100;
        
        // Create simulation
        simulation::Simulation sim(config);
        
        // Set up progress callback
        sim.set_progress_callback(progress_callback);
        
        // Create Sun-Earth CR3BP system
        auto solver = std::make_unique<physics::CR3BPSolver>(physics::systems::sun_earth());
        const auto& params = solver->parameters();
        
        std::cout << "System Parameters:\n";
        std::cout << "  Primary 1 (Sun) mass: " << std::scientific << params.primary1_mass << " kg\n";
        std::cout << "  Primary 2 (Earth) mass: " << std::scientific << params.primary2_mass << " kg\n";
        std::cout << "  Separation: " << std::scientific << params.separation << " m\n";
        std::cout << "  Angular velocity: " << std::scientific << params.angular_velocity << " rad/s\n";
        std::cout << "  Orbital period: " << std::fixed << std::setprecision(2) 
                  << (core::Constants::TWO_PI / params.angular_velocity) / core::Constants::DAY_TO_SECONDS 
                  << " days\n\n";
        
        // Add primary bodies (Sun and Earth)
        sim.add_body(core::Body(params.primary1_mass, 
                               core::Vector2d{params.primary1_x, 0.0}, 
                               core::Vector2d{0.0, 0.0}, "Sun"));
        sim.add_body(core::Body(params.primary2_mass, 
                               core::Vector2d{params.primary2_x, 0.0}, 
                               core::Vector2d{0.0, 0.0}, "Earth"));
        
        // Add test particle near L4 point
        auto l4_pos = params.get_l4_position();
        const double perturbation = 1.5e7;  // 15,000 km perturbation
        l4_pos.x += perturbation;
        
        std::cout << "L4 Position: " << l4_pos << "\n";
        std::cout << "Test particle position: " << l4_pos << "\n";
        std::cout << "Perturbation: " << std::scientific << perturbation << " m\n\n";
        
        sim.add_body(core::Body(1.0, l4_pos, core::Vector2d{0.0, 0.0}, "Test Particle"));
        
        // Set physics solver
        sim.set_physics_solver(std::move(solver));
        
        // Run simulation
        bool success = sim.run();
        
        if (success) {
            const auto& stats = sim.stats();
            std::cout << "\n\nSimulation Statistics:\n";
            std::cout << "  Total steps: " << stats.total_steps << "\n";
            std::cout << "  Output points: " << stats.output_points << "\n";
            std::cout << "  Elapsed time: " << std::fixed << std::setprecision(2) 
                      << stats.elapsed_time_seconds << " seconds\n";
            std::cout << "  Average step time: " << std::scientific << std::setprecision(3) 
                      << stats.average_step_time << " seconds\n";
            std::cout << "  Performance: " << std::fixed << std::setprecision(1) 
                      << (stats.total_steps / stats.elapsed_time_seconds) << " steps/second\n";
        }
        
        return success ? EXIT_SUCCESS : EXIT_FAILURE;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
}