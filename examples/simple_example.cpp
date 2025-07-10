/**
 * @file simple_example.cpp
 * @brief Simple example demonstrating basic usage of the orbital mechanics library
 * 
 * This example shows how to:
 * - Create a simple simulation
 * - Add bodies to the simulation
 * - Set up a physics solver
 * - Run the simulation and analyze results
 * 
 * @author Arnav Menon
 */

#include <orbital_mechanics/orbital_mechanics.hpp>
#include <iostream>
#include <iomanip>

using namespace orbital_mechanics;

int main() {
    std::cout << "Orbital Mechanics Library - Simple Example\n";
    std::cout << "Version: " << version_string() << "\n";
    std::cout << "==========================================\n\n";
    
    try {
        // Create simulation configuration
        simulation::SimulationConfig config;
        config.time_step = 1800.0;  // 30 minutes
        config.total_time = 30.0 * core::Constants::DAY_TO_SECONDS;  // 30 days
        config.output_interval = 48;  // Output every 24 hours
        config.integrator_type = "rk4";
        config.output_format = "text";
        config.output_filename = "simple_example_output.txt";
        config.write_velocities = true;
        config.console_update_interval = 50;
        
        std::cout << "Configuration:\n";
        std::cout << "  Time step: " << config.time_step << " seconds\n";
        std::cout << "  Total time: " << config.total_time / core::Constants::DAY_TO_SECONDS << " days\n";
        std::cout << "  Integrator: " << config.integrator_type << "\n";
        std::cout << "  Output format: " << config.output_format << "\n\n";
        
        // Create simulation
        simulation::Simulation sim(config);
        
        // Create Sun-Earth system
        auto solver = std::make_unique<physics::CR3BPSolver>(physics::systems::sun_earth());
        const auto& params = solver->parameters();
        
        // Add primary bodies
        sim.add_body(core::Body(params.primary1_mass, 
                               core::Vector2d{params.primary1_x, 0.0}, 
                               core::Vector2d{0.0, 0.0}, "Sun"));
        sim.add_body(core::Body(params.primary2_mass, 
                               core::Vector2d{params.primary2_x, 0.0}, 
                               core::Vector2d{0.0, 0.0}, "Earth"));
        
        // Add test particle near L4 point with small perturbation
        auto l4_pos = params.get_l4_position();
        l4_pos.x += 1e7;  // 10,000 km perturbation
        sim.add_body(core::Body(1.0, l4_pos, core::Vector2d{0.0, 0.0}, "Test Particle"));
        
        std::cout << "System setup:\n";
        std::cout << "  Primary 1 (Sun): " << std::scientific << params.primary1_mass << " kg\n";
        std::cout << "  Primary 2 (Earth): " << std::scientific << params.primary2_mass << " kg\n";
        std::cout << "  L4 position: " << l4_pos << " m\n";
        std::cout << "  Angular velocity: " << params.angular_velocity << " rad/s\n\n";
        
        // Set physics solver
        sim.set_physics_solver(std::move(solver));
        
        // Add a simple progress callback
        sim.set_progress_callback([](size_t step, size_t total, double elapsed, double sim_time) {
            if (step % 100 == 0 || step == total - 1) {
                double percent = 100.0 * step / total;
                double days = sim_time / core::Constants::DAY_TO_SECONDS;
                std::cout << "Progress: " << std::fixed << std::setprecision(1) 
                          << std::setw(5) << percent << "% - Day " 
                          << std::setw(6) << days << " (Elapsed: " 
                          << std::setprecision(2) << elapsed << "s)\n";
            }
        });
        
        // Run simulation
        std::cout << "Running simulation...\n";
        bool success = sim.run();
        
        if (success) {
            const auto& stats = sim.stats();
            std::cout << "\nSimulation completed successfully!\n";
            std::cout << "Statistics:\n";
            std::cout << "  Total steps: " << stats.total_steps << "\n";
            std::cout << "  Output points: " << stats.output_points << "\n";
            std::cout << "  Elapsed time: " << std::fixed << std::setprecision(2) 
                      << stats.elapsed_time_seconds << " seconds\n";
            std::cout << "  Performance: " << std::setprecision(0) 
                      << (stats.total_steps / stats.elapsed_time_seconds) << " steps/second\n";
            std::cout << "  Output file: " << config.output_filename << "\n";
            
            // Calculate some derived quantities
            double simulation_time_days = config.total_time / core::Constants::DAY_TO_SECONDS;
            double real_time_ratio = simulation_time_days / (stats.elapsed_time_seconds / core::Constants::DAY_TO_SECONDS);
            
            std::cout << "  Real-time ratio: " << std::scientific << std::setprecision(1) 
                      << real_time_ratio << "x (simulation runs " << real_time_ratio 
                      << " times faster than real time)\n";
        } else {
            std::cout << "Simulation failed!\n";
            return EXIT_FAILURE;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    
    return EXIT_SUCCESS;
}