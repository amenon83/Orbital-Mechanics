/**
 * @file cassini_simulation.cpp
 * @brief Main executable for Cassini Division simulation
 * 
 * This program simulates the dynamics of numerous test particles in Saturn's rings,
 * demonstrating the formation of the Cassini Division due to the 2:1 orbital
 * resonance with Saturn's moon Mimas.
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
    
    // Calculate current time in Mimas orbits
    const double mimas_period = 81018.0;  // Mimas orbital period in seconds
    const double mimas_orbits = current_time / mimas_period;
    
    std::cout << "\rProgress: " << std::setw(3) << percent << "% (" 
              << std::setw(8) << step << "/" << total_steps 
              << "), Orbits: " << std::fixed << std::setprecision(1) 
              << std::setw(8) << mimas_orbits << ", "
              << "Elapsed: " << std::setprecision(1) << elapsed_time << "s" << std::flush;
}

int main(int argc, char* argv[]) {
    std::cout << "Orbital Mechanics Library - Cassini Division Simulation\n";
    std::cout << "Version: " << version_string() << "\n";
    std::cout << "=======================================================\n\n";
    
    try {
        // Parse command line arguments
        int num_particles = 100;
        if (argc > 1) {
            num_particles = std::atoi(argv[1]);
            if (num_particles <= 0) {
                std::cerr << "Error: Number of particles must be positive" << std::endl;
                return EXIT_FAILURE;
            }
        }
        
        // Simulation configuration
        simulation::SimulationConfig config;
        config.time_step = 100.0;  // 100 seconds
        config.total_time = 10.0 * 81018.0;  // 10 Mimas orbits
        config.output_interval = 50;  // Output every 50 steps
        config.integrator_type = "rk4";
        config.output_format = "text";
        config.output_filename = "cassini_output.txt";
        config.write_velocities = false;  // Only positions for visualization
        config.console_update_interval = 1000;
        
        // Create simulation
        simulation::Simulation sim(config);
        
        // Set up progress callback
        sim.set_progress_callback(progress_callback);
        
        // Create Saturn-Mimas CR3BP system
        auto solver = std::make_unique<physics::CR3BPSolver>(physics::systems::saturn_mimas());
        const auto& params = solver->parameters();
        
        std::cout << "System Parameters:\n";
        std::cout << "  Primary 1 (Saturn) mass: " << std::scientific << params.primary1_mass << " kg\n";
        std::cout << "  Primary 2 (Mimas) mass: " << std::scientific << params.primary2_mass << " kg\n";
        std::cout << "  Separation: " << std::scientific << params.separation << " m\n";
        std::cout << "  Angular velocity: " << std::scientific << params.angular_velocity << " rad/s\n";
        std::cout << "  Mimas orbital period: " << std::fixed << std::setprecision(2) 
                  << (core::Constants::TWO_PI / params.angular_velocity) / 3600.0 << " hours\n";
        std::cout << "  Number of particles: " << num_particles << "\n\n";
        
        // Add primary bodies (Saturn and Mimas)
        sim.add_body(core::Body(params.primary1_mass, 
                               core::Vector2d{params.primary1_x, 0.0}, 
                               core::Vector2d{0.0, 0.0}, "Saturn"));
        sim.add_body(core::Body(params.primary2_mass, 
                               core::Vector2d{params.primary2_x, 0.0}, 
                               core::Vector2d{0.0, 0.0}, "Mimas"));
        
        // Add test particles in the Cassini Division region
        std::random_device rd;
        std::mt19937 gen(rd());
        const double min_radius = 1.10e8;  // Inside Cassini Division
        const double max_radius = 1.30e8;  // Outside Cassini Division
        std::uniform_real_distribution<> radius_dist(min_radius, max_radius);
        std::uniform_real_distribution<> angle_dist(0.0, core::Constants::TWO_PI);
        
        std::cout << "Initializing " << num_particles << " test particles...\n";
        std::cout << "  Radial range: " << std::scientific << min_radius 
                  << " to " << max_radius << " m\n";
        std::cout << "  Radial range: " << std::fixed << std::setprecision(1) 
                  << (min_radius / 1e6) << " to " << (max_radius / 1e6) << " km\n\n";
        
        for (int i = 0; i < num_particles; ++i) {
            const double radius = radius_dist(gen);
            const double angle = angle_dist(gen);
            
            // Create particle in circular orbit around Saturn
            auto particle = core::make_circular_orbit_body(
                1.0, radius, params.primary1_mass, angle, 
                "Particle_" + std::to_string(i));
            
            // Transform to rotating frame
            const double omega = params.angular_velocity;
            auto pos = particle.position();
            auto vel = particle.velocity();
            
            // Subtract rotation velocity to get velocity in rotating frame
            vel.x -= -omega * pos.y;
            vel.y -= omega * pos.x;
            particle.set_velocity(vel);
            
            sim.add_body(particle);
        }
        
        // Set physics solver
        sim.set_physics_solver(std::move(solver));
        
        // Performance warning for large simulations
        if (num_particles > 1000) {
            std::cout << "WARNING: Large number of particles may result in long simulation time.\n";
            std::cout << "Consider using fewer particles for initial testing.\n\n";
        }
        
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
            std::cout << "  Particle performance: " << std::scientific << std::setprecision(2) 
                      << (stats.total_steps * num_particles / stats.elapsed_time_seconds) 
                      << " particle-steps/second\n";
        }
        
        return success ? EXIT_SUCCESS : EXIT_FAILURE;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
}