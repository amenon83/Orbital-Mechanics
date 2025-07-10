#pragma once

#include "simulation.hpp"
#include "../core/constants.hpp"
#include <iostream>
#include <iomanip>
#include <random>
#include <chrono>

namespace orbital_mechanics::simulation {

inline bool SimulationConfig::validate() const {
    return time_step > 0.0 && 
           total_time > 0.0 && 
           output_interval > 0 && 
           console_update_interval > 0 &&
           !integrator_type.empty() &&
           !output_format.empty() &&
           !output_filename.empty();
}

inline Simulation::Simulation(SimulationConfig config) 
    : config_(std::move(config)) {
    if (!config_.validate()) {
        throw std::invalid_argument("Invalid simulation configuration");
    }
}

inline Simulation::~Simulation() {
    cleanup();
}

inline void Simulation::add_body(const core::Body& body) {
    bodies_.push_back(body);
}

inline void Simulation::add_bodies(const std::vector<core::Body>& bodies) {
    bodies_.insert(bodies_.end(), bodies.begin(), bodies.end());
}

inline void Simulation::set_physics_solver(std::unique_ptr<physics::CR3BPSolver> solver) {
    physics_solver_ = std::move(solver);
    if (physics_solver_) {
        derivative_func_ = physics_solver_->create_derivative_function();
    }
}

inline void Simulation::set_derivative_function(integrators::DerivativeFunction func) {
    derivative_func_ = std::move(func);
}

inline void Simulation::set_progress_callback(ProgressCallback callback) {
    progress_callback_ = std::move(callback);
}

inline bool Simulation::run() {
    if (!initialize()) {
        return false;
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    const size_t total_steps = static_cast<size_t>(config_.total_time / config_.time_step);
    stats_.total_steps = total_steps;
    
    std::cout << "Running simulation with " << bodies_.size() << " bodies for " 
              << total_steps << " steps..." << std::endl;
    
    // Main simulation loop
    for (size_t step = 0; step < total_steps; ++step) {
        const double current_time = step * config_.time_step;
        
        // Integrate each body (skip primaries if using physics solver)
        size_t start_idx = (physics_solver_ != nullptr) ? 2 : 0;
        
        for (size_t i = start_idx; i < bodies_.size(); ++i) {
            auto& body = bodies_[i];
            
            // Create state for integrator
            integrators::State state(body.position(), body.velocity());
            
            // Integrate one step
            auto new_state = integrator_->integrate_step(
                state, body.mass(), current_time, config_.time_step, 
                derivative_func_);
            
            // Update body state
            body.set_state(new_state.position, new_state.velocity);
        }
        
        // Output data if needed
        if (step % config_.output_interval == 0 || step == total_steps - 1) {
            if (!write_output_data(current_time)) {
                std::cerr << "Error writing output data at step " << step << std::endl;
                return false;
            }
            stats_.output_points++;
        }
        
        // Update progress
        if (step % config_.console_update_interval == 0 || step == total_steps - 1) {
            auto current_time_point = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration<double>(current_time_point - start_time);
            update_progress(step, total_steps, elapsed.count(), current_time);
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    stats_.elapsed_time_seconds = std::chrono::duration<double>(end_time - start_time).count();
    stats_.average_step_time = stats_.elapsed_time_seconds / total_steps;
    
    cleanup();
    
    std::cout << "\nSimulation completed successfully!" << std::endl;
    std::cout << "Total time: " << std::fixed << std::setprecision(2) 
              << stats_.elapsed_time_seconds << " seconds" << std::endl;
    std::cout << "Output points: " << stats_.output_points << std::endl;
    
    return true;
}

inline void Simulation::reset() {
    cleanup();
    bodies_.clear();
    stats_.reset();
}

inline bool Simulation::initialize() {
    if (!validate_configuration()) {
        return false;
    }
    
    if (!setup_integrator()) {
        return false;
    }
    
    if (!setup_data_writer()) {
        return false;
    }
    
    if (!derivative_func_) {
        std::cerr << "Error: No derivative function set" << std::endl;
        return false;
    }
    
    return true;
}

inline void Simulation::cleanup() {
    if (data_writer_) {
        data_writer_->close();
    }
}

inline bool Simulation::setup_integrator() {
    try {
        integrator_ = integrators::create_integrator(config_.integrator_type);
        std::cout << "Using " << integrator_->name() << " integrator (order " 
                  << integrator_->order() << ")" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error creating integrator: " << e.what() << std::endl;
        return false;
    }
}

inline bool Simulation::setup_data_writer() {
    try {
        data_writer_ = io::create_data_writer(config_.output_format, config_.write_velocities);
        
        if (!data_writer_->open(config_.output_filename)) {
            std::cerr << "Error opening output file: " << config_.output_filename << std::endl;
            return false;
        }
        
        std::cout << "Writing output to " << config_.output_filename 
                  << " (format: " << data_writer_->format_name() << ")" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error creating data writer: " << e.what() << std::endl;
        return false;
    }
}

inline bool Simulation::validate_configuration() const {
    if (bodies_.empty()) {
        std::cerr << "Error: No bodies added to simulation" << std::endl;
        return false;
    }
    
    return config_.validate();
}

inline void Simulation::update_progress(size_t step, size_t total_steps, 
                                       double elapsed_time, double current_time) {
    if (progress_callback_) {
        progress_callback_(step, total_steps, elapsed_time, current_time);
    } else {
        // Default progress display
        const int percent = static_cast<int>((100.0 * step) / total_steps);
        std::cout << "\rProgress: " << percent << "% (" << step << "/" << total_steps 
                  << "), Time: " << std::fixed << std::setprecision(1) 
                  << elapsed_time << "s" << std::flush;
    }
}

inline bool Simulation::write_output_data(double current_time) {
    io::DataPoint data(current_time);
    
    for (const auto& body : bodies_) {
        data.add_body_state(body.position(), body.velocity());
    }
    
    return data_writer_->write(data);
}

inline bool run_lagrange_simulation(const SimulationConfig& config, 
                                   double perturbation_distance) {
    try {
        Simulation sim(config);
        
        // Create Sun-Earth CR3BP system
        auto solver = std::make_unique<physics::CR3BPSolver>(physics::systems::sun_earth());
        const auto& params = solver->parameters();
        
        // Add primary bodies (Sun and Earth)
        sim.add_body(core::Body(params.primary1_mass, 
                               core::Vector2d{params.primary1_x, 0.0}, 
                               core::Vector2d{0.0, 0.0}, "Sun"));
        sim.add_body(core::Body(params.primary2_mass, 
                               core::Vector2d{params.primary2_x, 0.0}, 
                               core::Vector2d{0.0, 0.0}, "Earth"));
        
        // Add test particle near L4 point
        auto l4_pos = params.get_l4_position();
        l4_pos.x += perturbation_distance;
        sim.add_body(core::Body(1.0, l4_pos, core::Vector2d{0.0, 0.0}, "Test Particle"));
        
        // Set physics solver
        sim.set_physics_solver(std::move(solver));
        
        return sim.run();
    } catch (const std::exception& e) {
        std::cerr << "Error in Lagrange simulation: " << e.what() << std::endl;
        return false;
    }
}

inline bool run_cassini_simulation(const SimulationConfig& config,
                                  int num_particles,
                                  double min_radius,
                                  double max_radius) {
    try {
        Simulation sim(config);
        
        // Create Saturn-Mimas CR3BP system
        auto solver = std::make_unique<physics::CR3BPSolver>(physics::systems::saturn_mimas());
        const auto& params = solver->parameters();
        
        // Add primary bodies (Saturn and Mimas)
        sim.add_body(core::Body(params.primary1_mass, 
                               core::Vector2d{params.primary1_x, 0.0}, 
                               core::Vector2d{0.0, 0.0}, "Saturn"));
        sim.add_body(core::Body(params.primary2_mass, 
                               core::Vector2d{params.primary2_x, 0.0}, 
                               core::Vector2d{0.0, 0.0}, "Mimas"));
        
        // Add test particles
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> radius_dist(min_radius, max_radius);
        std::uniform_real_distribution<> angle_dist(0.0, core::Constants::TWO_PI);
        
        for (int i = 0; i < num_particles; ++i) {
            const double radius = radius_dist(gen);
            const double angle = angle_dist(gen);
            
            // Create particle in circular orbit around Saturn
            auto particle = core::make_circular_orbit_body(1.0, radius, params.primary1_mass, 
                                                          angle, "Particle_" + std::to_string(i));
            
            // Transform to rotating frame
            const double omega = params.angular_velocity;
            auto pos = particle.position();
            auto vel = particle.velocity();
            
            // Subtract rotation velocity
            vel.x -= -omega * pos.y;
            vel.y -= omega * pos.x;
            particle.set_velocity(vel);
            
            sim.add_body(particle);
        }
        
        // Set physics solver
        sim.set_physics_solver(std::move(solver));
        
        return sim.run();
    } catch (const std::exception& e) {
        std::cerr << "Error in Cassini simulation: " << e.what() << std::endl;
        return false;
    }
}

}  // namespace orbital_mechanics::simulation