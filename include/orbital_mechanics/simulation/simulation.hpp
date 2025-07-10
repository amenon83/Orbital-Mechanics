#pragma once

#include "../core/body.hpp"
#include "../integrators/integrator.hpp"
#include "../io/data_writer.hpp"
#include "../physics/three_body.hpp"
#include <vector>
#include <memory>
#include <functional>
#include <chrono>

namespace orbital_mechanics::simulation {

/**
 * @brief Configuration for simulation parameters
 */
struct SimulationConfig {
    double time_step = 1.0;           // Integration time step (s)
    double total_time = 1000.0;       // Total simulation time (s)
    int output_interval = 1;          // Output data every N steps
    std::string integrator_type = "rk4";  // Integrator type
    std::string output_format = "text";   // Output format
    std::string output_filename = "output.txt";  // Output filename
    bool write_velocities = false;    // Include velocities in output
    int console_update_interval = 100;  // Console progress update interval
    
    // Validation
    bool validate() const;
};

/**
 * @brief Statistics collected during simulation
 */
struct SimulationStats {
    double elapsed_time_seconds = 0.0;
    size_t total_steps = 0;
    size_t output_points = 0;
    double average_step_time = 0.0;
    
    void reset() {
        elapsed_time_seconds = 0.0;
        total_steps = 0;
        output_points = 0;
        average_step_time = 0.0;
    }
};

/**
 * @brief Progress callback function type
 * @param step Current simulation step
 * @param total_steps Total number of steps
 * @param elapsed_time Elapsed time in seconds
 * @param current_time Current simulation time
 */
using ProgressCallback = std::function<void(size_t, size_t, double, double)>;

/**
 * @brief Main simulation class for orbital mechanics problems
 * 
 * This class coordinates the numerical integration of orbital mechanics
 * problems, handles I/O operations, and provides progress monitoring.
 */
class Simulation {
public:
    /**
     * @brief Constructs a simulation with given configuration
     * @param config Simulation configuration
     */
    explicit Simulation(SimulationConfig config);
    
    /**
     * @brief Destructor
     */
    ~Simulation();
    
    /**
     * @brief Adds a body to the simulation
     * @param body Body to add
     */
    void add_body(const core::Body& body);
    
    /**
     * @brief Adds multiple bodies to the simulation
     * @param bodies Vector of bodies to add
     */
    void add_bodies(const std::vector<core::Body>& bodies);
    
    /**
     * @brief Sets the physics solver for the simulation
     * @param solver Physics solver (e.g., CR3BP solver)
     */
    void set_physics_solver(std::unique_ptr<physics::CR3BPSolver> solver);
    
    /**
     * @brief Sets a custom derivative function
     * @param func Derivative function
     */
    void set_derivative_function(integrators::DerivativeFunction func);
    
    /**
     * @brief Sets a progress callback function
     * @param callback Progress callback function
     */
    void set_progress_callback(ProgressCallback callback);
    
    /**
     * @brief Runs the simulation
     * @return true if simulation completed successfully
     */
    bool run();
    
    /**
     * @brief Gets the current simulation statistics
     * @return Reference to simulation statistics
     */
    const SimulationStats& stats() const { return stats_; }
    
    /**
     * @brief Gets the current simulation configuration
     * @return Reference to simulation configuration
     */
    const SimulationConfig& config() const { return config_; }
    
    /**
     * @brief Gets the current bodies in the simulation
     * @return Reference to bodies vector
     */
    const std::vector<core::Body>& bodies() const { return bodies_; }
    
    /**
     * @brief Resets the simulation state
     */
    void reset();
    
private:
    SimulationConfig config_;
    std::vector<core::Body> bodies_;
    std::unique_ptr<integrators::Integrator> integrator_;
    std::unique_ptr<io::DataWriter> data_writer_;
    std::unique_ptr<physics::CR3BPSolver> physics_solver_;
    integrators::DerivativeFunction derivative_func_;
    ProgressCallback progress_callback_;
    SimulationStats stats_;
    
    // Internal methods
    bool initialize();
    void cleanup();
    bool setup_integrator();
    bool setup_data_writer();
    bool validate_configuration() const;
    void update_progress(size_t step, size_t total_steps, double elapsed_time, double current_time);
    bool write_output_data(double current_time);
};

/**
 * @brief Convenience function to create and run a Lagrange point simulation
 * @param config Simulation configuration
 * @param perturbation_distance Distance to perturb from L4 point
 * @return true if simulation completed successfully
 */
bool run_lagrange_simulation(const SimulationConfig& config, 
                            double perturbation_distance = 1.5e7);

/**
 * @brief Convenience function to create and run a Cassini division simulation
 * @param config Simulation configuration
 * @param num_particles Number of test particles
 * @param min_radius Minimum particle radius
 * @param max_radius Maximum particle radius
 * @return true if simulation completed successfully
 */
bool run_cassini_simulation(const SimulationConfig& config,
                           int num_particles = 100,
                           double min_radius = 1.10e8,
                           double max_radius = 1.30e8);

}  // namespace orbital_mechanics::simulation