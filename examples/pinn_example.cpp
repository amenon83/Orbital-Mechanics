/**
 * @file pinn_example.cpp
 * @brief Example demonstrating PINN-based fluid dynamics simulation
 * 
 * This example shows how to use the PINN framework for simulating
 * fluid dynamics in Saturn's rings and demonstrates the key differences
 * between particle-based and continuum approaches.
 */

#include <orbital_mechanics/orbital_mechanics.hpp>
#include <orbital_mechanics/pinn/fluid_dynamics.hpp>
#include <orbital_mechanics/pinn/fluid_dynamics_impl.hpp>
#include <iostream>
#include <iomanip>

using namespace orbital_mechanics;
using namespace orbital_mechanics::pinn;

int main() {
    std::cout << "PINN Fluid Dynamics Example\n";
    std::cout << "===========================\n\n";
    
    try {
        // Initialize ring fluid parameters
        RingFluidParameters params;
        
        std::cout << "Ring System Parameters:\n";
        std::cout << "  Saturn mass: " << std::scientific << params.saturn_mass << " kg\n";
        std::cout << "  Mimas mass: " << std::scientific << params.mimas_mass << " kg\n";
        std::cout << "  Mimas orbital radius: " << std::scientific << params.mimas_orbital_radius << " m\n";
        std::cout << "  Mimas orbital period: " << std::fixed << std::setprecision(2) 
                  << params.mimas_orbital_period / 3600.0 << " hours\n";
        std::cout << "  Ring inner radius: " << std::scientific << params.ring_inner_radius << " m\n";
        std::cout << "  Ring outer radius: " << std::scientific << params.ring_outer_radius << " m\n";
        std::cout << "  Sound speed: " << std::fixed << params.sound_speed << " m/s\n";
        std::cout << "  Viscosity: " << std::scientific << params.viscosity << " m²/s\n\n";
        
        // Initialize fluid dynamics solver
        RingFluidDynamics fluid_dynamics(params);
        
        // Test point near 2:1 resonance
        double resonance_radius = std::pow(core::Constants::G * params.saturn_mass * 
                                         std::pow(params.mimas_orbital_period / 2.0, 2) / 
                                         (4.0 * core::Constants::PI * core::Constants::PI), 1.0/3.0);
        
        std::cout << "2:1 Resonance Analysis:\n";
        std::cout << "  Calculated resonance radius: " << std::scientific << resonance_radius << " m\n";
        std::cout << "  Resonance radius (km): " << std::fixed << std::setprecision(1) 
                  << resonance_radius / 1000.0 << " km\n";
        std::cout << "  Observed Cassini Division: ~118,000 km\n\n";
        
        // Test domain points
        std::vector<DomainPoint> test_points = {
            {1.1e8, 0.0, 0.0},  // Inner ring
            {resonance_radius, 0.0, 0.0},  // Resonance location
            {1.3e8, 0.0, 0.0},  // Outer ring
            {1.2e8, core::Constants::PI, params.mimas_orbital_period / 2.0}  // Different time/angle
        };
        
        std::cout << "Initial Fluid Conditions:\n";
        std::cout << "  Radius (km)    Density (kg/m²)  V_r (m/s)  V_θ (m/s)    Pressure (Pa)\n";
        std::cout << "  -----------    ---------------  ---------  ---------    -------------\n";
        
        for (const auto& point : test_points) {
            FluidState initial_state = fluid_dynamics.initial_conditions(point);
            
            std::cout << std::fixed << std::setprecision(1) << std::setw(12) << point.r / 1000.0
                      << std::setw(16) << std::setprecision(2) << initial_state.density
                      << std::setw(11) << std::setprecision(1) << initial_state.velocity_r
                      << std::setw(11) << std::setprecision(0) << initial_state.velocity_theta
                      << std::setw(13) << std::scientific << std::setprecision(2) << initial_state.pressure
                      << "\n";
        }
        
        std::cout << "\n";
        
        // Demonstrate PINN setup
        std::cout << "PINN Configuration:\n";
        std::vector<int> hidden_layers = {64, 64, 64, 64};
        FluidPINN pinn(params, hidden_layers);
        
        std::cout << "  Network architecture: [3";
        for (int layer : hidden_layers) {
            std::cout << ", " << layer;
        }
        std::cout << ", 4]\n";
        std::cout << "  Input: (r, θ, t) - normalized coordinates\n";
        std::cout << "  Output: (ρ, v_r, v_θ, p) - fluid state variables\n";
        std::cout << "  Activation: Hyperbolic tangent\n";
        std::cout << "  Total parameters: ~" << (3*64 + 64 + 64*64 + 64 + 64*64 + 64 + 64*64 + 64 + 64*4 + 4) << "\n\n";
        
        // Generate sample training points
        std::cout << "Training Data Generation:\n";
        auto collocation_points = pinn.generate_collocation_points(1000);
        auto boundary_points = pinn.generate_boundary_points(100);
        auto initial_points = pinn.generate_initial_points(100);
        
        std::cout << "  Collocation points: " << collocation_points.size() << " (interior domain)\n";
        std::cout << "  Boundary points: " << boundary_points.size() << " (ring edges)\n";
        std::cout << "  Initial points: " << initial_points.size() << " (t=0 conditions)\n\n";
        
        // Demonstrate physics-informed loss calculation
        std::cout << "Physics-Informed Loss Components:\n";
        double physics_loss = pinn.physics_loss(collocation_points);
        double boundary_loss = pinn.boundary_loss(boundary_points);
        double initial_loss = pinn.initial_loss(initial_points);
        
        std::cout << "  Physics loss: " << std::scientific << std::setprecision(3) << physics_loss 
                  << " (PDE residuals)\n";
        std::cout << "  Boundary loss: " << std::scientific << std::setprecision(3) << boundary_loss 
                  << " (edge conditions)\n";
        std::cout << "  Initial loss: " << std::scientific << std::setprecision(3) << initial_loss 
                  << " (t=0 conditions)\n";
        
        double total_loss = pinn.total_loss(collocation_points, boundary_points, initial_points);
        std::cout << "  Total loss: " << std::scientific << std::setprecision(3) << total_loss << "\n\n";
        
        // Demonstrate prediction
        std::cout << "PINN Predictions (untrained network):\n";
        std::cout << "  Radius (km)    Predicted ρ      Predicted v_r    Predicted v_θ    Predicted p\n";
        std::cout << "  -----------    -----------      -------------    -------------    -----------\n";
        
        for (const auto& point : test_points) {
            FluidState predicted = pinn.predict(point);
            
            std::cout << std::fixed << std::setprecision(1) << std::setw(12) << point.r / 1000.0
                      << std::setw(16) << std::setprecision(2) << predicted.density
                      << std::setw(17) << std::setprecision(1) << predicted.velocity_r
                      << std::setw(17) << std::setprecision(0) << predicted.velocity_theta
                      << std::setw(13) << std::scientific << std::setprecision(2) << predicted.pressure
                      << "\n";
        }
        
        std::cout << "\n";
        
        // Comparison with particle approach
        std::cout << "Method Comparison:\n";
        std::cout << "==================\n";
        std::cout << "Particle-based Simulation:\n";
        std::cout << "  + Physically intuitive\n";
        std::cout << "  + Captures individual particle dynamics\n";
        std::cout << "  + Well-established numerical methods\n";
        std::cout << "  - Computationally expensive for large N\n";
        std::cout << "  - Discrete nature limits resolution\n";
        std::cout << "  - Difficult to enforce conservation laws exactly\n\n";
        
        std::cout << "PINN-based Simulation:\n";
        std::cout << "  + Continuous representation\n";
        std::cout << "  + Automatic satisfaction of physics constraints\n";
        std::cout << "  + Efficient for large-scale problems\n";
        std::cout << "  + Differentiable and invertible\n";
        std::cout << "  - Requires careful network design\n";
        std::cout << "  - Training can be challenging\n";
        std::cout << "  - Less interpretable than particle methods\n\n";
        
        std::cout << "Key Innovations:\n";
        std::cout << "• Physics-informed loss functions enforce PDEs\n";
        std::cout << "• Automatic differentiation for gradient computation\n";
        std::cout << "• Mesh-free approach eliminates discretization errors\n";
        std::cout << "• Neural network approximates solution globally\n";
        std::cout << "• Hybrid C++/Python implementation for performance\n\n";
        
        std::cout << "Applications:\n";
        std::cout << "• Cassini Division formation dynamics\n";
        std::cout << "• Real-time ring evolution predictions\n";
        std::cout << "• Parameter estimation from observations\n";
        std::cout << "• Optimization of ring stability conditions\n";
        std::cout << "• Design of future ring exploration missions\n\n";
        
        std::cout << "Example completed successfully!\n";
        std::cout << "To run the full PINN simulation, use: ./pinn_cassini_simulation\n";
        
        return EXIT_SUCCESS;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
}