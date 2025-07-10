/**
 * @file test_integrators.cpp
 * @brief Unit tests for numerical integrators
 */

#include <gtest/gtest.h>
#include <orbital_mechanics/integrators/integrator.hpp>
#include <orbital_mechanics/integrators/integrator_impl.hpp>
#include <cmath>

using namespace orbital_mechanics::integrators;
using namespace orbital_mechanics::core;

class IntegratorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Simple harmonic oscillator: x'' = -k*x, with k=1
        // Analytical solution: x(t) = A*cos(t) + B*sin(t)
        // For x(0) = 1, x'(0) = 0: x(t) = cos(t)
        derivative_func = [](const State& state, double /*mass*/, double /*time*/, void* /*user_data*/) {
            return Derivative(state.velocity, Vector2d(-state.position.x, -state.position.y));
        };
        
        initial_state = State(Vector2d(1.0, 0.0), Vector2d(0.0, 1.0));
        dt = 0.01;
        mass = 1.0;
        tolerance = 1e-3;
    }
    
    DerivativeFunction derivative_func;
    State initial_state;
    double dt, mass, tolerance;
};

TEST_F(IntegratorTest, EulerIntegrator) {
    auto integrator = std::make_unique<EulerIntegrator>();
    
    EXPECT_EQ(integrator->name(), "Euler");
    EXPECT_EQ(integrator->order(), 1);
    EXPECT_FALSE(integrator->is_symplectic());
    
    State state = initial_state;
    double time = 0.0;
    
    // Integrate for one step
    state = integrator->integrate_step(state, mass, time, dt, derivative_func);
    
    // Check that the state has changed
    EXPECT_NE(state.position.x, initial_state.position.x);
    EXPECT_NE(state.velocity.x, initial_state.velocity.x);
    
    // For a small time step, should be reasonably close to analytical solution
    double expected_x = std::cos(dt);
    double expected_vx = -std::sin(dt);
    EXPECT_NEAR(state.position.x, expected_x, 0.1);  // Euler is not very accurate
    EXPECT_NEAR(state.velocity.x, expected_vx, 0.1);
}

TEST_F(IntegratorTest, RK4Integrator) {
    auto integrator = std::make_unique<RK4Integrator>();
    
    EXPECT_EQ(integrator->name(), "RK4");
    EXPECT_EQ(integrator->order(), 4);
    EXPECT_FALSE(integrator->is_symplectic());
    
    State state = initial_state;
    double time = 0.0;
    
    // Integrate for one step
    state = integrator->integrate_step(state, mass, time, dt, derivative_func);
    
    // RK4 should be much more accurate than Euler
    double expected_x = std::cos(dt);
    double expected_vx = -std::sin(dt);
    EXPECT_NEAR(state.position.x, expected_x, 1e-6);
    EXPECT_NEAR(state.velocity.x, expected_vx, 1e-6);
}

TEST_F(IntegratorTest, VerletIntegrator) {
    auto integrator = std::make_unique<VerletIntegrator>();
    
    EXPECT_EQ(integrator->name(), "Verlet");
    EXPECT_EQ(integrator->order(), 2);
    EXPECT_TRUE(integrator->is_symplectic());
    
    State state = initial_state;
    double time = 0.0;
    
    // Integrate for one step
    state = integrator->integrate_step(state, mass, time, dt, derivative_func);
    
    // Verlet should be reasonably accurate
    double expected_x = std::cos(dt);
    double expected_vx = -std::sin(dt);
    EXPECT_NEAR(state.position.x, expected_x, 1e-4);
    EXPECT_NEAR(state.velocity.x, expected_vx, 1e-4);
}

TEST_F(IntegratorTest, IntegratorFactory) {
    auto euler = create_integrator("euler");
    EXPECT_EQ(euler->name(), "Euler");
    
    auto rk4 = create_integrator("rk4");
    EXPECT_EQ(rk4->name(), "RK4");
    
    auto verlet = create_integrator("verlet");
    EXPECT_EQ(verlet->name(), "Verlet");
    
    // Test case insensitivity
    auto euler_caps = create_integrator("EULER");
    EXPECT_EQ(euler_caps->name(), "Euler");
    
    // Test invalid integrator type
    EXPECT_THROW(create_integrator("invalid"), std::invalid_argument);
}

TEST_F(IntegratorTest, EnergyConservation) {
    // Test energy conservation for harmonic oscillator
    auto rk4 = create_integrator("rk4");
    auto verlet = create_integrator("verlet");
    
    State state_rk4 = initial_state;
    State state_verlet = initial_state;
    
    // Calculate initial energy
    double initial_energy = 0.5 * (state_rk4.velocity.magnitude_squared() + 
                                  state_rk4.position.magnitude_squared());
    
    // Integrate for many steps
    const int num_steps = 1000;
    for (int i = 0; i < num_steps; ++i) {
        double time = i * dt;
        state_rk4 = rk4->integrate_step(state_rk4, mass, time, dt, derivative_func);
        state_verlet = verlet->integrate_step(state_verlet, mass, time, dt, derivative_func);
    }
    
    // Calculate final energies
    double final_energy_rk4 = 0.5 * (state_rk4.velocity.magnitude_squared() + 
                                    state_rk4.position.magnitude_squared());
    double final_energy_verlet = 0.5 * (state_verlet.velocity.magnitude_squared() + 
                                       state_verlet.position.magnitude_squared());
    
    // Verlet should conserve energy better than RK4
    double energy_error_rk4 = std::abs(final_energy_rk4 - initial_energy) / initial_energy;
    double energy_error_verlet = std::abs(final_energy_verlet - initial_energy) / initial_energy;
    
    EXPECT_LT(energy_error_verlet, energy_error_rk4);
    EXPECT_LT(energy_error_verlet, 0.01);  // Should be less than 1% error
}

TEST_F(IntegratorTest, AccuracyComparison) {
    // Compare accuracy of different integrators for harmonic oscillator
    auto euler = create_integrator("euler");
    auto rk4 = create_integrator("rk4");
    
    State state_euler = initial_state;
    State state_rk4 = initial_state;
    
    const double final_time = 1.0;
    const int num_steps = static_cast<int>(final_time / dt);
    
    // Integrate to final time
    for (int i = 0; i < num_steps; ++i) {
        double time = i * dt;
        state_euler = euler->integrate_step(state_euler, mass, time, dt, derivative_func);
        state_rk4 = rk4->integrate_step(state_rk4, mass, time, dt, derivative_func);
    }
    
    // Compare with analytical solution
    double analytical_x = std::cos(final_time);
    double analytical_vx = -std::sin(final_time);
    
    double error_euler = std::abs(state_euler.position.x - analytical_x);
    double error_rk4 = std::abs(state_rk4.position.x - analytical_x);
    
    // RK4 should be more accurate than Euler
    EXPECT_LT(error_rk4, error_euler);
    EXPECT_LT(error_rk4, 1e-5);
}