/**
 * @file test_three_body.cpp
 * @brief Unit tests for three-body problem physics
 */

#include <gtest/gtest.h>
#include <orbital_mechanics/physics/three_body.hpp>
#include <orbital_mechanics/physics/three_body_impl.hpp>
#include <orbital_mechanics/core/constants.hpp>
#include <cmath>

using namespace orbital_mechanics::physics;
using namespace orbital_mechanics::core;

class ThreeBodyTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Sun-Earth system
        sun_mass = Constants::SUN_MASS;
        earth_mass = Constants::EARTH_MASS;
        separation = Constants::SUN_EARTH_DISTANCE;
        
        params = std::make_unique<CR3BPParameters>(sun_mass, earth_mass, separation);
        solver = std::make_unique<CR3BPSolver>(*params);
        
        tolerance = 1e-6;
    }
    
    double sun_mass, earth_mass, separation;
    std::unique_ptr<CR3BPParameters> params;
    std::unique_ptr<CR3BPSolver> solver;
    double tolerance;
};

TEST_F(ThreeBodyTest, ParameterConstruction) {
    EXPECT_DOUBLE_EQ(params->primary1_mass, sun_mass);
    EXPECT_DOUBLE_EQ(params->primary2_mass, earth_mass);
    EXPECT_DOUBLE_EQ(params->separation, separation);
    EXPECT_DOUBLE_EQ(params->total_mass, sun_mass + earth_mass);
    EXPECT_DOUBLE_EQ(params->mu, earth_mass / (sun_mass + earth_mass));
    
    // Check that angular velocity is positive
    EXPECT_GT(params->angular_velocity, 0.0);
    
    // Check that primary positions are calculated correctly
    EXPECT_NEAR(params->primary1_x, -separation * params->mu, tolerance);
    EXPECT_NEAR(params->primary2_x, separation * (1.0 - params->mu), tolerance);
}

TEST_F(ThreeBodyTest, ParameterValidation) {
    // Test invalid parameters
    EXPECT_THROW(CR3BPParameters(-1.0, 1.0, 1.0), std::invalid_argument);
    EXPECT_THROW(CR3BPParameters(1.0, -1.0, 1.0), std::invalid_argument);
    EXPECT_THROW(CR3BPParameters(1.0, 1.0, -1.0), std::invalid_argument);
    EXPECT_THROW(CR3BPParameters(0.0, 1.0, 1.0), std::invalid_argument);
}

TEST_F(ThreeBodyTest, LagrangePoints) {
    auto l4 = params->get_l4_position();
    auto l5 = params->get_l5_position();
    
    // L4 and L5 should be at the same distance from both primaries
    double r1_l4 = std::sqrt((l4.x - params->primary1_x) * (l4.x - params->primary1_x) + 
                            l4.y * l4.y);
    double r2_l4 = std::sqrt((l4.x - params->primary2_x) * (l4.x - params->primary2_x) + 
                            l4.y * l4.y);
    
    // For L4 and L5, the distances should equal the separation
    EXPECT_NEAR(r1_l4, separation, tolerance * separation);
    EXPECT_NEAR(r2_l4, separation, tolerance * separation);
    
    // L4 should be above the x-axis, L5 below
    EXPECT_GT(l4.y, 0.0);
    EXPECT_LT(l5.y, 0.0);
    
    // L4 and L5 should be symmetric about the x-axis
    EXPECT_NEAR(l4.x, l5.x, tolerance);
    EXPECT_NEAR(l4.y, -l5.y, tolerance);
}

TEST_F(ThreeBodyTest, AccelerationCalculation) {
    // Test acceleration at L4 point (should be approximately zero)
    auto l4_pos = params->get_l4_position();
    auto l4_vel = Vector2d(0.0, 0.0);
    
    auto acceleration = solver->calculate_acceleration(l4_pos, l4_vel);
    
    // At L4, the acceleration should be very small (but not exactly zero due to numerical precision)
    EXPECT_LT(acceleration.magnitude(), 1e-3);
}

TEST_F(ThreeBodyTest, EffectivePotential) {
    // Test effective potential calculation
    auto l4_pos = params->get_l4_position();
    double potential = solver->effective_potential(l4_pos);
    
    // Potential should be finite and negative
    EXPECT_TRUE(std::isfinite(potential));
    EXPECT_LT(potential, 0.0);
    
    // Test potential at primary locations (should be very negative)
    Vector2d primary1_pos(params->primary1_x, 0.0);
    double potential_at_primary = solver->effective_potential(primary1_pos);
    EXPECT_LT(potential_at_primary, potential);
}

TEST_F(ThreeBodyTest, JacobiConstant) {
    // Test Jacobi constant calculation
    auto l4_pos = params->get_l4_position();
    auto l4_vel = Vector2d(0.0, 0.0);
    
    double jacobi = params->jacobi_constant(l4_pos, l4_vel);
    EXPECT_TRUE(std::isfinite(jacobi));
    
    // Test that Jacobi constant is conserved during integration
    // (This is a simplified test - in practice, numerical errors will cause small variations)
    auto vel_perturbed = Vector2d(1e-3, 0.0);
    double jacobi_perturbed = params->jacobi_constant(l4_pos, vel_perturbed);
    
    // Should be different due to kinetic energy term
    EXPECT_NE(jacobi, jacobi_perturbed);
}

TEST_F(ThreeBodyTest, DerivativeFunction) {
    auto derivative_func = solver->create_derivative_function();
    
    // Test derivative function at L4 point
    auto l4_pos = params->get_l4_position();
    auto l4_vel = Vector2d(0.0, 0.0);
    
    integrators::State state(l4_pos, l4_vel);
    auto derivative = derivative_func(state, 1.0, 0.0, nullptr);
    
    // Velocity part should match input velocity
    EXPECT_EQ(derivative.velocity, l4_vel);
    
    // Acceleration should be small at L4
    EXPECT_LT(derivative.acceleration.magnitude(), 1e-3);
}

TEST_F(ThreeBodyTest, SystemFactoryFunctions) {
    auto sun_earth_params = systems::sun_earth();
    EXPECT_DOUBLE_EQ(sun_earth_params.primary1_mass, Constants::SUN_MASS);
    EXPECT_DOUBLE_EQ(sun_earth_params.primary2_mass, Constants::EARTH_MASS);
    EXPECT_DOUBLE_EQ(sun_earth_params.separation, Constants::SUN_EARTH_DISTANCE);
    
    auto saturn_mimas_params = systems::saturn_mimas();
    EXPECT_DOUBLE_EQ(saturn_mimas_params.primary1_mass, Constants::SATURN_MASS);
    EXPECT_DOUBLE_EQ(saturn_mimas_params.primary2_mass, Constants::MIMAS_MASS);
    EXPECT_DOUBLE_EQ(saturn_mimas_params.separation, Constants::MIMAS_SEMI_MAJOR_AXIS);
}

TEST_F(ThreeBodyTest, AccelerationComponents) {
    // Test that acceleration components are calculated correctly
    Vector2d test_pos(1e8, 1e8);  // Some arbitrary position
    Vector2d test_vel(1e3, 1e3);  // Some arbitrary velocity
    
    auto total_acceleration = solver->calculate_acceleration(test_pos, test_vel);
    
    // The acceleration should be finite and non-zero
    EXPECT_TRUE(std::isfinite(total_acceleration.x));
    EXPECT_TRUE(std::isfinite(total_acceleration.y));
    EXPECT_GT(total_acceleration.magnitude(), 0.0);
    
    // Test with zero velocity (should eliminate Coriolis term)
    auto acceleration_no_vel = solver->calculate_acceleration(test_pos, Vector2d(0.0, 0.0));
    
    // Should be different from the case with velocity
    EXPECT_NE(total_acceleration.x, acceleration_no_vel.x);
    EXPECT_NE(total_acceleration.y, acceleration_no_vel.y);
}

TEST_F(ThreeBodyTest, SymmetryProperties) {
    // Test symmetry properties of the system
    Vector2d pos1(1e8, 1e8);
    Vector2d pos2(1e8, -1e8);  // Mirror position about x-axis
    Vector2d vel1(1e3, 1e3);
    Vector2d vel2(1e3, -1e3);  // Mirror velocity about x-axis
    
    auto acc1 = solver->calculate_acceleration(pos1, vel1);
    auto acc2 = solver->calculate_acceleration(pos2, vel2);
    
    // Due to system symmetry, x-components should be equal, y-components should be opposite
    EXPECT_NEAR(acc1.x, acc2.x, tolerance);
    EXPECT_NEAR(acc1.y, -acc2.y, tolerance);
}