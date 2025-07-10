/**
 * @file test_simulation.cpp
 * @brief Unit tests for simulation framework
 */

#include <gtest/gtest.h>
#include <orbital_mechanics/simulation/simulation.hpp>
#include <orbital_mechanics/simulation/simulation_impl.hpp>
#include <orbital_mechanics/physics/three_body.hpp>
#include <orbital_mechanics/physics/three_body_impl.hpp>
#include <fstream>
#include <filesystem>

using namespace orbital_mechanics::simulation;
using namespace orbital_mechanics::physics;
using namespace orbital_mechanics::core;

class SimulationTest : public ::testing::Test {
protected:
    void SetUp() override {
        config.time_step = 0.1;
        config.total_time = 1.0;
        config.output_interval = 1;
        config.integrator_type = "rk4";
        config.output_format = "text";
        config.output_filename = "test_output.txt";
        config.write_velocities = false;
        config.console_update_interval = 10;
        
        // Clean up any existing test files
        cleanup_test_files();
    }
    
    void TearDown() override {
        cleanup_test_files();
    }
    
    void cleanup_test_files() {
        std::filesystem::remove("test_output.txt");
        std::filesystem::remove("test_output.csv");
    }
    
    SimulationConfig config;
};

TEST_F(SimulationTest, ConfigValidation) {
    EXPECT_TRUE(config.validate());
    
    // Test invalid configurations
    SimulationConfig invalid_config = config;
    invalid_config.time_step = -1.0;
    EXPECT_FALSE(invalid_config.validate());
    
    invalid_config = config;
    invalid_config.total_time = 0.0;
    EXPECT_FALSE(invalid_config.validate());
    
    invalid_config = config;
    invalid_config.output_interval = 0;
    EXPECT_FALSE(invalid_config.validate());
    
    invalid_config = config;
    invalid_config.integrator_type = "";
    EXPECT_FALSE(invalid_config.validate());
}

TEST_F(SimulationTest, SimulationConstruction) {
    // Valid configuration should work
    EXPECT_NO_THROW(Simulation sim(config));
    
    // Invalid configuration should throw
    SimulationConfig invalid_config = config;
    invalid_config.time_step = -1.0;
    EXPECT_THROW(Simulation sim(invalid_config), std::invalid_argument);
}

TEST_F(SimulationTest, BodyManagement) {
    Simulation sim(config);
    
    // Initially no bodies
    EXPECT_EQ(sim.bodies().size(), 0);
    
    // Add a body
    Body body1(1.0, Vector2d(1.0, 0.0), Vector2d(0.0, 1.0), "Body1");
    sim.add_body(body1);
    EXPECT_EQ(sim.bodies().size(), 1);
    EXPECT_EQ(sim.bodies()[0].name(), "Body1");
    
    // Add multiple bodies
    std::vector<Body> bodies;
    bodies.emplace_back(2.0, Vector2d(2.0, 0.0), Vector2d(0.0, 2.0), "Body2");
    bodies.emplace_back(3.0, Vector2d(3.0, 0.0), Vector2d(0.0, 3.0), "Body3");
    sim.add_bodies(bodies);
    
    EXPECT_EQ(sim.bodies().size(), 3);
    EXPECT_EQ(sim.bodies()[1].name(), "Body2");
    EXPECT_EQ(sim.bodies()[2].name(), "Body3");
}

TEST_F(SimulationTest, PhysicsSolverSetup) {
    Simulation sim(config);
    
    auto solver = std::make_unique<CR3BPSolver>(systems::sun_earth());
    sim.set_physics_solver(std::move(solver));
    
    // Should be able to run simulation now (though it will fail without bodies)
    // This tests that the solver was set properly
}

TEST_F(SimulationTest, SimpleSimulationRun) {
    // Create a simple simulation with a single body
    config.total_time = 0.1;  // Very short simulation
    config.output_interval = 1;
    
    Simulation sim(config);
    
    // Add a simple body
    Body body(1.0, Vector2d(1.0, 0.0), Vector2d(0.0, 1.0), "TestBody");
    sim.add_body(body);
    
    // Set a simple derivative function (harmonic oscillator)
    auto derivative_func = [](const integrators::State& state, double /*mass*/, double /*time*/, void* /*user_data*/) {
        return integrators::Derivative(state.velocity, Vector2d(-state.position.x, -state.position.y));
    };
    sim.set_derivative_function(derivative_func);
    
    // Run simulation
    bool success = sim.run();
    EXPECT_TRUE(success);
    
    // Check statistics
    const auto& stats = sim.stats();
    EXPECT_GT(stats.total_steps, 0);
    EXPECT_GT(stats.output_points, 0);
    EXPECT_GT(stats.elapsed_time_seconds, 0.0);
    
    // Check that output file was created
    EXPECT_TRUE(std::filesystem::exists("test_output.txt"));
}

TEST_F(SimulationTest, ProgressCallback) {
    config.total_time = 0.1;
    config.console_update_interval = 1;
    
    Simulation sim(config);
    
    Body body(1.0, Vector2d(1.0, 0.0), Vector2d(0.0, 1.0), "TestBody");
    sim.add_body(body);
    
    auto derivative_func = [](const integrators::State& state, double /*mass*/, double /*time*/, void* /*user_data*/) {
        return integrators::Derivative(state.velocity, Vector2d(-state.position.x, -state.position.y));
    };
    sim.set_derivative_function(derivative_func);
    
    // Set up progress callback
    int callback_count = 0;
    auto progress_callback = [&callback_count](size_t /*step*/, size_t /*total*/, double /*elapsed*/, double /*current*/) {
        callback_count++;
    };
    sim.set_progress_callback(progress_callback);
    
    bool success = sim.run();
    EXPECT_TRUE(success);
    EXPECT_GT(callback_count, 0);
}

TEST_F(SimulationTest, SimulationReset) {
    Simulation sim(config);
    
    Body body(1.0, Vector2d(1.0, 0.0), Vector2d(0.0, 1.0), "TestBody");
    sim.add_body(body);
    
    EXPECT_EQ(sim.bodies().size(), 1);
    
    sim.reset();
    EXPECT_EQ(sim.bodies().size(), 0);
}

TEST_F(SimulationTest, OutputFormats) {
    config.total_time = 0.1;
    config.output_interval = 1;
    
    // Test text format
    config.output_format = "text";
    config.output_filename = "test_output.txt";
    
    Simulation sim_text(config);
    Body body(1.0, Vector2d(1.0, 0.0), Vector2d(0.0, 1.0), "TestBody");
    sim_text.add_body(body);
    
    auto derivative_func = [](const integrators::State& state, double /*mass*/, double /*time*/, void* /*user_data*/) {
        return integrators::Derivative(state.velocity, Vector2d(-state.position.x, -state.position.y));
    };
    sim_text.set_derivative_function(derivative_func);
    
    EXPECT_TRUE(sim_text.run());
    EXPECT_TRUE(std::filesystem::exists("test_output.txt"));
    
    // Test CSV format
    config.output_format = "csv";
    config.output_filename = "test_output.csv";
    
    Simulation sim_csv(config);
    sim_csv.add_body(body);
    sim_csv.set_derivative_function(derivative_func);
    
    EXPECT_TRUE(sim_csv.run());
    EXPECT_TRUE(std::filesystem::exists("test_output.csv"));
}

TEST_F(SimulationTest, ConvenienceFunctions) {
    config.total_time = 0.1;
    config.output_filename = "lagrange_test.txt";
    
    // Test Lagrange simulation
    bool success = run_lagrange_simulation(config, 1e6);
    EXPECT_TRUE(success);
    EXPECT_TRUE(std::filesystem::exists("lagrange_test.txt"));
    
    std::filesystem::remove("lagrange_test.txt");
    
    // Test Cassini simulation
    config.output_filename = "cassini_test.txt";
    success = run_cassini_simulation(config, 10, 1e8, 1.5e8);
    EXPECT_TRUE(success);
    EXPECT_TRUE(std::filesystem::exists("cassini_test.txt"));
    
    std::filesystem::remove("cassini_test.txt");
}

TEST_F(SimulationTest, ErrorHandling) {
    // Test simulation without bodies
    Simulation sim(config);
    
    auto derivative_func = [](const integrators::State& state, double /*mass*/, double /*time*/, void* /*user_data*/) {
        return integrators::Derivative(state.velocity, Vector2d(-state.position.x, -state.position.y));
    };
    sim.set_derivative_function(derivative_func);
    
    // Should fail because no bodies were added
    bool success = sim.run();
    EXPECT_FALSE(success);
    
    // Test simulation without derivative function
    Simulation sim2(config);
    Body body(1.0, Vector2d(1.0, 0.0), Vector2d(0.0, 1.0), "TestBody");
    sim2.add_body(body);
    
    // Should fail because no derivative function was set
    success = sim2.run();
    EXPECT_FALSE(success);
}