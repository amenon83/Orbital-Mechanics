/**
 * @file test_body.cpp
 * @brief Unit tests for Body class
 */

#include <gtest/gtest.h>
#include <orbital_mechanics/core/body.hpp>
#include <orbital_mechanics/core/constants.hpp>
#include <stdexcept>

using namespace orbital_mechanics::core;

class BodyTest : public ::testing::Test {
protected:
    void SetUp() override {
        pos = Vector2d(1.0, 2.0);
        vel = Vector2d(3.0, 4.0);
        mass = 5.0;
        name = "TestBody";
        body = std::make_unique<Body>(mass, pos, vel, name);
    }
    
    Vector2d pos, vel;
    double mass;
    std::string name;
    std::unique_ptr<Body> body;
    const double tolerance = 1e-10;
};

TEST_F(BodyTest, ConstructorTest) {
    EXPECT_DOUBLE_EQ(body->mass(), mass);
    EXPECT_EQ(body->position(), pos);
    EXPECT_EQ(body->velocity(), vel);
    EXPECT_EQ(body->name(), name);
    
    // Test individual component getters
    EXPECT_DOUBLE_EQ(body->x(), 1.0);
    EXPECT_DOUBLE_EQ(body->y(), 2.0);
    EXPECT_DOUBLE_EQ(body->vx(), 3.0);
    EXPECT_DOUBLE_EQ(body->vy(), 4.0);
}

TEST_F(BodyTest, ConstructorWithInvalidMass) {
    EXPECT_THROW(Body(-1.0, pos, vel), std::invalid_argument);
    EXPECT_THROW(Body(0.0, pos, vel), std::invalid_argument);
}

TEST_F(BodyTest, StateSetters) {
    Vector2d new_pos(5.0, 6.0);
    Vector2d new_vel(7.0, 8.0);
    
    body->set_position(new_pos);
    EXPECT_EQ(body->position(), new_pos);
    
    body->set_velocity(new_vel);
    EXPECT_EQ(body->velocity(), new_vel);
    
    Vector2d another_pos(9.0, 10.0);
    Vector2d another_vel(11.0, 12.0);
    body->set_state(another_pos, another_vel);
    EXPECT_EQ(body->position(), another_pos);
    EXPECT_EQ(body->velocity(), another_vel);
}

TEST_F(BodyTest, KineticEnergy) {
    double expected_ke = 0.5 * mass * vel.magnitude_squared();
    EXPECT_DOUBLE_EQ(body->kinetic_energy(), expected_ke);
    
    // Test with zero velocity
    body->set_velocity(Vector2d(0.0, 0.0));
    EXPECT_DOUBLE_EQ(body->kinetic_energy(), 0.0);
}

TEST_F(BodyTest, DistanceCalculations) {
    Body other_body(1.0, Vector2d(4.0, 6.0), Vector2d(0.0, 0.0));
    
    double distance = body->distance_to(other_body);
    double expected_distance = std::sqrt(9.0 + 16.0);  // sqrt((4-1)^2 + (6-2)^2)
    EXPECT_DOUBLE_EQ(distance, expected_distance);
    
    double distance_sq = body->distance_squared_to(other_body);
    EXPECT_DOUBLE_EQ(distance_sq, 25.0);
    
    Vector2d displacement = body->displacement_to(other_body);
    EXPECT_EQ(displacement, Vector2d(3.0, 4.0));
}

TEST_F(BodyTest, StateUpdates) {
    double dt = 0.1;
    Vector2d initial_pos = body->position();
    Vector2d initial_vel = body->velocity();
    
    // Test position update
    body->update_position(dt);
    Vector2d expected_pos = initial_pos + initial_vel * dt;
    EXPECT_EQ(body->position(), expected_pos);
    EXPECT_EQ(body->velocity(), initial_vel);  // Velocity should be unchanged
    
    // Reset position
    body->set_position(initial_pos);
    
    // Test velocity update
    Vector2d acceleration(2.0, 3.0);
    body->update_velocity(acceleration, dt);
    Vector2d expected_vel = initial_vel + acceleration * dt;
    EXPECT_EQ(body->velocity(), expected_vel);
    EXPECT_EQ(body->position(), initial_pos);  // Position should be unchanged
    
    // Reset velocity
    body->set_velocity(initial_vel);
    
    // Test combined state update
    body->update_state(acceleration, dt);
    EXPECT_EQ(body->velocity(), initial_vel + acceleration * dt);
    EXPECT_EQ(body->position(), initial_pos + (initial_vel + acceleration * dt) * dt);
}

TEST_F(BodyTest, PositionChecking) {
    Vector2d test_pos(1.0, 2.0);
    EXPECT_TRUE(body->is_at_position(test_pos));
    EXPECT_FALSE(body->is_at_position(Vector2d(5.0, 6.0)));
    
    // Test with tolerance
    Vector2d near_pos(1.0 + 1e-11, 2.0 + 1e-11);
    EXPECT_TRUE(body->is_at_position(near_pos, 1e-10));
    EXPECT_FALSE(body->is_at_position(near_pos, 1e-12));
}

TEST_F(BodyTest, ToString) {
    std::string str = body->to_string();
    EXPECT_NE(str.find("TestBody"), std::string::npos);
    EXPECT_NE(str.find("5"), std::string::npos);  // mass
}

TEST_F(BodyTest, FactoryFunctions) {
    Body origin_body = make_body_at_origin(10.0, "Origin");
    EXPECT_DOUBLE_EQ(origin_body.mass(), 10.0);
    EXPECT_EQ(origin_body.position(), Vector2d(0.0, 0.0));
    EXPECT_EQ(origin_body.velocity(), Vector2d(0.0, 0.0));
    EXPECT_EQ(origin_body.name(), "Origin");
    
    // Test circular orbit body
    double radius = 1.0;
    double central_mass = Constants::EARTH_MASS;
    Body orbit_body = make_circular_orbit_body(1.0, radius, central_mass, 0.0, "Orbiter");
    
    EXPECT_DOUBLE_EQ(orbit_body.mass(), 1.0);
    EXPECT_NEAR(orbit_body.position().magnitude(), radius, tolerance);
    
    // Check that orbital velocity is approximately correct
    double expected_orbital_speed = std::sqrt(Constants::G * central_mass / radius);
    EXPECT_NEAR(orbit_body.velocity().magnitude(), expected_orbital_speed, tolerance);
}

TEST_F(BodyTest, FactoryFunctionValidation) {
    EXPECT_THROW(make_circular_orbit_body(-1.0, 1.0, 1.0), std::invalid_argument);
    EXPECT_THROW(make_circular_orbit_body(1.0, -1.0, 1.0), std::invalid_argument);
    EXPECT_THROW(make_circular_orbit_body(1.0, 1.0, -1.0), std::invalid_argument);
}