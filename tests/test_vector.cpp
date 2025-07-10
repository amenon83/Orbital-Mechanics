/**
 * @file test_vector.cpp
 * @brief Unit tests for Vector2 class
 */

#include <gtest/gtest.h>
#include <orbital_mechanics/core/vector.hpp>
#include <cmath>

using namespace orbital_mechanics::core;

class Vector2Test : public ::testing::Test {
protected:
    void SetUp() override {
        v1 = Vector2d(3.0, 4.0);
        v2 = Vector2d(1.0, 2.0);
        zero = Vector2d(0.0, 0.0);
    }
    
    Vector2d v1, v2, zero;
    const double tolerance = 1e-10;
};

TEST_F(Vector2Test, ConstructorTest) {
    Vector2d v(3.0, 4.0);
    EXPECT_DOUBLE_EQ(v.x, 3.0);
    EXPECT_DOUBLE_EQ(v.y, 4.0);
    
    Vector2d v_default;
    EXPECT_DOUBLE_EQ(v_default.x, 0.0);
    EXPECT_DOUBLE_EQ(v_default.y, 0.0);
}

TEST_F(Vector2Test, ArithmeticOperators) {
    Vector2d result = v1 + v2;
    EXPECT_DOUBLE_EQ(result.x, 4.0);
    EXPECT_DOUBLE_EQ(result.y, 6.0);
    
    result = v1 - v2;
    EXPECT_DOUBLE_EQ(result.x, 2.0);
    EXPECT_DOUBLE_EQ(result.y, 2.0);
    
    result = v1 * 2.0;
    EXPECT_DOUBLE_EQ(result.x, 6.0);
    EXPECT_DOUBLE_EQ(result.y, 8.0);
    
    result = v1 / 2.0;
    EXPECT_DOUBLE_EQ(result.x, 1.5);
    EXPECT_DOUBLE_EQ(result.y, 2.0);
}

TEST_F(Vector2Test, CompoundAssignmentOperators) {
    Vector2d v = v1;
    v += v2;
    EXPECT_DOUBLE_EQ(v.x, 4.0);
    EXPECT_DOUBLE_EQ(v.y, 6.0);
    
    v = v1;
    v -= v2;
    EXPECT_DOUBLE_EQ(v.x, 2.0);
    EXPECT_DOUBLE_EQ(v.y, 2.0);
    
    v = v1;
    v *= 2.0;
    EXPECT_DOUBLE_EQ(v.x, 6.0);
    EXPECT_DOUBLE_EQ(v.y, 8.0);
    
    v = v1;
    v /= 2.0;
    EXPECT_DOUBLE_EQ(v.x, 1.5);
    EXPECT_DOUBLE_EQ(v.y, 2.0);
}

TEST_F(Vector2Test, ComparisonOperators) {
    Vector2d v1_copy(3.0, 4.0);
    EXPECT_TRUE(v1 == v1_copy);
    EXPECT_FALSE(v1 != v1_copy);
    EXPECT_FALSE(v1 == v2);
    EXPECT_TRUE(v1 != v2);
}

TEST_F(Vector2Test, DotProduct) {
    double dot = v1.dot(v2);
    EXPECT_DOUBLE_EQ(dot, 11.0);  // 3*1 + 4*2 = 11
    
    dot = v1.dot(zero);
    EXPECT_DOUBLE_EQ(dot, 0.0);
}

TEST_F(Vector2Test, CrossProduct) {
    double cross = v1.cross(v2);
    EXPECT_DOUBLE_EQ(cross, 2.0);  // 3*2 - 4*1 = 2
    
    cross = v1.cross(zero);
    EXPECT_DOUBLE_EQ(cross, 0.0);
}

TEST_F(Vector2Test, Magnitude) {
    double mag_sq = v1.magnitude_squared();
    EXPECT_DOUBLE_EQ(mag_sq, 25.0);  // 3^2 + 4^2 = 25
    
    double mag = v1.magnitude();
    EXPECT_DOUBLE_EQ(mag, 5.0);
    
    EXPECT_DOUBLE_EQ(zero.magnitude(), 0.0);
}

TEST_F(Vector2Test, Normalization) {
    Vector2d normalized = v1.normalized();
    EXPECT_NEAR(normalized.x, 0.6, tolerance);
    EXPECT_NEAR(normalized.y, 0.8, tolerance);
    EXPECT_NEAR(normalized.magnitude(), 1.0, tolerance);
    
    // Test zero vector normalization
    Vector2d zero_normalized = zero.normalized();
    EXPECT_DOUBLE_EQ(zero_normalized.x, 0.0);
    EXPECT_DOUBLE_EQ(zero_normalized.y, 0.0);
}

TEST_F(Vector2Test, Perpendicular) {
    Vector2d perp = v1.perpendicular();
    EXPECT_DOUBLE_EQ(perp.x, -4.0);
    EXPECT_DOUBLE_EQ(perp.y, 3.0);
    
    // Check that it's actually perpendicular
    EXPECT_NEAR(v1.dot(perp), 0.0, tolerance);
}

TEST_F(Vector2Test, UtilityFunctions) {
    EXPECT_TRUE(zero.is_zero());
    EXPECT_FALSE(v1.is_zero());
    
    EXPECT_TRUE(zero.is_near_zero());
    EXPECT_FALSE(v1.is_near_zero());
    
    Vector2d small(1e-12, 1e-12);
    EXPECT_TRUE(small.is_near_zero());
}

TEST_F(Vector2Test, NonMemberOperators) {
    Vector2d result = 2.0 * v1;
    EXPECT_DOUBLE_EQ(result.x, 6.0);
    EXPECT_DOUBLE_EQ(result.y, 8.0);
}