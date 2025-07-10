#pragma once

#include <cmath>
#include <ostream>
#include <type_traits>

namespace orbital_mechanics::core {

/**
 * @brief A 2D vector class for position, velocity, and force calculations
 * @tparam T The scalar type (typically double)
 */
template<typename T = double>
class Vector2 {
    static_assert(std::is_arithmetic_v<T>, "Vector2 requires arithmetic type");
    
public:
    T x{};
    T y{};
    
    // Constructors
    constexpr Vector2() = default;
    constexpr Vector2(T x_val, T y_val) noexcept : x(x_val), y(y_val) {}
    
    // Copy and move constructors
    constexpr Vector2(const Vector2&) = default;
    constexpr Vector2(Vector2&&) = default;
    
    // Assignment operators
    constexpr Vector2& operator=(const Vector2&) = default;
    constexpr Vector2& operator=(Vector2&&) = default;
    
    // Arithmetic operators
    constexpr Vector2 operator+(const Vector2& other) const noexcept {
        return {x + other.x, y + other.y};
    }
    
    constexpr Vector2 operator-(const Vector2& other) const noexcept {
        return {x - other.x, y - other.y};
    }
    
    constexpr Vector2 operator*(T scalar) const noexcept {
        return {x * scalar, y * scalar};
    }
    
    constexpr Vector2 operator/(T scalar) const noexcept {
        return {x / scalar, y / scalar};
    }
    
    // Compound assignment operators
    constexpr Vector2& operator+=(const Vector2& other) noexcept {
        x += other.x;
        y += other.y;
        return *this;
    }
    
    constexpr Vector2& operator-=(const Vector2& other) noexcept {
        x -= other.x;
        y -= other.y;
        return *this;
    }
    
    constexpr Vector2& operator*=(T scalar) noexcept {
        x *= scalar;
        y *= scalar;
        return *this;
    }
    
    constexpr Vector2& operator/=(T scalar) noexcept {
        x /= scalar;
        y /= scalar;
        return *this;
    }
    
    // Comparison operators
    constexpr bool operator==(const Vector2& other) const noexcept {
        return x == other.x && y == other.y;
    }
    
    constexpr bool operator!=(const Vector2& other) const noexcept {
        return !(*this == other);
    }
    
    // Vector operations
    constexpr T dot(const Vector2& other) const noexcept {
        return x * other.x + y * other.y;
    }
    
    constexpr T cross(const Vector2& other) const noexcept {
        return x * other.y - y * other.x;
    }
    
    constexpr T magnitude_squared() const noexcept {
        return x * x + y * y;
    }
    
    T magnitude() const noexcept {
        return std::sqrt(magnitude_squared());
    }
    
    Vector2 normalized() const noexcept {
        const T mag = magnitude();
        return (mag > T{0}) ? *this / mag : Vector2{};
    }
    
    constexpr Vector2 perpendicular() const noexcept {
        return {-y, x};
    }
    
    // Utility functions
    constexpr bool is_zero() const noexcept {
        return x == T{0} && y == T{0};
    }
    
    constexpr bool is_near_zero(T tolerance = T{1e-10}) const noexcept {
        return magnitude_squared() < tolerance * tolerance;
    }
};

// Non-member operators
template<typename T>
constexpr Vector2<T> operator*(T scalar, const Vector2<T>& vec) noexcept {
    return vec * scalar;
}

template<typename T>
std::ostream& operator<<(std::ostream& os, const Vector2<T>& vec) {
    return os << "(" << vec.x << ", " << vec.y << ")";
}

// Type aliases
using Vector2d = Vector2<double>;
using Vector2f = Vector2<float>;

}  // namespace orbital_mechanics::core