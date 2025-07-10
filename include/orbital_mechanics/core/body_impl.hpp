#pragma once

#include "body.hpp"
#include "constants.hpp"
#include <sstream>
#include <iomanip>

namespace orbital_mechanics::core {

inline Body::Body(double mass, const Vector2d& position, const Vector2d& velocity, 
                  std::string name)
    : mass_(mass), position_(position), velocity_(velocity), name_(std::move(name)) {
    if (mass <= 0.0) {
        throw std::invalid_argument("Body mass must be positive");
    }
}

inline void Body::set_state(const Vector2d& position, const Vector2d& velocity) noexcept {
    position_ = position;
    velocity_ = velocity;
}

inline double Body::kinetic_energy() const noexcept {
    return 0.5 * mass_ * velocity_.magnitude_squared();
}

inline double Body::distance_to(const Body& other) const noexcept {
    return (position_ - other.position_).magnitude();
}

inline double Body::distance_squared_to(const Body& other) const noexcept {
    return (position_ - other.position_).magnitude_squared();
}

inline Vector2d Body::displacement_to(const Body& other) const noexcept {
    return other.position_ - position_;
}

inline void Body::update_position(double dt) noexcept {
    position_ += velocity_ * dt;
}

inline void Body::update_velocity(const Vector2d& acceleration, double dt) noexcept {
    velocity_ += acceleration * dt;
}

inline void Body::update_state(const Vector2d& acceleration, double dt) noexcept {
    velocity_ += acceleration * dt;
    position_ += velocity_ * dt;
}

inline bool Body::is_at_position(const Vector2d& pos, double tolerance) const noexcept {
    return (position_ - pos).magnitude_squared() < tolerance * tolerance;
}

inline std::string Body::to_string() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    oss << "Body";
    if (!name_.empty()) {
        oss << " '" << name_ << "'";
    }
    oss << " (mass=" << mass_ << ", pos=" << position_ << ", vel=" << velocity_ << ")";
    return oss.str();
}

inline Body make_circular_orbit_body(double mass, double radius, 
                                    double central_mass, double angle,
                                    std::string name) {
    if (mass <= 0.0 || radius <= 0.0 || central_mass <= 0.0) {
        throw std::invalid_argument("Mass and radius must be positive");
    }
    
    // Calculate orbital velocity for circular orbit
    const double orbital_speed = std::sqrt(Constants::G * central_mass / radius);
    
    // Position and velocity vectors
    const Vector2d position{radius * std::cos(angle), radius * std::sin(angle)};
    const Vector2d velocity{-orbital_speed * std::sin(angle), orbital_speed * std::cos(angle)};
    
    return Body(mass, position, velocity, std::move(name));
}

}  // namespace orbital_mechanics::core