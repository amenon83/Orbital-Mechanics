#pragma once

#include "vector.hpp"
#include <memory>
#include <string>

namespace orbital_mechanics::core {

/**
 * @brief Represents a celestial body with mass, position, and velocity
 * 
 * This class encapsulates the properties of a celestial body in a simulation,
 * including its mass, position, and velocity vectors. It provides methods
 * for updating the body's state and calculating derived quantities.
 */
class Body {
public:
    /**
     * @brief Constructs a Body with specified properties
     * @param mass Mass of the body in kg
     * @param position Initial position vector in m
     * @param velocity Initial velocity vector in m/s
     * @param name Optional name for the body
     */
    Body(double mass, const Vector2d& position, const Vector2d& velocity, 
         std::string name = "");
    
    // Getters
    [[nodiscard]] double mass() const noexcept { return mass_; }
    [[nodiscard]] const Vector2d& position() const noexcept { return position_; }
    [[nodiscard]] const Vector2d& velocity() const noexcept { return velocity_; }
    [[nodiscard]] const std::string& name() const noexcept { return name_; }
    
    // Individual component getters for convenience
    [[nodiscard]] double x() const noexcept { return position_.x; }
    [[nodiscard]] double y() const noexcept { return position_.y; }
    [[nodiscard]] double vx() const noexcept { return velocity_.x; }
    [[nodiscard]] double vy() const noexcept { return velocity_.y; }
    
    // Setters
    void set_position(const Vector2d& position) noexcept { position_ = position; }
    void set_velocity(const Vector2d& velocity) noexcept { velocity_ = velocity; }
    void set_state(const Vector2d& position, const Vector2d& velocity) noexcept;
    
    // Physics calculations
    [[nodiscard]] double kinetic_energy() const noexcept;
    [[nodiscard]] double distance_to(const Body& other) const noexcept;
    [[nodiscard]] double distance_squared_to(const Body& other) const noexcept;
    [[nodiscard]] Vector2d displacement_to(const Body& other) const noexcept;
    
    // State updates
    void update_position(double dt) noexcept;
    void update_velocity(const Vector2d& acceleration, double dt) noexcept;
    void update_state(const Vector2d& acceleration, double dt) noexcept;
    
    // Utility functions
    [[nodiscard]] bool is_at_position(const Vector2d& pos, double tolerance = 1e-10) const noexcept;
    [[nodiscard]] std::string to_string() const;
    
private:
    double mass_;
    Vector2d position_;
    Vector2d velocity_;
    std::string name_;
};

/**
 * @brief Creates a body at rest at the origin
 * @param mass Mass of the body in kg
 * @param name Optional name for the body
 * @return A new Body instance
 */
[[nodiscard]] inline Body make_body_at_origin(double mass, std::string name = "") {
    return Body(mass, Vector2d{0.0, 0.0}, Vector2d{0.0, 0.0}, std::move(name));
}

/**
 * @brief Creates a body with circular orbital velocity
 * @param mass Mass of the body in kg
 * @param radius Orbital radius in m
 * @param central_mass Mass of the central body in kg
 * @param angle Initial angle in radians
 * @param name Optional name for the body
 * @return A new Body instance
 */
[[nodiscard]] Body make_circular_orbit_body(double mass, double radius, 
                                           double central_mass, double angle = 0.0,
                                           std::string name = "");

}  // namespace orbital_mechanics::core