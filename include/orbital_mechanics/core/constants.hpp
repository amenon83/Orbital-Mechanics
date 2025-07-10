#pragma once

#include <cmath>

namespace orbital_mechanics::core {

/**
 * @brief Physical constants used in orbital mechanics calculations
 */
struct Constants {
    static constexpr double PI = 3.141592653589793238462643383279502884197;
    static constexpr double TWO_PI = 2.0 * PI;
    static constexpr double SQRT_3 = 1.7320508075688772935274463415058723669428;
    
    // Gravitational constant (m³/kg/s²)
    static constexpr double G = 6.67430e-11;
    
    // Solar System masses (kg)
    static constexpr double SUN_MASS = 1.989e30;
    static constexpr double EARTH_MASS = 5.972e24;
    static constexpr double SATURN_MASS = 5.6834e26;
    static constexpr double MIMAS_MASS = 3.75e19;
    
    // Distances (m)
    static constexpr double AU = 1.496e11;  // Astronomical unit
    static constexpr double SUN_EARTH_DISTANCE = AU;
    static constexpr double MIMAS_SEMI_MAJOR_AXIS = 1.85539e8;
    
    // Time conversions (s)
    static constexpr double DAY_TO_SECONDS = 86400.0;
    static constexpr double YEAR_TO_SECONDS = 365.25 * DAY_TO_SECONDS;
    
    // Numerical tolerances
    static constexpr double EPSILON = 1e-15;
    static constexpr double SMALL_NUMBER = 1e-10;
};

}  // namespace orbital_mechanics::core