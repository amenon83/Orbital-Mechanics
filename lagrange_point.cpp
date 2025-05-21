/**
 * @file lagrange_point.cpp
 * @brief Simulates the motion of a test particle near the L4 Lagrange point
 * of the Sun-Earth system using the restricted three-body problem model.
 * @details This program numerically integrates the equations of motion for a
 * massless test particle under the gravitational influence of the Sun
 * and Earth in a co-rotating reference frame centered at the
 * Sun-Earth barycenter. It employs a simple Euler integration method.
 * Outputs the positions of the Sun, Earth, and the test particle
 * to a text file ("L4_output.txt") over time.
 *
 * @author Arnav Menon
 */

 // Libraries to include
 #include <iostream>
 #include <vector>
 #include <cmath>
 #include <fstream>
 #include <string>
 #include <stdexcept>
 #include <limits>
 
 // Here, we define physical constants and simulation parameters in a namespace
 namespace Constants {
     constexpr double G = 6.67430e-11; // Gravitational constant in SI units
     constexpr double SUN_MASS = 1.989e30; // Mass of the Sun in kg
     constexpr double EARTH_MASS = 5.972e24; // Mass of the Earth in kg
     constexpr double TEST_PARTICLE_MASS = 1.0; // Mass of the test particle in kg
     constexpr double SUN_EARTH_DISTANCE = 1.496e11; // 1 AU
     constexpr double TIME_STEP = 3600.0; // Time step for the simulation integration in seconds (1 hour)
     constexpr int NUM_STEPS = 8760; // Total number of simulation steps (1 year)
     constexpr double L4_POSITION_PERTURBATION = 1.5e7; // Small displacement to the test particle
 }
 
 /**
  * @class Body
  * @brief Represents a celestial body in the simulation.
  * @details Stores mass, position (x, y), and velocity (vx, vy)
  * in the rotating reference frame. Includes methods to compute forces
  * and update state based on Euler integration.
  */
 class Body {
 private:
     // Mass, position, and velocity of the body in SI units
     double mass_kg; // mass
     double pos_x_m; // x-position
     double pos_y_m; // y-position
     double vel_x_mps; // x-velocity
     double vel_y_mps; // y-velocity
 
 public:
     /**
      * @brief Constructs a Body object.
      * @param m Mass in kg.
      * @param xPos Initial x-position in meters.
      * @param yPos Initial y-position in meters.
      * @param xVel Initial x-velocity in m/s.
      * @param yVel Initial y-velocity in m/s.
      */
     Body(double m, double xPos, double yPos, double xVel, double yVel) :
         mass_kg(m),
         pos_x_m(xPos),
         pos_y_m(yPos),
         vel_x_mps(xVel),
         vel_y_mps(yVel) {}
 
     // --- Getters for body properties ---
     double getMass() const { return mass_kg; }
     double getX() const { return pos_x_m; }
     double getY() const { return pos_y_m; }
     double getVx() const { return vel_x_mps; }
     double getVy() const { return vel_y_mps; }
 
     /**
      * @brief Computes the net force on this body in the rotating reference frame.
      * @details This function calculates the gravitational forces from M1 and M2,
      * the centrifugal force, and the Coriolis force acting on the test particle.
      * 
      * The equations of motion in the rotating frame are:
      * F_net = F_g1 + F_g2 + F_centrifugal + F_coriolis,
      * where F = m * a, and accelerations are:
      * a_g1 = -G*M1*(r - r1)/|r - r1|^3
      * a_g2 = -G*M2*(r - r2)/|r - r2|^3
      * a_centrifugal = omega^2 * r
      * a_coriolis = -2 * omega x v
      * 
      * @param M1 The first primary body (The Sun).
      * @param M2 The second primary body (Earth).
      * @param omega The angular velocity of the rotating frame in rad/s.
      * @param[out] fx The calculated net force component in the x-direction (Newtons).
      * @param[out] fy The calculated net force component in the y-direction (Newtons).
      */
     void computeForcesInRotatingFrame(const Body& M1, const Body& M2, double omega, double& fx, double& fy) const {
         // Calculate vector displacements from M1 and M2 to this body
         double dx1 = pos_x_m - M1.pos_x_m;
         double dy1 = pos_y_m - M1.pos_y_m;
         double dx2 = pos_x_m - M2.pos_x_m;
         double dy2 = pos_y_m - M2.pos_y_m;
 
         // Calculate squared distances to avoid immediate sqrt
         double r1_sq = dx1 * dx1 + dy1 * dy1;
         double r2_sq = dx2 * dx2 + dy2 * dy2;
 
         // Handle potential division by zero if particle coincides with a primary body
         if (r1_sq == 0.0 || r2_sq == 0.0) {
             std::cerr << "Warning: Test particle coincided with a primary body. Setting force to zero." << std::endl;
             fx = 0.0;
             fy = 0.0;
             return;
         }
 
         // Calculate distances and inverse cube distances for gravitational force
         double r1 = std::sqrt(r1_sq);
         double r2 = std::sqrt(r2_sq);

         double inv_r1_cubed = 1.0 / (r1_sq * r1);
         double inv_r2_cubed = 1.0 / (r2_sq * r2);
 
         // --- Calculate Gravitational Forces ---
         // Acceleration:
         // a_g1 = -G*M1*(r - r1)/|r - r1|^3
         // a_g2 = -G*M2*(r - r2)/|r - r2|^3
         double F_g1x = -Constants::G * M1.mass_kg * this->mass_kg * dx1 * inv_r1_cubed;
         double F_g1y = -Constants::G * M1.mass_kg * this->mass_kg * dy1 * inv_r1_cubed;
         double F_g2x = -Constants::G * M2.mass_kg * this->mass_kg * dx2 * inv_r2_cubed;
         double F_g2y = -Constants::G * M2.mass_kg * this->mass_kg * dy2 * inv_r2_cubed;
 
         // --- Calculate Centrifugal Force ---
         // Acceleration:
         // a_centrifugal = omega^2 * r
         double F_cen_x = this->mass_kg * omega * omega * pos_x_m;
         double F_cen_y = this->mass_kg * omega * omega * pos_y_m;
 
         // --- Calculate Coriolis Force ---
         // Acceleration:
         // a_coriolis = -2 * omega x v
         double F_cor_x = 2.0 * this->mass_kg * omega * vel_y_mps;
         double F_cor_y = -2.0 * this->mass_kg * omega * vel_x_mps;
 
         // --- Sum forces ---
         fx = F_g1x + F_g2x + F_cen_x + F_cor_x;
         fy = F_g1y + F_g2y + F_cen_y + F_cor_y;
     }
 
     /**
      * @brief Updates the body's velocity based on applied force with Euler integration.
      * @details v_new = v_old + (a * dt)
      * @param fx Force in x-direction (N).
      * @param fy Force in y-direction (N).
      * @param timeStep The simulation time step (s).
      */
     void updateVelocity(double fx, double fy, double timeStep) {
         // To test the code, sometime we want to set M1 or M2 to zero.
         // This error is for those who set the test mass to zero instead...
         if (mass_kg == 0.0) {
             std::cerr << "Warning: Attempted to update velocity for zero mass body." << std::endl;
             return;
         }

         // Calculate acceleration: a = F/m
         double ax = fx / mass_kg;
         double ay = fy / mass_kg;
         // Update velocity: v = v + (a * dt)
         vel_x_mps += ax * timeStep;
         vel_y_mps += ay * timeStep;
     }
 
     /**
      * @brief Updates the body's position based on its velocity using Euler method.
      * @details x_new = x_old + (v * dt)
      * @param timeStep The simulation time step (s).
      */
     void updatePosition(double timeStep) {
         pos_x_m += vel_x_mps * timeStep;
         pos_y_m += vel_y_mps * timeStep;
     }
 };
 
 /**
  * @brief Main function to run the L4 Lagrange point simulation.
  * @return EXIT_SUCCESS if successful, EXIT_FAILURE otherwise.
  */
 int main() {
     std::cout << "Starting Restricted Three-Body Simulation for L4..." << std::endl;
 
     // --- Setup Primary Bodies (Sun and Earth) ---
     const double m1 = Constants::SUN_MASS;
     const double m2 = Constants::EARTH_MASS;
     const double totalMass = m1 + m2;
     const double a = Constants::SUN_EARTH_DISTANCE;
 
     // Calculate positions relative to the barycenter (center of mass)
     // The barycenter is the origin (0,0) of the rotating frame.
     const double x_sun = -a * m2 / totalMass;
     const double x_earth = a * m1 / totalMass;
 
     // Create a vector to hold the bodies
     std::vector<Body> bodies;
     bodies.emplace_back(m1, x_sun, 0.0, 0.0, 0.0); // Add Sun (M1) - assumed stationary in the rotating frame
     bodies.emplace_back(m2, x_earth, 0.0, 0.0, 0.0); // Add Earth (M2) - assumed stationary in the rotating frame
 
     // Get references to Sun and Earth for easier access
     const Body& sun = bodies[0];
     const Body& earth = bodies[1];
 
     // Calculate the angular velocity of the rotating frame (Kepler's 3rd Law)
     // omega^2 = G * (M1 + M2) / a^3
     double omega_rad_per_s = std::sqrt(Constants::G * totalMass / std::pow(a, 3));
     std::cout << "Rotating frame omega: " << omega_rad_per_s << " rad/s" << std::endl;
 
     // --- Setup Test Particle ---
     // L4 forms an equilateral triangle with M1 and M2 such that, relative to the barycenter,
     // its coordinates are < (x1+x2)/2 , sqrt(3)/2 * a >.

     // Note that (x_sun + x_earth)/2 = a/2 * (m1-m2)/(m1+m2) and
     // that the height of an equilateral triangle of side length a is sqrt(3)/2 * a.
     double x_L4_bary = a * (m1 - m2) / (2.0 * totalMass);
     double y_L4_bary = a * std::sqrt(3) / 2.0;
 
     std::cout << "Approximate L4 relative to barycenter: (" << x_L4_bary << ", " << y_L4_bary << ")" << std::endl;
 
     // Set initial position of the test particle slightly perturbed from L4
     double initial_test_x = x_L4_bary + Constants::L4_POSITION_PERTURBATION;
     double initial_test_y = y_L4_bary;
     bodies.emplace_back(Constants::TEST_PARTICLE_MASS, initial_test_x, initial_test_y, 0.0, 0.0); // velocity can be zero
     Body& testParticle = bodies[2]; // Get a reference to the test particle
 
     // --- Setup Output File ---
     const std::string outputFilename = "L4_output.txt";
     std::ofstream outFile(outputFilename);
 
     // The bad ending.
     if (!outFile.is_open()) {
         std::cerr << "Error: Could not open output file '" << outputFilename << "' for writing." << std::endl;
         return EXIT_FAILURE;
     }
     
     // Customize output format
     std::cout << "Writing simulation data to " << outputFilename << std::endl;
     outFile << "# Time(days) SunX(m) SunY(m) EarthX(m) EarthY(m) TestX(m) TestY(m)\n";
     outFile.precision(15); // Use high precision for coordinates
 
     // --- Simulation Loop ---
     std::cout << "Running simulation for " << Constants::NUM_STEPS << " steps ("
               << Constants::NUM_STEPS * Constants::TIME_STEP / (24.0 * 3600.0) << " days)..." << std::endl;

     for (int step = 0; step < Constants::NUM_STEPS; ++step) {
         // Calculate net force on the test particle
         double force_x = 0.0, force_y = 0.0;
         testParticle.computeForcesInRotatingFrame(sun, earth, omega_rad_per_s, force_x, force_y);
 
         // Update test particle's velocity using the calculated force (Euler step)
         testParticle.updateVelocity(force_x, force_y, Constants::TIME_STEP);
 
         // Update test particle's position using the new velocity (Euler step)
         testParticle.updatePosition(Constants::TIME_STEP);
 
         // Write the current state of all bodies to the output file
         double currentTimeDays = (step + 1) * Constants::TIME_STEP / (24.0 * 3600.0); // Calculate current time in days for output
         outFile << currentTimeDays << " "
                 << sun.getX() << " " << sun.getY() << " "      // Sun position
                 << earth.getX() << " " << earth.getY() << " "  // Earth position
                 << testParticle.getX() << " " << testParticle.getY() << "\n"; // Test particle position
     }
 
     // The good ending.
     outFile.close();
     std::cout << "Simulation finished successfully." << std::endl;
     return EXIT_SUCCESS;
 }
