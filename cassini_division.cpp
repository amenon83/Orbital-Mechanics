/**
 * @file cassini_division.cpp
 * @brief Simulates the clearing of the Cassini Division in Saturn's rings.
 * @details This program simulates the motion of numerous test particles initially
 * orbiting Saturn in the region near Mimas. It uses the restricted three-body
 * problem model (Saturn, Mimas, test particle) in a rotating reference frame
 * co-rotating with Mimas. The simulation demonstrates how gravitational
 * resonance with Mimas ejects particles from the 2:1 resonance region,
 * forming the Cassini Division. It employs a 4th-order Runge-Kutta (RK4)
 * integration method for higher accuracy over long simulation times.
 * Outputs particle positions to a text file ("cassini_output.txt").
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
 #include <random>
 #include <iomanip>
 #include <chrono>
 
 // Here, we define physical constants and simulation parameters in a namespace
 namespace Constants {
     constexpr double PI = 3.141592653589793; // How many digits of pi can you recite?
     constexpr double G = 6.67430e-11; // Gravitational constant in SI units
     constexpr double SATURN_MASS = 5.6834e26; // Mass of Saturn in kg
     constexpr double MIMAS_MASS = 3.75e19; // Mass of Mimas in kg
     constexpr double TEST_PARTICLE_MASS = 1.0; // Mass of the test particle in kg
     constexpr double MIMAS_SEMI_MAJOR_AXIS = 1.85539e8; // Semi-major axis of Mimas in meters
     constexpr double TIME_STEP = 100.0; // Time step in seconds
     constexpr int NUM_STEPS = 10000000; // Total number of simulation steps
     constexpr int NUM_TEST_PARTICLES = 100; // Number of test particles
     constexpr int OUTPUT_INTERVAL = 50; // Write data to file every N steps
     constexpr int CONSOLE_UPDATE_INTERVAL = NUM_STEPS / 20; // Update console progress often
 
     // --- Particle Initialization Parameters ---
     // Minimum initial orbital radius for test particles (m) - Inside Cassini Division
     constexpr double MIN_PARTICLE_RADIUS = 1.10e8;
     // Maximum initial orbital radius for test particles (m) - Outside Cassini Division
     constexpr double MAX_PARTICLE_RADIUS = 1.30e8;

     // Output filename
     const std::string OUTPUT_FILENAME = "cassini_output.txt";
 }
 
 /**
  * @struct Vector2D
  * @brief Structure to represent 2D vectors (position, velocity, force, etc.).
  */
 struct Vector2D {
     double x = 0.0;
     double y = 0.0;
 };
 
 /**
  * @class Body
  * @brief Represents a celestial body in the simulation (Saturn, Mimas, or test particle).
  * @details Stores mass and state vectors (position, velocity) in the rotating frame.
  */
 class Body {
 private:
     double mass_kg; // Mass
     Vector2D position_m; // Position vector {x, y}
     Vector2D velocity_mps; // Velocity vector {vx, vy}
 
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
         position_m{xPos, yPos},
         velocity_mps{xVel, yVel} {}
 
     // --- Getters for body properties ---
     double getMass() const { return mass_kg; }
     Vector2D getPosition() const { return position_m; }
     Vector2D getVelocity() const { return velocity_mps; }
     double getX() const { return position_m.x; }
     double getY() const { return position_m.y; }
     double getVx() const { return velocity_mps.x; }
     double getVy() const { return velocity_mps.y; }
 
     /**
      * @brief Sets the position and velocity state of the body directly.
      * @param newPos The new position vector {x, y}.
      * @param newVel The new velocity vector {vx, vy}.
      */
     void setState(const Vector2D& newPos, const Vector2D& newVel) {
         position_m = newPos;
         velocity_mps = newVel;
     }
 };
 
 /**
  * @struct Derivatives
  * @brief Holds the time derivatives needed for RK4: dr/dt and dv/dt.
  */
 struct Derivatives {
     Vector2D dv; // Represents dr/dt = velocity
     Vector2D da; // Represents dv/dt = acceleration
 };
 
 /**
  * @brief Calculates the time derivatives (velocity and acceleration) for a body's state.
  * @details This function is the core of the physics calculation for the RK4 integrator.
  * It takes a state (position, velocity) and computes the acceleration acting on
  * the body at that state due to gravity (M1, M2), centrifugal, and Coriolis effects
  * in the rotating frame.
  * 
  * The equations of motion in the rotating frame are:
  * F_net = F_g1 + F_g2 + F_centrifugal + F_coriolis,
  * where F = m * a, and accelerations are:
  * a_g1 = -G*M1*(r - r1)/|r - r1|^3
  * a_g2 = -G*M2*(r - r2)/|r - r2|^3
  * a_centrifugal = omega^2 * r
  * a_coriolis = -2 * omega x v
  * 
  * 
  * @param pos Current position vector {x, y} (m).
  * @param vel Current velocity vector {vx, vy} (m/s).
  * @param mass Mass of the body being evaluated (kg).
  * @param M1 The first primary body (Saturn).
  * @param M2 The second primary body (Mimas).
  * @param omega The angular velocity of the rotating frame (rad/s).
  * @return Derivatives struct containing {velocity, acceleration}.
  */
 Derivatives calculateDerivatives(
     const Vector2D& pos,
     const Vector2D& vel,
     double mass,
     const Body& M1,
     const Body& M2,
     double omega)
 {
     Derivatives output;
     // We can directly put in a velocity
     output.dv = vel;
 
     // To test the code, sometime we want to set M1 or M2 to zero.
     // This error is for those who set the test mass to zero instead...
     if (mass <= 0.0) {
         std::cerr << "Warning: calculateDerivatives called with zero or negative mass." << std::endl;
         output.da = {0.0, 0.0}; // Return zero acceleration
         return output;
     }
 
     // Calculate vector displacements from M1 and M2 to the current position
     const double dx1 = pos.x - M1.getX();
     const double dy1 = pos.y - M1.getY();
     const double dx2 = pos.x - M2.getX();
     const double dy2 = pos.y - M2.getY();
 
     // Squared distances
     const double r1_sq = dx1 * dx1 + dy1 * dy1;
     const double r2_sq = dx2 * dx2 + dy2 * dy2;
 
     // Handle potential division by zero if particle coincides with a primary body
     if (r1_sq == 0.0 || r2_sq == 0.0) {
         std::cerr << "Warning: Particle coincided with a primary body during derivative calculation." << std::endl;
         output.da = {0.0, 0.0}; // Return zero acceleration
         return output;
     }
 
     // Calculate distances and inverse cube distances for gravitational force
     const double r1 = std::sqrt(r1_sq);
     const double r2 = std::sqrt(r2_sq);

     const double inv_r1_cubed = 1.0 / (r1_sq * r1);
     const double inv_r2_cubed = 1.0 / (r2_sq * r2);
 
     // --- Calculate Gravitational Forces ---
     // Acceleration:
     // a_g1 = -G*M1*(r - r1)/|r - r1|^3
     // a_g2 = -G*M2*(r - r2)/|r - r2|^3
     const double a_g1x = -Constants::G * M1.getMass() * dx1 * inv_r1_cubed;
     const double a_g1y = -Constants::G * M1.getMass() * dy1 * inv_r1_cubed;
     const double a_g2x = -Constants::G * M2.getMass() * dx2 * inv_r2_cubed;
     const double a_g2y = -Constants::G * M2.getMass() * dy2 * inv_r2_cubed;
 
     // --- Calculate Centrifugal Force ---
     // Acceleration:
     // a_centrifugal = omega^2 * r
     const double a_cen_x = omega * omega * pos.x;
     const double a_cen_y = omega * omega * pos.y;
 
     // --- Calculate Coriolis Force ---
     // Acceleration:
     // a_coriolis = -2 * omega x v
     const double a_cor_x = 2.0 * omega * vel.y;
     const double a_cor_y = -2.0 * omega * vel.x;
 
     // --- Sum accelerations ---
     output.da.x = a_g1x + a_g2x + a_cen_x + a_cor_x;
     output.da.y = a_g1y + a_g2y + a_cen_y + a_cor_y;
 
     return output;
 }
 
 /**
  * @brief Initializes the vector of Body objects for the simulation.
  * @details Sets up Saturn and Mimas at their fixed positions in the rotating frame
  * and distributes test particles randomly within a specified radial range.
  * Initial velocities for test particles are calculated assuming near-circular
  * orbits around Saturn (inertial frame) and then transformed to the rotating frame.
  * @param[out] bodies The vector of Body objects to be initialized.
  * @param omega The angular velocity of the rotating frame (rad/s).
  */
 void initializeSimulationBodies(std::vector<Body>& bodies, double omega) {
     // Setup random number generation
     std::random_device rd;
     std::mt19937 gen(rd());

     // Uniform distributions for radius and angle
     std::uniform_real_distribution<> radius_dist(Constants::MIN_PARTICLE_RADIUS, Constants::MAX_PARTICLE_RADIUS);
     std::uniform_real_distribution<> angle_dist(0.0, 2.0 * Constants::PI); // Full circle
     
     // Let's ensure the vector is empty before filling
     bodies.clear();
 
     // --- Setup Primary Bodies (Saturn and Mimas) ---
     const double m1 = Constants::SATURN_MASS;
     const double m2 = Constants::MIMAS_MASS;
     const double totalMass = m1 + m2;
     const double a = Constants::MIMAS_SEMI_MAJOR_AXIS;
 
     // Calculate positions relative to the barycenter (center of mass)
     // The barycenter is the origin (0,0) of the rotating frame.
     const double x_saturn = -a * m2 / totalMass;
     const double x_mimas = a * m1 / totalMass;
 
     bodies.emplace_back(m1, x_saturn, 0.0, 0.0, 0.0); // Add Saturn (M1) - stationary in the rotating frame
     bodies.emplace_back(m2, x_mimas, 0.0, 0.0, 0.0); // Add Mimas (M2) - stationary in the rotating frame
 
     // --- Initialize Test Particles ---
     std::cout << "Initializing " << Constants::NUM_TEST_PARTICLES << " test particles..." << std::endl;
     for (int i = 0; i < Constants::NUM_TEST_PARTICLES; ++i) {
         // Generate random initial polar coordinates (relative to barycenter)
         const double r = radius_dist(gen); // Initial radial distance
         const double theta = angle_dist(gen); // Initial angle

         // Convert polar to Cartesian coordinates
         const double x = r * std::cos(theta);
         const double y = r * std::sin(theta);

         // Inertial Frame:
         // Calculate initial velocity assuming a circular orbit around Saturn in the inertial frame
         //
         // v = sqrt(G * M1 / r)
         // Note: This is an approximation, neglecting Mimas' influence initially.
         const double v_inertial_mag = std::sqrt(Constants::G * m1 / r);

         // Velocity components for a counter-clockwise circular orbit in inertial frame
         const double vx_inertial = -v_inertial_mag * std::sin(theta);
         const double vy_inertial =  v_inertial_mag * std::cos(theta);
 
         // Rotating Frame:
         // Transform initial velocity from inertial frame to rotating frame
         //
         // v_rotating = v_inertial - (omega x r)
         // where omega = (0, 0, omega) and r = (x, y, 0)
         // omega x r = (-omega*y, omega*x, 0)
         const double vx_rotating = vx_inertial - (-omega * y);
         const double vy_rotating = vy_inertial - (omega * x);
 
         // Add the test particle to the vector
         bodies.emplace_back(Constants::TEST_PARTICLE_MASS, x, y, vx_rotating, vy_rotating);
     }
      std::cout << "Initialization complete." << std::endl;
 }
 
 /**
  * @brief Opens the output file for writing simulation data.
  * @param[out] outFile The output file stream object.
  * @param filename The name of the file to open.
  * @return true if the file was opened successfully, false otherwise.
  */
 bool openOutputFile(std::ofstream& outFile, const std::string& filename) {
     outFile.open(filename);

     // Check if the file opened successfully
     if (!outFile.is_open()) {
         std::cerr << "Error: Could not open output file '" << filename << "' for writing." << std::endl;
         return false;
     }
     
     // Set output format for floating point numbers
     outFile << std::fixed << std::setprecision(6); // Fixed point notation with 6 decimal places
     std::cout << "\nWriting simulation data to " << filename << std::endl;
     return true;
 }
 
 /**
  * @brief Writes the header line to the output data file.
  * @param outFile The output file stream object.
  * @param numParticles The number of test particles being tracked.
  */
 void writeOutputFileHeader(std::ofstream& outFile, int numParticles) {
     outFile << "# Simulation Data for Cassini Division Clearing\n";
     outFile << "# Rotating frame centered at Saturn-Mimas barycenter.\n";
     outFile << "# Format: Time(s)";

     // Add column headers for each test particle's position
     for (int i = 0; i < numParticles; ++i) {
         outFile << " P" << i << "_x(m) P" << i << "_y(m)";
     }

     outFile << "\n";
 }
 
 /**
  * @brief Main function to run the Cassini Division simulation.
  * @return EXIT_SUCCESS if successful, EXIT_FAILURE otherwise.
  */
 int main() {
     // Start timing the simulation
     auto start_time = std::chrono::high_resolution_clock::now();

     // Console output for simulation start
     std::cout << "Starting Cassini Division Clearing Simulation..." << std::endl;
     std::cout << "Bodies: Saturn, Mimas, " << Constants::NUM_TEST_PARTICLES << " test particles." << std::endl;

     if (Constants::NUM_TEST_PARTICLES > 10000) {
         std::cout << "WARNING: Simulating a large number of particles ("
                   << Constants::NUM_TEST_PARTICLES << "), this may take a while..." << std::endl;
     }

     if (Constants::NUM_STEPS > 1000000) {
         std::cout << "INFO: Simulating for a large number of steps ("
                   << Constants::NUM_STEPS << "), expecting long runtime." << std::endl;
     }
 
     // --- Setup System Parameters ---
     const double m1 = Constants::SATURN_MASS;
     const double m2 = Constants::MIMAS_MASS;
     const double totalMass = m1 + m2;
     const double a = Constants::MIMAS_SEMI_MAJOR_AXIS;
 
     // Calculations for the orbit information
     const double omega = std::sqrt(Constants::G * totalMass / std::pow(a, 3));
     const double mimas_period_s = 2.0 * (Constants::PI / omega);
     const double total_simulation_time_s = static_cast<double>(Constants::NUM_STEPS) * Constants::TIME_STEP;
     const double total_mimas_orbits = total_simulation_time_s / mimas_period_s;
 
     // Orbit information
     std::cout << "\n--- Orbit Information ---" << std::endl;
     std::cout << "Rotating frame omega: " << omega << " rad/s" << std::endl;
     std::cout << "Mimas orbital period: " << mimas_period_s / (24.0 * 3600.0) << " days (" << mimas_period_s << " s)" << std::endl;
     std::cout << "Total simulation time: " << total_mimas_orbits << " Mimas orbits ("
               << total_simulation_time_s / (24.0*3600.0) << " days)" << std::endl;
     std::cout << "Simulation steps: " << Constants::NUM_STEPS
               << ", Time step: " << Constants::TIME_STEP << " s" << std::endl;
     std::cout << "------------------------\n" << std::endl;
 
 
     // --- Initialize Bodies ---
     std::vector<Body> bodies;
     bodies.reserve(2 + Constants::NUM_TEST_PARTICLES); // Reserve space to avoid reallocations during initialization
     initializeSimulationBodies(bodies, omega);
 
     // Get references to Saturn and Mimas
     const Body& saturn = bodies[0];
     const Body& mimas = bodies[1];
 
     // The bad ending.
     std::ofstream outFile;
     if (!openOutputFile(outFile, Constants::OUTPUT_FILENAME)) {
         return EXIT_FAILURE; // Exit if file cannot be opened
     }

     // Write header to the output file
     writeOutputFileHeader(outFile, Constants::NUM_TEST_PARTICLES);
 
     // --- Simulation Loop ---
     std::cout << "Running simulation..." << std::endl;
     int last_percent_reported = -1; // For progress reporting
 
     for (int step = 0; step < Constants::NUM_STEPS; ++step) {
         // Integrate only the test particles (indices 2 onwards)
         for (size_t i = 2; i < bodies.size(); ++i) {
             // Particle info
             const Vector2D pos0 = bodies[i].getPosition();
             const Vector2D vel0 = bodies[i].getVelocity();
             const double mass = bodies[i].getMass();
             const double dt = Constants::TIME_STEP;
 
             // --- RK4 Steps ---
             // 1. Calculate derivatives at the beginning of the step (k1)
             const Derivatives d1 = calculateDerivatives(pos0, vel0, mass, saturn, mimas, omega);
             const Vector2D k1_dv = {d1.dv.x * dt, d1.dv.y * dt}; // dr = v*dt
             const Vector2D k1_da = {d1.da.x * dt, d1.da.y * dt}; // dv = a*dt
 
             // 2. Calculate derivatives at the midpoint using k1 (k2)
             // Position at midpoint: pos0 + k1_dv * 0.5
             // Velocity at midpoint: vel0 + k1_da * 0.5
             const Vector2D pos1 = {pos0.x + k1_dv.x * 0.5, pos0.y + k1_dv.y * 0.5};
             const Vector2D vel1 = {vel0.x + k1_da.x * 0.5, vel0.y + k1_da.y * 0.5};
             const Derivatives d2 = calculateDerivatives(pos1, vel1, mass, saturn, mimas, omega);
             const Vector2D k2_dv = {d2.dv.x * dt, d2.dv.y * dt};
             const Vector2D k2_da = {d2.da.x * dt, d2.da.y * dt};
 
             // 3. Calculate derivatives at the midpoint using k2 (k3)
             // Position at midpoint: pos0 + k2_dv * 0.5
             // Velocity at midpoint: vel0 + k2_da * 0.5
             const Vector2D pos2 = {pos0.x + k2_dv.x * 0.5, pos0.y + k2_dv.y * 0.5};
             const Vector2D vel2 = {vel0.x + k2_da.x * 0.5, vel0.y + k2_da.y * 0.5};
             const Derivatives d3 = calculateDerivatives(pos2, vel2, mass, saturn, mimas, omega);
             const Vector2D k3_dv = {d3.dv.x * dt, d3.dv.y * dt};
             const Vector2D k3_da = {d3.da.x * dt, d3.da.y * dt};
 
             // 4. Calculate derivatives at the end of the step using k3 (k4)
             // Position at end: pos0 + k3_dv
             // Velocity at end: vel0 + k3_da
             const Vector2D pos3 = {pos0.x + k3_dv.x, pos0.y + k3_dv.y};
             const Vector2D vel3 = {vel0.x + k3_da.x, vel0.y + k3_da.y};
             const Derivatives d4 = calculateDerivatives(pos3, vel3, mass, saturn, mimas, omega);
             const Vector2D k4_dv = {d4.dv.x * dt, d4.dv.y * dt};
             const Vector2D k4_da = {d4.da.x * dt, d4.da.y * dt};
 
             // --- Combine RK4 results to update state ---
             // Final Position: pos_new = pos0 + (k1_dv + 2*k2_dv + 2*k3_dv + k4_dv) / 6.0
             const Vector2D finalPos = {
                 pos0.x + (k1_dv.x + 2.0 * k2_dv.x + 2.0 * k3_dv.x + k4_dv.x) / 6.0,
                 pos0.y + (k1_dv.y + 2.0 * k2_dv.y + 2.0 * k3_dv.y + k4_dv.y) / 6.0
             };

             // Final Velocity: vel_new = vel0 + (k1_da + 2*k2_da + 2*k3_da + k4_da) / 6.0
              const Vector2D finalVel = {
                 vel0.x + (k1_da.x + 2.0 * k2_da.x + 2.0 * k3_da.x + k4_da.x) / 6.0,
                 vel0.y + (k1_da.y + 2.0 * k2_da.y + 2.0 * k3_da.y + k4_da.y) / 6.0
             };
 
             // Update the particle's state in the vector
             bodies[i].setState(finalPos, finalVel);
         }
 
         // --- Output Data periodically ---
         // Write state to file at specified intervals or on the last step
         if (step % Constants::OUTPUT_INTERVAL == 0 || step == Constants::NUM_STEPS - 1) {
             const double currentTime = static_cast<double>(step + 1) * Constants::TIME_STEP; // Time at end of step
             outFile << currentTime;

             // Write positions of all test particles (index 2 onwards)
             for (size_t i = 2; i < bodies.size(); ++i) {
                 outFile << " " << bodies[i].getX() << " " << bodies[i].getY();
             }

             outFile << "\n"; // Newline
         }
 
         // --- Update Console Progress ---
         // Calculate current progress percentage
         const int current_percent = static_cast<int>((100.0 * (step + 1)) / Constants::NUM_STEPS);

         // Report progress if percentage increased and it's time for a console update
         if (current_percent > last_percent_reported && ( (step + 1) % Constants::CONSOLE_UPDATE_INTERVAL == 0 || current_percent == 100) ) {
              // We can use carriage return '\r' to overwrite the previous progress line
              std::cout << "Progress: " << current_percent << "% \r" << std::flush;
              last_percent_reported = current_percent;
         }
 
     }
     
     // The good ending.
     outFile.close();
     std::cout << "\nSimulation finished successfully." << std::endl;
 
     // Stop timing and report duration
     auto end_time = std::chrono::high_resolution_clock::now();
     std::chrono::duration<double> elapsed_seconds = end_time - start_time;
     std::cout << "Total simulation time: " << std::fixed << std::setprecision(2)
               << elapsed_seconds.count() << " seconds." << std::endl;
 
     return EXIT_SUCCESS;
 }