# Orbital Mechanics Library

A modern, high-performance C++ library for orbital mechanics simulations, featuring the Circular Restricted Three-Body Problem (CR3BP) with applications to Lagrange point dynamics and ring system evolution.

## Features

- **Modern C++17** design with header-only architecture
- **Multiple numerical integrators** (Euler, RK4, Verlet) with accuracy and performance trade-offs
- **Circular Restricted Three-Body Problem** solver with built-in Sun-Earth and Saturn-Mimas systems
- **Comprehensive testing** with Google Test framework
- **Flexible I/O** supporting text, CSV, and HDF5 output formats
- **CMake build system** with cross-platform support
- **Extensive documentation** with Doxygen API docs and examples

## Quick Start

### Prerequisites

- C++17 compatible compiler (GCC 7+, Clang 6+, MSVC 2017+)
- CMake 3.16+
- Git

### Building

```bash
git clone <repository-url>
cd Orbital-Mechanics
mkdir build && cd build
cmake ..
make -j4
```

### Running Examples

```bash
# Lagrange point simulation
./src/lagrange_simulation

# Cassini division simulation
./src/cassini_simulation

# Simple example
./examples/simple_example

# Performance comparison
./examples/performance_comparison
```

### Running Tests

```bash
# Run all tests
make test

# Or run directly
./tests/orbital_mechanics_tests
```

## Library Usage

### Basic Example

```cpp
#include <orbital_mechanics/orbital_mechanics.hpp>
using namespace orbital_mechanics;

// Create simulation configuration
simulation::SimulationConfig config;
config.time_step = 3600.0;  // 1 hour
config.total_time = 365.25 * 24 * 3600;  // 1 year
config.integrator_type = "rk4";
config.output_filename = "output.txt";

// Create simulation
simulation::Simulation sim(config);

// Set up Sun-Earth system
auto solver = std::make_unique<physics::CR3BPSolver>(
    physics::systems::sun_earth());
sim.set_physics_solver(std::move(solver));

// Add bodies and run
sim.add_body(/* ... */);
sim.run();
```

### Advanced Usage

```cpp
// Custom integrator comparison
auto rk4 = integrators::create_integrator("rk4");
auto verlet = integrators::create_integrator("verlet");

// Custom derivative function
auto derivative_func = [](const integrators::State& state, 
                         double mass, double time, void* user_data) {
    return integrators::Derivative(state.velocity, calculate_acceleration(state));
};

// Multiple output formats
auto writer = io::create_data_writer("csv", true);  // Include velocities
```

## Applications

### 1. Lagrange Point Dynamics

Simulates test particle motion near the L4 Lagrange point of the Sun-Earth system, demonstrating:
- Quasi-stable orbital dynamics
- Tadpole and horseshoe orbits
- Effects of perturbations on equilibrium points

### 2. Cassini Division Formation

Models the clearing of Saturn's rings due to gravitational resonance with Mimas:
- 2:1 orbital resonance effects
- Particle ejection dynamics
- Ring gap formation over time

## Architecture

The library follows a modular design with clear separation of concerns:

```
orbital_mechanics/
├── core/           # Basic data structures (Vector2, Body, Constants)
├── integrators/    # Numerical integration schemes
├── physics/        # Physics calculations (CR3BP solver)
├── io/            # Data input/output handling
└── simulation/    # High-level simulation framework
```

## Performance

- **Optimized for speed**: Modern C++ with compiler optimizations
- **Memory efficient**: Header-only design with minimal allocations
- **Scalable**: Support for thousands of particles in ring simulations
- **Accurate**: 4th-order Runge-Kutta integration with configurable time steps

## Configuration Options

### CMake Options

```bash
cmake -DBUILD_TESTS=ON \
      -DBUILD_EXAMPLES=ON \
      -DUSE_OPENMP=ON \
      -DUSE_HDF5=ON \
      ..
```

### Simulation Parameters

- **Time step**: Balance between accuracy and performance
- **Integrator type**: euler, rk4, verlet
- **Output format**: text, csv, hdf5
- **Output frequency**: Control data volume

## Testing

The library includes comprehensive tests covering:

- **Unit tests** for all major components
- **Integration tests** for complete simulations
- **Performance benchmarks** comparing integrators
- **Physics validation** with known analytical solutions

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this library in academic work, please cite:

```bibtex
@software{orbital_mechanics_library,
  author = {Arnav Menon},
  title = {Orbital Mechanics Library: Modern C++ Framework for Three-Body Dynamics},
  year = {2024},
  version = {1.0.0},
  url = {https://github.com/username/orbital-mechanics}
}
```

## References

1. Szebehely, V. (1967). *Theory of Orbits: The Restricted Problem of Three Bodies*
2. Murray, C. D., & Dermott, S. F. (1999). *Solar System Dynamics*
3. Danby, J. M. A. (1992). *Fundamentals of Celestial Mechanics*

## Contact

- Author: Arnav Menon
- Email: [email@example.com]
- GitHub: [github.com/username]