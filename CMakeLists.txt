cmake_minimum_required(VERSION 3.16)
project(OrbitalMechanics VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set build type if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler-specific options
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wpedantic")
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    set(CMAKE_CXX_FLAGS "/W4")
    set(CMAKE_CXX_FLAGS_DEBUG "/Od /Zi /DDEBUG")
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /DNDEBUG")
endif()

# Add include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Find packages
find_package(Threads REQUIRED)
find_package(OpenMP)

# Option to enable/disable components
option(BUILD_TESTS "Build unit tests" ON)
option(BUILD_EXAMPLES "Build example applications" ON)
option(BUILD_BENCHMARKS "Build performance benchmarks" ON)
option(USE_OPENMP "Enable OpenMP parallelization" ON)
option(USE_HDF5 "Enable HDF5 output support" ON)

# HDF5 support
if(USE_HDF5)
    find_package(HDF5 COMPONENTS C CXX)
    if(HDF5_FOUND)
        add_definitions(-DUSE_HDF5)
        include_directories(${HDF5_INCLUDE_DIRS})
    else()
        message(WARNING "HDF5 not found, disabling HDF5 support")
        set(USE_HDF5 OFF)
    endif()
endif()

# OpenMP support
if(USE_OPENMP AND OpenMP_CXX_FOUND)
    add_definitions(-DUSE_OPENMP)
endif()

# Create main library (header-only)
add_library(orbital_mechanics INTERFACE)
target_include_directories(orbital_mechanics INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(orbital_mechanics INTERFACE cxx_std_17)

# Link libraries
target_link_libraries(orbital_mechanics INTERFACE Threads::Threads)

if(USE_OPENMP AND OpenMP_CXX_FOUND)
    target_link_libraries(orbital_mechanics INTERFACE OpenMP::OpenMP_CXX)
endif()

if(USE_HDF5 AND HDF5_FOUND)
    target_link_libraries(orbital_mechanics INTERFACE ${HDF5_LIBRARIES})
endif()

# Add subdirectories
add_subdirectory(src)

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(BUILD_BENCHMARKS AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks)
    add_subdirectory(benchmarks)
endif()

# Installation
install(TARGETS orbital_mechanics
    EXPORT OrbitalMechanicsTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

install(DIRECTORY include/ DESTINATION include)

install(EXPORT OrbitalMechanicsTargets
    FILE OrbitalMechanicsTargets.cmake
    NAMESPACE OrbitalMechanics::
    DESTINATION lib/cmake/OrbitalMechanics
)

# Create config files
include(CMakePackageConfigHelpers)
configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/OrbitalMechanicsConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/OrbitalMechanicsConfig.cmake
    INSTALL_DESTINATION lib/cmake/OrbitalMechanics
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/OrbitalMechanicsConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/OrbitalMechanicsConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/OrbitalMechanicsConfigVersion.cmake
    DESTINATION lib/cmake/OrbitalMechanics
)