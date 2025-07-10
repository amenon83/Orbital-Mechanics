# Find Google Test
find_package(GTest QUIET)

if(NOT GTest_FOUND)
    # If Google Test is not found, try to download it
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        v1.14.0
    )
    
    # For Windows: Prevent overriding the parent project's compiler/linker settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    
    FetchContent_MakeAvailable(googletest)
    
    # Aliases for consistency
    add_library(GTest::GTest ALIAS gtest)
    add_library(GTest::Main ALIAS gtest_main)
endif()

# Test executable
add_executable(orbital_mechanics_tests
    test_main.cpp
    test_vector.cpp
    test_body.cpp
    test_integrators.cpp
    test_three_body.cpp
    test_simulation.cpp
)

target_link_libraries(orbital_mechanics_tests
    orbital_mechanics
    GTest::GTest
    GTest::Main
)

# Add test to CTest
add_test(NAME OrbitalMechanicsTests COMMAND orbital_mechanics_tests)

# Set properties for test discovery
set_property(TARGET orbital_mechanics_tests PROPERTY FOLDER "Tests")

# Enable test discovery in IDEs
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.10")
    include(GoogleTest)
    gtest_discover_tests(orbital_mechanics_tests)
endif()