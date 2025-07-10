# Example applications
add_executable(simple_example simple_example.cpp)
target_link_libraries(simple_example orbital_mechanics)

add_executable(performance_comparison performance_comparison.cpp)
target_link_libraries(performance_comparison orbital_mechanics)

# Install examples
install(TARGETS simple_example performance_comparison
    RUNTIME DESTINATION bin/examples
)