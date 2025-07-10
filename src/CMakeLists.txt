# Main executables
add_executable(lagrange_simulation lagrange_simulation.cpp)
target_link_libraries(lagrange_simulation orbital_mechanics)

add_executable(cassini_simulation cassini_simulation.cpp)
target_link_libraries(cassini_simulation orbital_mechanics)

# Install executables
install(TARGETS lagrange_simulation cassini_simulation
    RUNTIME DESTINATION bin
)