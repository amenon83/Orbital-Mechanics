/**
 * @file test_main.cpp
 * @brief Main test file for orbital mechanics library
 * 
 * This file contains the main function for running all unit tests
 * for the orbital mechanics library.
 * 
 * @author Arnav Menon
 */

#include <gtest/gtest.h>
#include <orbital_mechanics/orbital_mechanics.hpp>

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}