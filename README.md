# Orbital Mechanics Simulation Project

This project simulates two scenarios based on the restricted three-body problem: the motion of a test particle near the L4 Lagrange point of the Sun-Earth system and the formation of the Cassini Division in Saturn's rings due to resonance with the moon Mimas. Both simulations utilize C++ for the core physics calculations and Python for visualization.

## Overview

The project consists of two main parts:

1.  **L4 Lagrange Point Simulation:**
    * Simulates the motion of a massless test particle in the vicinity of the L4 Lagrange point within the rotating reference frame defined by the Sun ($M_1$) and the Earth ($M_2$).
    * Calculates the particle's trajectory by considering the gravitational forces from the Sun and Earth, as well as the centrifugal and Coriolis forces arising from the rotating frame.
    * The simulation outputs the particle's position over time to `L4_output.txt`.
    * A Python script visualizes the results, showing the overall system, a zoomed-in view of the particle's path, and a static plot of the complete trajectory.

2.  **Cassini Division Simulation:**
    * Simulates the dynamics of numerous test particles initially placed in circular orbits between Saturn ($M_1$) and its moon Mimas ($M_2$).
    * Demonstrates how the 2:1 orbital resonance with Mimas clears a gap in the particle distribution over time, forming the Cassini Division.
    * Similar to the L4 simulation, this part calculates the gravitational, centrifugal, and Coriolis forces for each particle in the rotating reference frame defined by Saturn and Mimas.
    * The simulation outputs particle positions to `cassini_output.txt`.
    * A Python script animates the particle distribution, showing the gradual formation of the gap over many Mimas orbits.

## Theoretical Background: The Restricted Three-Body Problem

Both simulations are based on the **Restricted Three-Body Problem**. This problem considers the motion of a body with negligible mass ($m$) under the gravitational influence of two massive bodies ($M_1$ and $M_2$) that revolve in circles around their common center of mass. We analyze the motion in a reference frame that rotates with the same angular velocity ($\omega$) as the two massive bodies, keeping them stationary in this frame.

### Key Equations

1.  **Angular Velocity ($\omega$):** The angular velocity of the rotating frame (and the $M_1$-$M_2$ system) is given by Kepler's Third Law:
    $$ \omega^2 = \frac{G(M_1 + M_2)}{a^3} $$
    where $G$ is the gravitational constant and $a$ is the constant separation distance between $M_1$ and $M_2$.

2.  **Coordinates in Rotating Frame:** We place the origin at the center of mass and the x-axis along the line connecting $M_1$ and $M_2$. Their coordinates are:
    * $M_1$: $x_1 = -a \frac{M_2}{M_1 + M_2}$
    * $M_2$: $x_2 = a \frac{M_1}{M_1 + M_2}$

3.  **Equation of Motion:** The acceleration of the test particle $m$ at position $\vec{r} = (x, y)$ in the rotating frame is given by the vector equation:
    $$ \frac{d^2\vec{r}}{dt^2} = \frac{\vec{F}_1}{m} + \frac{\vec{F}_2}{m} - \vec{\omega} \times (\vec{\omega} \times \vec{r}) - 2\vec{\omega} \times \frac{d\vec{r}}{dt} $$
    where:
    * $\vec{F}_1$, $\vec{F}_2$ are the gravitational forces from $M_1$ and $M_2$, respectively.
    * $- \vec{\omega} \times (\vec{\omega} \times \vec{r})$ is the centrifugal acceleration.
    * $- 2\vec{\omega} \times \frac{d\vec{r}}{dt}$ is the Coriolis acceleration.

4.  **Component Form:** Writing the equation of motion in terms of $x$ and $y$ components yields:
    $$ \ddot{x} = -\frac{G M_1 (x - x_1)}{r_1^3} - \frac{G M_2 (x - x_2)}{r_2^3} + \omega^2 x + 2\omega \dot{y} $$
    $$ \ddot{y} = -\frac{G M_1 y}{r_1^3} - \frac{G M_2 y}{r_2^3} + \omega^2 y - 2\omega \dot{x} $$
    where:
    * $r_1 = \sqrt{(x - x_1)^2 + y^2}$ is the distance from $m$ to $M_1$.
    * $r_2 = \sqrt{(x - x_2)^2 + y^2}$ is the distance from $m$ to $M_2$.
    * $(\dot{x}, \dot{y})$ is the velocity and $(\ddot{x}, \ddot{y})$ is the acceleration in the rotating frame.
    * The $\omega^2 x$ and $\omega^2 y$ terms represent the centrifugal acceleration components.
    * The $2\omega \dot{y}$ and $-2\omega \dot{x}$ terms represent the Coriolis acceleration components.

These equations are numerically integrated in the C++ scripts to simulate the particle trajectories.

## Project Files

* `lagrange_point.cpp`: C++ source code for the L4 simulation. Outputs `L4_output.txt`.
* `l4_visualization.py`: Python script to visualize the L4 simulation results from `L4_output.txt`.
* `cassini_division.cpp`: C++ source code for the Cassini Division simulation. Outputs `cassini_output.txt`.
* `cassini_visualization.py`: Python script to visualize the Cassini Division simulation results from `cassini_output.txt`.

## Usage Instructions

These instructions assume you have a C++ compiler (like g++) and Python (with necessary libraries like Matplotlib and NumPy) installed.

1.  **Compile C++ Code:**
    Open a terminal or command prompt in the project directory.
    * For the L4 simulation:
        ```bash
        g++ lagrange_point.cpp -o lagrange_point
        ```
    * For the Cassini simulation:
        ```bash
        g++ cassini_division.cpp -o cassini_division
        ```

2.  **Run C++ Simulations:**
    * Execute the compiled L4 simulation:
        ```bash
        ./lagrange_point
        ```
        This will generate the `L4_output.txt` file.
    * Execute the compiled Cassini simulation:
        ```bash
        ./cassini_division
        ```
        This will generate the `cassini_output.txt` file.

3.  **Run Python Visualizations:**
    * Visualize the L4 results:
        ```bash
        python l4_visualization.py
        ```
    * Visualize the Cassini Division formation:
        ```bash
        python cassini_visualization.py
        ```

This should open animation windows displaying the results of the simulations. Ensure the Python scripts have read permissions for the respective `.txt` output files.