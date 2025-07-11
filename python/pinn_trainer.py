#!/usr/bin/env python3
"""
Physics-Informed Neural Network trainer for Cassini Division formation simulation.

This module implements the neural network training for the fluid dynamics
simulation of Saturn's rings using PyTorch.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Optional
import os
import json
from tqdm import tqdm
import argparse

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class FluidPINN(nn.Module):
    """
    Physics-Informed Neural Network for fluid dynamics in Saturn's rings.
    
    The network takes (r, theta, t) as input and outputs (density, v_r, v_theta, pressure).
    """
    
    def __init__(self, hidden_layers: List[int] = [64, 64, 64, 64], 
                 activation: str = 'tanh'):
        super(FluidPINN, self).__init__()
        
        self.hidden_layers = hidden_layers
        self.activation_name = activation
        
        # Define activation function
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'swish':
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # Build network layers
        layers = []
        input_dim = 3  # r, theta, t
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(self.activation)
            input_dim = hidden_dim
        
        # Output layer: density, v_r, v_theta, pressure
        layers.append(nn.Linear(input_dim, 4))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights using Xavier initialization."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 3) containing (r, theta, t)
        
        Returns:
            Output tensor of shape (batch_size, 4) containing (density, v_r, v_theta, pressure)
        """
        return self.network(x)
    
    def predict_fluid_state(self, r: torch.Tensor, theta: torch.Tensor, t: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Predict fluid state variables at given coordinates.
        
        Args:
            r: Radial coordinates
            theta: Angular coordinates  
            t: Time coordinates
        
        Returns:
            Dictionary containing fluid state variables
        """
        inputs = torch.stack([r, theta, t], dim=1)
        outputs = self.forward(inputs)
        
        return {
            'density': outputs[:, 0],
            'velocity_r': outputs[:, 1],
            'velocity_theta': outputs[:, 2],
            'pressure': outputs[:, 3]
        }

class RingFluidParameters:
    """Parameters for Saturn ring fluid dynamics simulation."""
    
    def __init__(self):
        # Physical constants
        self.G = 6.67430e-11  # Gravitational constant
        self.saturn_mass = 5.6834e26  # Mass of Saturn (kg)
        self.mimas_mass = 3.75e19     # Mass of Mimas (kg)
        self.mimas_orbital_radius = 1.85539e8  # Mimas orbital radius (m)
        
        # Derived quantities
        self.mimas_orbital_period = 2 * np.pi * np.sqrt(
            self.mimas_orbital_radius**3 / (self.G * self.saturn_mass)
        )
        
        # Ring parameters
        self.ring_inner_radius = 1.0e8   # 100,000 km
        self.ring_outer_radius = 1.4e8   # 140,000 km
        self.viscosity = 1e3             # Kinematic viscosity (m²/s)
        self.sound_speed = 1e2           # Sound speed (m/s)
        self.surface_density_scale = 1e3 # Characteristic surface density (kg/m²)
    
    def keplerian_angular_velocity(self, r: torch.Tensor) -> torch.Tensor:
        """Calculate Keplerian angular velocity at radius r."""
        return torch.sqrt(self.G * self.saturn_mass / (r**3))
    
    def gravitational_acceleration(self, r: torch.Tensor) -> torch.Tensor:
        """Calculate gravitational acceleration at radius r."""
        return self.G * self.saturn_mass / (r**2)
    
    def mimas_perturbation_potential(self, r: torch.Tensor, theta: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Calculate Mimas perturbation potential."""
        mimas_angle = 2 * np.pi * t / self.mimas_orbital_period
        mimas_x = self.mimas_orbital_radius * torch.cos(mimas_angle)
        mimas_y = self.mimas_orbital_radius * torch.sin(mimas_angle)
        
        particle_x = r * torch.cos(theta)
        particle_y = r * torch.sin(theta)
        
        distance = torch.sqrt((particle_x - mimas_x)**2 + (particle_y - mimas_y)**2)
        # Avoid singularity
        distance = torch.clamp(distance, min=1e6)
        
        return -self.G * self.mimas_mass / distance

class PINNTrainer:
    """Trainer for the Physics-Informed Neural Network."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize parameters
        self.params = RingFluidParameters()
        
        # Initialize network
        self.model = FluidPINN(
            hidden_layers=config.get('hidden_layers', [64, 64, 64, 64]),
            activation=config.get('activation', 'tanh')
        ).to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.get('learning_rate', 1e-3),
            weight_decay=config.get('weight_decay', 1e-5)
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=config.get('lr_step_size', 1000),
            gamma=config.get('lr_gamma', 0.9)
        )
        
        # Loss weights
        self.physics_weight = config.get('physics_weight', 1.0)
        self.boundary_weight = config.get('boundary_weight', 100.0)
        self.initial_weight = config.get('initial_weight', 100.0)
        
        # Training data
        self.n_collocation = config.get('n_collocation_points', 10000)
        self.n_boundary = config.get('n_boundary_points', 1000)
        self.n_initial = config.get('n_initial_points', 1000)
        
        # Training history
        self.loss_history = []
        self.physics_loss_history = []
        self.boundary_loss_history = []
        self.initial_loss_history = []
    
    def normalize_inputs(self, r: torch.Tensor, theta: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Normalize inputs to [-1, 1] range."""
        r_norm = 2 * (r - self.params.ring_inner_radius) / (self.params.ring_outer_radius - self.params.ring_inner_radius) - 1
        theta_norm = 2 * theta / (2 * np.pi) - 1
        t_norm = 2 * t / (10 * self.params.mimas_orbital_period) - 1
        
        return torch.stack([r_norm, theta_norm, t_norm], dim=1)
    
    def generate_collocation_points(self, n_points: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate random collocation points in the domain."""
        r = torch.rand(n_points, device=self.device) * (self.params.ring_outer_radius - self.params.ring_inner_radius) + self.params.ring_inner_radius
        theta = torch.rand(n_points, device=self.device) * 2 * np.pi
        t = torch.rand(n_points, device=self.device) * 10 * self.params.mimas_orbital_period
        
        return r, theta, t
    
    def generate_boundary_points(self, n_points: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate points on the boundary."""
        n_per_boundary = n_points // 2
        
        # Inner boundary
        r_inner = torch.full((n_per_boundary,), self.params.ring_inner_radius, device=self.device)
        theta_inner = torch.rand(n_per_boundary, device=self.device) * 2 * np.pi
        t_inner = torch.rand(n_per_boundary, device=self.device) * 10 * self.params.mimas_orbital_period
        
        # Outer boundary
        r_outer = torch.full((n_per_boundary,), self.params.ring_outer_radius, device=self.device)
        theta_outer = torch.rand(n_per_boundary, device=self.device) * 2 * np.pi
        t_outer = torch.rand(n_per_boundary, device=self.device) * 10 * self.params.mimas_orbital_period
        
        r = torch.cat([r_inner, r_outer])
        theta = torch.cat([theta_inner, theta_outer])
        t = torch.cat([t_inner, t_outer])
        
        return r, theta, t
    
    def generate_initial_points(self, n_points: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate points at initial time."""
        r = torch.rand(n_points, device=self.device) * (self.params.ring_outer_radius - self.params.ring_inner_radius) + self.params.ring_inner_radius
        theta = torch.rand(n_points, device=self.device) * 2 * np.pi
        t = torch.zeros(n_points, device=self.device)
        
        return r, theta, t
    
    def compute_derivatives(self, r: torch.Tensor, theta: torch.Tensor, t: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute derivatives using automatic differentiation."""
        r.requires_grad_(True)
        theta.requires_grad_(True)
        t.requires_grad_(True)
        
        inputs = self.normalize_inputs(r, theta, t)
        outputs = self.model(inputs)
        
        density = outputs[:, 0]
        velocity_r = outputs[:, 1]
        velocity_theta = outputs[:, 2]
        pressure = outputs[:, 3]
        
        # Compute first derivatives
        drho_dr = torch.autograd.grad(density, r, torch.ones_like(density), create_graph=True)[0]
        drho_dtheta = torch.autograd.grad(density, theta, torch.ones_like(density), create_graph=True)[0]
        drho_dt = torch.autograd.grad(density, t, torch.ones_like(density), create_graph=True)[0]
        
        dvr_dr = torch.autograd.grad(velocity_r, r, torch.ones_like(velocity_r), create_graph=True)[0]
        dvr_dtheta = torch.autograd.grad(velocity_r, theta, torch.ones_like(velocity_r), create_graph=True)[0]
        dvr_dt = torch.autograd.grad(velocity_r, t, torch.ones_like(velocity_r), create_graph=True)[0]
        
        dvtheta_dr = torch.autograd.grad(velocity_theta, r, torch.ones_like(velocity_theta), create_graph=True)[0]
        dvtheta_dtheta = torch.autograd.grad(velocity_theta, theta, torch.ones_like(velocity_theta), create_graph=True)[0]
        dvtheta_dt = torch.autograd.grad(velocity_theta, t, torch.ones_like(velocity_theta), create_graph=True)[0]
        
        dp_dr = torch.autograd.grad(pressure, r, torch.ones_like(pressure), create_graph=True)[0]
        dp_dtheta = torch.autograd.grad(pressure, theta, torch.ones_like(pressure), create_graph=True)[0]
        
        return {
            'density': density,
            'velocity_r': velocity_r,
            'velocity_theta': velocity_theta,
            'pressure': pressure,
            'drho_dr': drho_dr,
            'drho_dtheta': drho_dtheta,
            'drho_dt': drho_dt,
            'dvr_dr': dvr_dr,
            'dvr_dtheta': dvr_dtheta,
            'dvr_dt': dvr_dt,
            'dvtheta_dr': dvtheta_dr,
            'dvtheta_dtheta': dvtheta_dtheta,
            'dvtheta_dt': dvtheta_dt,
            'dp_dr': dp_dr,
            'dp_dtheta': dp_dtheta
        }
    
    def physics_loss(self, r: torch.Tensor, theta: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute physics-informed loss from governing equations."""
        derivs = self.compute_derivatives(r, theta, t)
        
        # Continuity equation: ∂ρ/∂t + (1/r)∂(rρv_r)/∂r + (1/r)∂(ρv_θ)/∂θ = 0
        continuity = (derivs['drho_dt'] + 
                     (1/r) * (derivs['density'] * derivs['velocity_r'] + 
                             r * derivs['drho_dr'] * derivs['velocity_r'] + 
                             r * derivs['density'] * derivs['dvr_dr']) +
                     (1/r) * (derivs['drho_dtheta'] * derivs['velocity_theta'] + 
                             derivs['density'] * derivs['dvtheta_dtheta']))
        
        # Radial momentum equation (simplified)
        grav_accel = self.params.gravitational_acceleration(r)
        radial_momentum = (derivs['dvr_dt'] + 
                          derivs['velocity_r'] * derivs['dvr_dr'] + 
                          (derivs['velocity_theta'] / r) * derivs['dvr_dtheta'] - 
                          derivs['velocity_theta']**2 / r +
                          derivs['dp_dr'] / derivs['density'] + 
                          grav_accel)
        
        # Azimuthal momentum equation (simplified)
        azimuthal_momentum = (derivs['dvtheta_dt'] + 
                             derivs['velocity_r'] * derivs['dvtheta_dr'] + 
                             (derivs['velocity_theta'] / r) * derivs['dvtheta_dtheta'] + 
                             derivs['velocity_r'] * derivs['velocity_theta'] / r +
                             derivs['dp_dtheta'] / (r * derivs['density']))
        
        # Equation of state: p = cs²ρ
        eos = derivs['pressure'] - self.params.sound_speed**2 * derivs['density']
        
        return torch.mean(continuity**2 + radial_momentum**2 + azimuthal_momentum**2 + eos**2)
    
    def boundary_loss(self, r: torch.Tensor, theta: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute boundary condition loss."""
        state = self.model.predict_fluid_state(r, theta, t)
        
        # Boundary conditions: reflecting walls, reduced density
        boundary_density = 0.1 * self.params.surface_density_scale
        boundary_vr = torch.zeros_like(state['velocity_r'])
        
        density_loss = torch.mean((state['density'] - boundary_density)**2)
        velocity_loss = torch.mean(state['velocity_r']**2)
        
        return density_loss + velocity_loss
    
    def initial_loss(self, r: torch.Tensor, theta: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute initial condition loss."""
        state = self.model.predict_fluid_state(r, theta, t)
        
        # Initial conditions: Gaussian density profile, Keplerian rotation
        keplerian_omega = self.params.keplerian_angular_velocity(r)
        initial_density = self.params.surface_density_scale * torch.exp(
            -0.5 * ((r - 1.2e8) / 1e7)**2
        )
        initial_vr = torch.zeros_like(state['velocity_r'])
        initial_vtheta = r * keplerian_omega
        
        density_loss = torch.mean((state['density'] - initial_density)**2)
        vr_loss = torch.mean((state['velocity_r'] - initial_vr)**2)
        vtheta_loss = torch.mean((state['velocity_theta'] - initial_vtheta)**2)
        
        return density_loss + vr_loss + vtheta_loss
    
    def train_step(self) -> Dict[str, float]:
        """Perform one training step."""
        self.optimizer.zero_grad()
        
        # Generate training points
        r_col, theta_col, t_col = self.generate_collocation_points(self.n_collocation)
        r_bound, theta_bound, t_bound = self.generate_boundary_points(self.n_boundary)
        r_init, theta_init, t_init = self.generate_initial_points(self.n_initial)
        
        # Compute losses
        physics_loss = self.physics_loss(r_col, theta_col, t_col)
        boundary_loss = self.boundary_loss(r_bound, theta_bound, t_bound)
        initial_loss = self.initial_loss(r_init, theta_init, t_init)
        
        # Total loss
        total_loss = (self.physics_weight * physics_loss + 
                     self.boundary_weight * boundary_loss + 
                     self.initial_weight * initial_loss)
        
        # Backward pass
        total_loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        
        return {
            'total_loss': total_loss.item(),
            'physics_loss': physics_loss.item(),
            'boundary_loss': boundary_loss.item(),
            'initial_loss': initial_loss.item()
        }
    
    def train(self, n_epochs: int) -> None:
        """Train the PINN model."""
        print(f"Starting training for {n_epochs} epochs...")
        
        with tqdm(range(n_epochs), desc="Training") as pbar:
            for epoch in pbar:
                losses = self.train_step()
                
                # Record losses
                self.loss_history.append(losses['total_loss'])
                self.physics_loss_history.append(losses['physics_loss'])
                self.boundary_loss_history.append(losses['boundary_loss'])
                self.initial_loss_history.append(losses['initial_loss'])
                
                # Update progress bar
                pbar.set_postfix({
                    'Total': f"{losses['total_loss']:.2e}",
                    'Physics': f"{losses['physics_loss']:.2e}",
                    'Boundary': f"{losses['boundary_loss']:.2e}",
                    'Initial': f"{losses['initial_loss']:.2e}"
                })
                
                # Save checkpoint
                if epoch % 1000 == 0:
                    self.save_checkpoint(epoch)
    
    def save_checkpoint(self, epoch: int) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss_history': self.loss_history,
            'config': self.config
        }
        
        os.makedirs('checkpoints', exist_ok=True)
        torch.save(checkpoint, f'checkpoints/pinn_checkpoint_{epoch}.pt')
    
    def plot_loss_history(self) -> None:
        """Plot training loss history."""
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.semilogy(self.loss_history)
        plt.title('Total Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        plt.subplot(2, 2, 2)
        plt.semilogy(self.physics_loss_history)
        plt.title('Physics Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        plt.subplot(2, 2, 3)
        plt.semilogy(self.boundary_loss_history)
        plt.title('Boundary Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        plt.subplot(2, 2, 4)
        plt.semilogy(self.initial_loss_history)
        plt.title('Initial Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        plt.tight_layout()
        plt.savefig('loss_history.png', dpi=300)
        plt.show()
    
    def generate_density_map(self, t: float, n_r: int = 100, n_theta: int = 100) -> np.ndarray:
        """Generate density map at given time."""
        r_grid = torch.linspace(self.params.ring_inner_radius, self.params.ring_outer_radius, n_r)
        theta_grid = torch.linspace(0, 2*np.pi, n_theta)
        
        R, Theta = torch.meshgrid(r_grid, theta_grid, indexing='ij')
        R_flat = R.flatten()
        Theta_flat = Theta.flatten()
        T_flat = torch.full_like(R_flat, t)
        
        with torch.no_grad():
            state = self.model.predict_fluid_state(R_flat, Theta_flat, T_flat)
            density = state['density'].cpu().numpy().reshape(n_r, n_theta)
        
        return density, R.cpu().numpy(), Theta.cpu().numpy()

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train PINN for Cassini Division simulation')
    parser.add_argument('--config', type=str, default='config.json', help='Configuration file')
    parser.add_argument('--epochs', type=int, default=10000, help='Number of training epochs')
    args = parser.parse_args()
    
    # Load configuration
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = {
            'hidden_layers': [64, 64, 64, 64],
            'activation': 'tanh',
            'learning_rate': 1e-3,
            'weight_decay': 1e-5,
            'lr_step_size': 1000,
            'lr_gamma': 0.9,
            'physics_weight': 1.0,
            'boundary_weight': 100.0,
            'initial_weight': 100.0,
            'n_collocation_points': 10000,
            'n_boundary_points': 1000,
            'n_initial_points': 1000
        }
    
    # Initialize trainer
    trainer = PINNTrainer(config)
    
    # Train model
    trainer.train(args.epochs)
    
    # Plot results
    trainer.plot_loss_history()
    
    # Generate density maps
    time_points = np.linspace(0, 10 * trainer.params.mimas_orbital_period, 11)
    
    os.makedirs('results', exist_ok=True)
    
    for i, t in enumerate(time_points):
        density, R, Theta = trainer.generate_density_map(t)
        
        plt.figure(figsize=(10, 8))
        plt.contourf(R * np.cos(Theta), R * np.sin(Theta), density, levels=50, cmap='viridis')
        plt.colorbar(label='Density (kg/m²)')
        plt.title(f'Ring Density at t = {t/trainer.params.mimas_orbital_period:.1f} Mimas orbits')
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.axis('equal')
        plt.savefig(f'results/density_map_{i:02d}.png', dpi=300)
        plt.close()
    
    print("Training completed! Check 'results' directory for density maps.")

if __name__ == '__main__':
    main()