import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from matplotlib.ticker import MultipleLocator

# Define examples as a Python dictionary
examples = {
    'dirichlet_boundaries_single_sine': {
        'initial_condition': lambda x, L, n: np.sin(n * np.pi * x / L),
        'boundary_conditions': {
            'type': 'Dirichlet',
            'left': lambda u, t: 0.0,   # u(0, t) = 0
            'right': lambda u, t: 0.0,  # u(L, t) = 0
        },
        'parameters': {
            'L': 1.0,
            'alpha': 0.5,  # Different conductivity
            'n': 1,        # Number of sine waves
            'Nx': 100,
            'T': 0.1,
        },
    },
    'dirichlet_boundaries_double_sine': {
        'initial_condition': lambda x, L, n: 0.5 + 0.5 * np.sin(n * np.pi * x / L),
        'boundary_conditions': {
            'type': 'Dirichlet',
            'left': lambda u, t: 0.5,  # Adjusted to match initial condition
            'right': lambda u, t: 0.5, # Adjusted to match initial condition
        },
        'parameters': {
            'L': 1.0,
            'alpha': 1.0,
            'n': 2,
            'Nx': 100,
            'T': 0.1,
        },
    },
    'dirichlet_boundaries_triple_sine': {
        'initial_condition': lambda x, L, n: 0.5 + 0.5 * np.sin(n * np.pi * x / L),
        'boundary_conditions': {
            'type': 'Dirichlet',
            'left': lambda u, t: 0.5,
            'right': lambda u, t: 0.5,
        },
        'parameters': {
            'L': 1.0,
            'alpha': 1.0,
            'n': 3,
            'Nx': 100,
            'T': 0.1,
        },
    },
    'neumann_boundaries_single_sine': {
        'initial_condition': lambda x, L, n: np.sin(n * np.pi * x / L),
        'boundary_conditions': {
            'type': 'Neumann',
            'left': lambda u, t: 0.0,   # ∂u/∂x(0, t) = 0
            'right': lambda u, t: 0.0,  # ∂u/∂x(L, t) = 0
        },
        'parameters': {
            'L': 1.0,
            'alpha': 0.8,
            'n': 1,
            'Nx': 100,
            'T': 0.1,
        },
    },
    'neumann_boundaries_double_sine': {
        'initial_condition': lambda x, L, n: 0.5 + 0.5 * np.sin(n * np.pi * x / L),
        'boundary_conditions': {
            'type': 'Neumann',
            'left': lambda u, t: 0.0,
            'right': lambda u, t: 0.0,
        },
        'parameters': {
            'L': 1.0,
            'alpha': 0.8,
            'n': 2,
            'Nx': 100,
            'T': 0.1,
        },
    },
    'neumann_boundaries_triple_sine': {
        'initial_condition': lambda x, L, n: 0.5 + 0.5 * np.sin(n * np.pi * x / L),
        'boundary_conditions': {
            'type': 'Neumann',
            'left': lambda u, t: 0.0,
            'right': lambda u, t: 0.0,
        },
        'parameters': {
            'L': 1.0,
            'alpha': 0.8,
            'n': 3,
            'Nx': 100,
            'T': 0.1,
        },
    },
    'mixed_neumann_dirichlet_single_sine': {
        'initial_condition': lambda x, L, n: np.sin(n * np.pi * x / L),
        'boundary_conditions': {
            'type': 'Mixed',
            'left': 'Neumann',
            'right': 'Dirichlet',
            'left_value': lambda u, t: 0.0,   # ∂u/∂x(0, t) = 0
            'right_value': lambda u, t: 0.0,  # u(L, t) = 0
        },
        'parameters': {
            'L': 1.0,
            'alpha': 1.2,
            'n': 1,
            'Nx': 100,
            'T': 0.1,
        },
    },
    'mixed_neumann_dirichlet_double_sine': {
        'initial_condition': lambda x, L, n: 0.5 + 0.5 * np.sin(n * np.pi * x / L),
        'boundary_conditions': {
            'type': 'Mixed',
            'left': 'Neumann',
            'right': 'Dirichlet',
            'left_value': lambda u, t: 0.0,
            'right_value': lambda u, t: 0.5,  # Adjusted to match initial condition
        },
        'parameters': {
            'L': 1.0,
            'alpha': 1.2,
            'n': 2,
            'Nx': 100,
            'T': 0.1,
        },
    },
    'mixed_neumann_dirichlet_triple_sine': {
        'initial_condition': lambda x, L, n: 0.5 + 0.5 * np.sin(n * np.pi * x / L),
        'boundary_conditions': {
            'type': 'Mixed',
            'left': 'Neumann',
            'right': 'Dirichlet',
            'left_value': lambda u, t: 0.0,
            'right_value': lambda u, t: 0.5,  # Adjusted to match initial condition
        },
        'parameters': {
            'L': 1.0,
            'alpha': 1.2,
            'n': 3,
            'Nx': 100,
            'T': 0.1,
        },
    },
}

# Loop through all examples
for example_name, example in examples.items():
    print(f"Running example: {example_name}")
    # Extract parameters from the example
    L = example['parameters']['L']
    alpha = example['parameters']['alpha']
    n_waves = example['parameters']['n']  # Number of sine waves
    Nx = example['parameters']['Nx']
    T = example['parameters']['T']
    
    dx = L / (Nx - 1)
    x = np.linspace(0, L, Nx)
    
    # Adjust dt to satisfy the stability condition
    dt_stable = dx**2 / (2 * alpha)
    dt = dt_stable * 0.5  # Use half the maximum stable time step for safety
    Nt = int(T / dt)
    
    print(f"Using dt = {dt:.2e}, Nt = {Nt}")
    
    # Initialize the solution array
    u = example['initial_condition'](x, L, n_waves)
    u_new = np.zeros_like(u)
    
    # Store the solution at each time step for animation
    u_history = []
    frame_interval = max(1, Nt // 300)  # Store approximately 300 frames
    
    # Define boundary condition functions
    boundary_type = example['boundary_conditions']['type']
    
    def apply_boundary_conditions(u, t):
        if boundary_type == 'Dirichlet':
            u[0] = example['boundary_conditions']['left'](u, t)
            u[-1] = example['boundary_conditions']['right'](u, t)
        elif boundary_type == 'Neumann':
            u[0] = u[1] - example['boundary_conditions']['left'](u, t) * dx
            u[-1] = u[-2] + example['boundary_conditions']['right'](u, t) * dx
        elif boundary_type == 'Mixed':
            # Left boundary
            if example['boundary_conditions']['left'] == 'Dirichlet':
                u[0] = example['boundary_conditions']['left_value'](u, t)
            elif example['boundary_conditions']['left'] == 'Neumann':
                u[0] = u[1] - example['boundary_conditions']['left_value'](u, t) * dx
            # Right boundary
            if example['boundary_conditions']['right'] == 'Dirichlet':
                u[-1] = example['boundary_conditions']['right_value'](u, t)
            elif example['boundary_conditions']['right'] == 'Neumann':
                u[-1] = u[-2] + example['boundary_conditions']['right_value'](u, t) * dx
        else:
            raise ValueError("Invalid boundary condition type.")
        return u
    
    # Time-stepping loop
    for n_step in range(Nt):
        t = n_step * dt
        # Compute the interior points
        u_new[1:-1] = u[1:-1] + alpha * dt / dx**2 * (u[2:] - 2 * u[1:-1] + u[:-2])
        # Apply boundary conditions
        u_new = apply_boundary_conditions(u_new, t)
        # Update the solution
        u = u_new.copy()
        # Store frames for animation at specified intervals
        if n_step % frame_interval == 0:
            u_history.append(u.copy())
    
    # Set up the plot with the desired styles
    fig, ax = plt.subplots()
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    ax.tick_params(axis='x', colors='white', which='both')
    ax.tick_params(axis='y', colors='white', which='both')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    ax.set_xlim(0, L)   # X-axis from 0 to L
    ax.set_ylim(0, 1)   # Y-axis from 0 to 1
    ax.set_xlabel('x')
    ax.set_ylabel('u(x, t)')
    ax.set_title(f'1D Heat Equation - {example_name}')
    
    # Set major ticks
    ax.set_xticks(np.linspace(0, L, 5))  # Major ticks at 0, L/4, L/2, 3L/4, L
    ax.set_yticks(np.linspace(0, 1, 5))  # Major ticks at 0, 0.25, 0.5, 0.75, 1
    # Set minor ticks
    ax.xaxis.set_minor_locator(MultipleLocator(L / 20))  # Minor ticks every L/20
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))    # Minor ticks every 0.05
    
    # Set x-axis labels only at 0 and L
    ax.set_xticklabels(['0', '', '', '', 'L'])
    # Set y-axis labels only at 0 and 1
    ax.set_yticklabels(['0', '', '', '', '1'])
    
    # Add gridlines
    ax.grid(which='major', color='white', linestyle='-', linewidth=1.0)
    # ax.grid(which='minor', color='white', linestyle='-', linewidth=1.0)
    
    line, = ax.plot([], [], color='red', lw=2)
    
    # Initialization function for the animation
    def init():
        line.set_data([], [])
        return line,
    
    # Animation function
    def animate(i):
        line.set_data(x, u_history[i])
        return line,
    
    # Create the animation
    ani = animation.FuncAnimation(fig, animate, frames=len(u_history),
                                  init_func=init, blit=True, interval=33)
    
    # Save the animation as an MP4 video
    output_filename = f"heat_{example_name.lower()}.mp4"
    print(f"Saving animation to {output_filename}")
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=30, metadata=dict(artist='User'), bitrate=1800)
    ani.save(output_filename, writer=writer)
    plt.close(fig)  # Close the figure to free memory

print("All examples have been processed and videos saved.")
