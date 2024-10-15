import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from mpl_toolkits.mplot3d import Axes3D  # Import for 3D plotting

# Define the boundary conditions and initial conditions
boundary_conditions_list = [
    {
        'name': 'dirichlet',
        'type': 'Dirichlet',
        'left': lambda u, t: 0.0,
        'right': lambda u, t: 0.0,
    },
    {
        'name': 'neumann',
        'type': 'Neumann',
        'left': lambda u, t: 0.0,
        'right': lambda u, t: 0.0,
    },
    {
        'name': 'mixed',
        'type': 'Mixed',
        'left': 'Neumann',
        'right': 'Dirichlet',
        'left_value': lambda u, t: 0.0,
        'right_value': lambda u, t: 0.0,
    },
]

# Sine wave numbers to loop over
n_values = [1, 2, 3, 4]

# Parameters
L = 1.0       # Length of the domain
Nx = 100      # Number of spatial grid points

# Equation selection: 'heat' or 'wave'
equation_type = 'wave'  # Change to 'wave' to solve the wave equation

# Adjust total simulation time based on the equation
if equation_type == 'heat':
    T = 0.1    # Total simulation time for heat equation
elif equation_type == 'wave':
    T = 1.0    # Longer total simulation time for wave equation

# Loop over boundary conditions
for bc in boundary_conditions_list:
    bc_name = bc['name']
    boundary_type = bc['type']

    # Loop over sine wave numbers
    for n in n_values:
        example_name = f"{bc_name}_boundaries_{n}_sine_{equation_type}"

        print(f"Running example: {example_name}")

        dx = L / (Nx - 1)
        x = np.linspace(0, L, Nx)

        # Adjust dt to satisfy the stability condition
        if equation_type == 'heat':
            # Thermal diffusivity for each boundary condition type
            alpha_values = {
                'dirichlet': 0.5,
                'neumann': 0.8,
                'mixed': 1.0,
            }
            alpha = alpha_values[bc_name]
            dt_stable = dx**2 / (2 * alpha)
            dt = dt_stable * 0.5  # Use half the maximum stable time step for safety
        elif equation_type == 'wave':
            # Wave speed
            c = 1.0
            dt_stable = dx / c
            dt = dt_stable * 0.9  # Use 90% of the maximum stable time step for safety

        Nt = int(T / dt)
        t_array = np.linspace(0, T, Nt)

        print(f"Using dt = {dt:.2e}, Nt = {Nt}")

        # Determine initial condition based on n
        if n == 1:
            if bc_name == 'neumann':
                initial_condition = lambda x: np.cos(n * np.pi * x / L)
            elif bc_name == 'mixed':
                initial_condition = lambda x: np.sin(n * np.pi * x / (2 * L))
            else:
                initial_condition = lambda x: np.sin(n * np.pi * x / L)
        else:
            # For n > 1, adjust mean and amplitude
            if bc_name == 'neumann':
                initial_condition = lambda x: 0.5 + 0.5 * np.cos(n * np.pi * x / L)
            elif bc_name == 'mixed':
                initial_condition = lambda x: 0.5 + 0.5 * np.sin(n * np.pi * x / (2 * L))
            else:
                initial_condition = lambda x: 0.5 + 0.5 * np.sin(n * np.pi * x / L)

        # Initialize the solution array
        u = initial_condition(x)
        u_new = np.zeros_like(u)
        U = np.zeros((Nt, Nx))  # Array to store u at all time steps
        U[0, :] = u.copy()

        # Initialize variables for wave equation
        if equation_type == 'wave':
            # Initial velocity (assuming zero initial velocity)
            initial_velocity = lambda x: np.zeros_like(x)
            v = initial_velocity(x)
            u_old = u.copy()

            # Compute u at the first time step using initial conditions
            u[1:-1] = u_old[1:-1] + dt * v[1:-1] + 0.5 * (c * dt / dx) ** 2 * (
                u_old[2:] - 2 * u_old[1:-1] + u_old[:-2]
            )
            # Apply boundary conditions to u
            def apply_boundary_conditions(u_array, t):
                if boundary_type == 'Dirichlet':
                    u_array[0] = bc['left'](u_array, t)
                    u_array[-1] = bc['right'](u_array, t)
                elif boundary_type == 'Neumann':
                    u_array[0] = u_array[1] - bc['left'](u_array, t) * dx
                    u_array[-1] = u_array[-2] + bc['right'](u_array, t) * dx
                elif boundary_type == 'Mixed':
                    # Left boundary
                    if bc['left'] == 'Dirichlet':
                        u_array[0] = bc['left_value'](u_array, t)
                    elif bc['left'] == 'Neumann':
                        u_array[0] = u_array[1] - bc['left_value'](u_array, t) * dx
                    # Right boundary
                    if bc['right'] == 'Dirichlet':
                        u_array[-1] = bc['right_value'](u_array, t)
                    elif bc['right'] == 'Neumann':
                        u_array[-1] = u_array[-2] + bc['right_value'](u_array, t) * dx
                else:
                    raise ValueError("Invalid boundary condition type.")
                return u_array

            # Adjust Dirichlet boundary values if needed
            if boundary_type == 'Dirichlet' and n > 1:
                bc['left'] = lambda u, t: 0.5
                bc['right'] = lambda u, t: 0.5
            if boundary_type == 'Mixed' and n > 1:
                bc['right_value'] = lambda u, t: 0.5

            # Apply boundary conditions
            u = apply_boundary_conditions(u, dt)
            U[1, :] = u.copy()
            start_step = 2  # Start time-stepping from step 2

        else:
            # For heat equation, only need apply boundary conditions function
            def apply_boundary_conditions(u_array, t):
                if boundary_type == 'Dirichlet':
                    u_array[0] = bc['left'](u_array, t)
                    u_array[-1] = bc['right'](u_array, t)
                elif boundary_type == 'Neumann':
                    u_array[0] = u_array[1] - bc['left'](u_array, t) * dx
                    u_array[-1] = u_array[-2] + bc['right'](u_array, t) * dx
                elif boundary_type == 'Mixed':
                    # Left boundary
                    if bc['left'] == 'Dirichlet':
                        u_array[0] = bc['left_value'](u_array, t)
                    elif bc['left'] == 'Neumann':
                        u_array[0] = u_array[1] - bc['left_value'](u_array, t) * dx
                    # Right boundary
                    if bc['right'] == 'Dirichlet':
                        u_array[-1] = bc['right_value'](u_array, t)
                    elif bc['right'] == 'Neumann':
                        u_array[-1] = u_array[-2] + bc['right_value'](u_array, t) * dx
                else:
                    raise ValueError("Invalid boundary condition type.")
                return u_array

            # Adjust Dirichlet boundary values if needed
            if boundary_type == 'Dirichlet' and n > 1:
                bc['left'] = lambda u, t: 0.5
                bc['right'] = lambda u, t: 0.5
            if boundary_type == 'Mixed' and n > 1:
                bc['right_value'] = lambda u, t: 0.5

            # Apply boundary conditions
            u = apply_boundary_conditions(u, 0)
            start_step = 1  # Start time-stepping from step 1

        # Store the solution at each time step for animation
        u_history = []
        frame_interval = max(1, Nt // 300)  # Store approximately 300 frames

        # Time-stepping loop
        for n_step in range(start_step, Nt):
            t = n_step * dt
            if equation_type == 'heat':
                # Compute the interior points for heat equation
                u_new[1:-1] = u[1:-1] + alpha * dt / dx**2 * (
                    u[2:] - 2 * u[1:-1] + u[:-2]
                )
                # Apply boundary conditions
                u_new = apply_boundary_conditions(u_new, t)
                # Update the solution
                u = u_new.copy()
            elif equation_type == 'wave':
                # Compute the interior points for wave equation
                u_new[1:-1] = (
                    2 * u[1:-1]
                    - u_old[1:-1]
                    + (c * dt / dx) ** 2 * (u[2:] - 2 * u[1:-1] + u[:-2])
                )
                # Apply boundary conditions
                u_new = apply_boundary_conditions(u_new, t)
                # Update the solution arrays
                u_old = u.copy()
                u = u_new.copy()

            # Store u in U for the 3D plot
            U[n_step, :] = u.copy()
            # Store frames for animation at specified intervals
            if n_step % frame_interval == 0:
                u_history.append((u.copy(), t))

        # Set up the figure with two subplots
        fig = plt.figure(figsize=(14, 6))  # Increased figure width for a larger 3D plot
        # Left subplot for animation
        ax1 = fig.add_subplot(1, 2, 1)
        # Right subplot for 3D plot
        ax2 = fig.add_subplot(1, 2, 2, projection='3d', proj_type='persp')

        # Configure left subplot (animation)
        fig.patch.set_facecolor('black')
        ax1.set_facecolor('black')
        ax1.tick_params(axis='x', colors='white', which='both')
        ax1.tick_params(axis='y', colors='white', which='both')
        ax1.xaxis.label.set_color('white')
        ax1.yaxis.label.set_color('white')
        ax1.set_xlim(0, L)   # X-axis from 0 to L (space)
        ax1.set_ylim(np.min(U), np.max(U))   # Y-axis adjusted based on solution
        ax1.set_xlabel('x')
        ax1.set_ylabel('u(x, t)')

        # Set major ticks
        ax1.set_xticks([0, L])
        ax1.set_xticklabels(['0', 'L'])
        ax1.set_yticks([np.min(U), np.max(U)])
        ax1.set_yticklabels([f'{np.min(U):.1f}', f'{np.max(U):.1f}'])

        # Remove minor ticks
        ax1.xaxis.set_minor_locator(plt.NullLocator())
        ax1.yaxis.set_minor_locator(plt.NullLocator())

        # Add gridlines
        ax1.grid(which='major', color='white', linestyle='-', linewidth=2.0)

        line1, = ax1.plot([], [], color='red', lw=2)

        # Configure right subplot (3D plot)
        ax2.set_facecolor('black')
        ax2.xaxis.label.set_color('white')
        ax2.yaxis.label.set_color('white')
        ax2.zaxis.label.set_color('white')
        ax2.tick_params(axis='x', colors='white')
        ax2.tick_params(axis='y', colors='white')
        ax2.tick_params(axis='z', colors='white')
        ax2.set_xlabel('x')
        ax2.set_ylabel('t')
        ax2.set_zlabel('u(x, t)')
        ax2.set_xlim(0, L)  # x-axis from 0 to L
        ax2.set_ylim(0, T)
        ax2.set_zlim(np.min(U), np.max(U))

        # Set x-axis labels only at 0 and L
        ax2.set_xticks([0, L])
        ax2.set_xticklabels(['0', 'L'])

        # Set z-axis (u-axis) labels
        ax2.set_zticks([np.min(U), np.max(U)])
        ax2.set_zticklabels([f'{np.min(U):.1f}', f'{np.max(U):.1f}'])

        # Prepare data for 3D plot
        X_mesh, T_mesh = np.meshgrid(x, t_array)
        # Plot the surface
        surf = ax2.plot_surface(X_mesh, T_mesh, U, cmap='viridis', alpha=0.7, zorder=1)

        # Adjust viewing angle to have x-axis left, t-axis right, u-axis up
        ax2.view_init(elev=30, azim=-60)

        # Move the red line slightly up on the u-axis
        epsilon = 0.0  # Offset to lift the line
        # Add a red line to highlight the current solution in the 3D plot
        current_line, = ax2.plot([], [], [], color='red', linewidth=3, zorder=2)

        # Initialization function for the animation
        def init():
            line1.set_data([], [])
            current_line.set_data([], [])
            current_line.set_3d_properties([])
            return line1, current_line

        # Animation function
        def animate(i):
            u_anim, current_t = u_history[i]
            # Update the 2D plot
            line1.set_data(x, u_anim)
            # Update the red line in the 3D plot
            current_line.set_data(x, [current_t]*len(x))
            current_line.set_3d_properties(u_anim + epsilon)
            return line1, current_line

        # Create the animation
        ani = animation.FuncAnimation(fig, animate, frames=len(u_history),
                                      init_func=init, blit=True, interval=33)

        # Save the animation as an MP4 video
        output_filename = f"{equation_type}_{example_name}.mp4"
        print(f"Saving animation to {output_filename}")
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=30, metadata=dict(artist='User'), bitrate=1800)
        ani.save(output_filename, writer=writer)
        plt.close(fig)  # Close the figure to free memory

print("All examples have been processed and videos saved.")
