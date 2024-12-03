import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define the damped pendulum ODE system
def damped_pendulum(t, state, omega, gamma):
    x, y = state
    dxdt = y
    dydt = -omega**2 * np.sin(x) - gamma * y
    return [dxdt, dydt]

# Parameters
omega = 1.0  # Natural frequency

# Damping coefficients for different cases
gamma_over = 3.0     # Overdamped (gamma > 2*omega)
gamma_critical = 2.0 # Critically damped (gamma = 2*omega)
gamma_under = 1.0    # Underdamped (gamma < 2*omega)

# Time span for the simulation
t_span = (0, 20)
t_eval = np.linspace(t_span[0], t_span[1], 1000)

# Generate a set of initial conditions spread evenly throughout the domain
x0_values = np.pi * np.array([-2.5, -1.3, 0.7, 1.3, 2])
y0_values = np.array([2, 1, -1.5])
initial_conditions = [[x0, y0] for x0 in x0_values for y0 in y0_values]

# Grid for vector field covering the entire domain with increased density
grid_points = 30
X, Y = np.meshgrid(
    np.linspace(-3 * np.pi, 3 * np.pi, grid_points),
    np.linspace(-2, 2, grid_points)  # Y-axis remains as -2 to 2
)

# Compute vector field without normalization (for direction only)
def vector_field(x, y, omega, gamma):
    dx = y
    dy = -omega**2 * np.sin(x) - gamma * y
    magnitude = np.hypot(dx, dy)
    # Avoid division by zero
    magnitude[magnitude == 0] = 1
    # return dx / magnitude, dy / magnitude
    return dx, dy


# List of cases with their corresponding damping coefficients and titles
cases = [
    (gamma_over, 'Overdamped', f'{gamma_over}ω'),
    (gamma_critical, 'Critically Damped', f'{gamma_critical}ω'),
    (gamma_under, 'Underdamped', f'{gamma_under}ω')
]

# Create a single figure with stacked subplots
fig, axes = plt.subplots(3, 1, figsize=(8, 12), constrained_layout=True)

# Solve and plot for each case
for ax, (gamma, case_title, gamma_label) in zip(axes, cases):
    # Compute vector field
    DX, DY = vector_field(X, Y, omega, gamma)
    
    # Plot vector field with equal-length black arrows
    ax.quiver(X, Y, DX, DY, color='black', angles='xy', scale_units='xy', scale=20, width=0.002)
    
    # Plot trajectories for all initial conditions
    for x0, y0 in initial_conditions:
        # Solve the ODE
        sol = solve_ivp(
            damped_pendulum,
            t_span,
            [x0, y0],
            args=(omega, gamma),
            t_eval=t_eval,
            rtol=1e-8,
            atol=1e-8
        )
        x, y = sol.y
        # Plot the trajectory in phase space
        line, = ax.plot(x, y, linewidth=2)  # Save the line object
        ax.plot(x0, y0, 'o', color=line.get_color())  # Use the same color as the trajectory
    
    # Axis settings
    ax.set_title(f'{case_title} (γ={gamma_label})', fontsize=14)
    ax.set_xlabel('x (radians)', fontsize=12)
    ax.set_ylabel('y (rad/s)', fontsize=12)
    ax.set_xlim(-3 * np.pi, 3 * np.pi)
    ax.set_ylim(-2, 2)
    
    # Set x-axis ticks in multiples of π
    ax.set_xticks(np.arange(-3 * np.pi, 3.5 * np.pi, np.pi))
    ax.set_xticklabels([f'{int(i)}π' if i != 0 else '0' for i in np.arange(-3, 4)])
    
    # Set y-axis ticks as linear values from -2 to 2
    ax.set_yticks(np.linspace(-2, 2, 5))
    ax.set_yticklabels([f'{y:.1f}' for y in np.linspace(-2, 2, 5)])
    
    ax.grid(True)

# Save the figure as a PDF
fig.suptitle('Phase Diagrams with Vector Fields', fontsize=16)
output_file = 'phase_diagrams.pdf'
fig.savefig(output_file)
plt.close(fig)

print(f"Figure saved as {output_file}")
