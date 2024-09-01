import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Yale Color Palette
yale_light_blue = '#286DC0'
yale_gray = '#666666'
yale_red = '#990000'

# Additional colors for combined plots
yale_green = '#0E7C7B'
yale_yellow = '#F4D35E'

# Define the system parameters
m = 1.0  # mass
t_max = 20
dt = 0.1
t = np.arange(0, t_max, dt)

# Function to compute the system dynamics
def system_dynamics(t, k, c, x0, x_dot0):
    omega_0 = np.sqrt(k/m)
    zeta = c / (2 * np.sqrt(m * k))
    omega_d = omega_0 * np.sqrt(1 - zeta**2)

    if zeta < 1:
        A = x0
        B = (x_dot0 + zeta * omega_0 * x0) / omega_d
        x_t = np.exp(-zeta * omega_0 * t) * (A * np.cos(omega_d * t) + B * np.sin(omega_d * t))
    elif zeta == 1:  # Critically damped
        A = x0
        B = x_dot0 + omega_0 * x0
        x_t = (A + B * t) * np.exp(-omega_0 * t)
    else:  # Overdamped
        r1 = -omega_0 * (zeta + np.sqrt(zeta**2 - 1))
        r2 = -omega_0 * (zeta - np.sqrt(zeta**2 - 1))
        A = (x_dot0 - r2 * x0) / (r1 - r2)
        B = x0 - A
        x_t = A * np.exp(r1 * t) + B * np.exp(r2 * t)

    x_dot_t = np.gradient(x_t, dt)
    return x_t, x_dot_t

# Function to add vector field to the phase diagram
def add_vector_field(ax, k, c):
    X, X_dot = np.meshgrid(np.linspace(-2, 2, 20), np.linspace(-2, 2, 20))
    U = X_dot
    V = -(c/m) * X_dot - (k/m) * X
    ax.quiver(X, X_dot, U, V, color=yale_gray, alpha=0.7, scale=50, width=0.003)

# Define distinct damping cases with more pronounced differences
damping_cases = [
    {"label": "underdamped", "k": 1.0, "c": 0.2, "color": yale_light_blue},
    {"label": "critically_damped", "k": 1.0, "c": 2.1 * np.sqrt(m), "color": yale_green},
    {"label": "overdamped", "k": 1.0, "c": 6.0 * np.sqrt(m), "color": yale_yellow},  # Significantly overdamped
]

# Initial condition sets
initial_conditions = [
    {"x0": 0.0, "x_dot0": 2.0},  # Zero displacement, non-zero velocity
    {"x0": 1.0, "x_dot0": 0.0},  # Non-zero displacement, zero velocity
]

# Animation function
def animate(i, x_t, x_dot_t):
    # Update time plot
    line_time.set_data(t[:i], x_t[:i])
    dot_time.set_data([t[i]], [x_t[i]])  # Current position as Yale Red dot
    
    # Update phase plot
    line_phase.set_data(x_t[:i], x_dot_t[:i])
    dot_phase.set_data([x_t[i]], [x_dot_t[i]])  # Current position as Yale Red dot
    
    # Update mass position (moving vertically now)
    mass.set_data([0], [x_t[i]])  # Move the mass vertically
    spring.set_data([0, 0], [0, x_t[i]])  # Spring is now vertical

    return line_time, dot_time, line_phase, dot_phase, mass, spring

# Generate and save the animation for each case and each initial condition
for case in damping_cases:
    for i, ic in enumerate(initial_conditions):
        x_t, x_dot_t = system_dynamics(t, case["k"], case["c"], ic["x0"], ic["x_dot0"])

        # Create figure and axes for each animation
        fig, axs = plt.subplots(1, 3, figsize=(18, 6), facecolor='black')
        ax_system, ax_time, ax_phase = axs
        for ax in axs:
            ax.set_facecolor('black')
            ax.grid(True, color=yale_gray, alpha=0.2)
            ax.tick_params(colors=yale_gray)
        
        # Initialize plots
        line_time, = ax_time.plot([], [], color=case["color"])
        dot_time, = ax_time.plot([], [], 'o', color=yale_red)
        line_phase, = ax_phase.plot([], [], color=case["color"])
        dot_phase, = ax_phase.plot([], [], 'o', color=yale_red)
        mass, = ax_system.plot([], [], 'o', color=yale_red, markersize=10)
        spring, = ax_system.plot([], [], color=case["color"], lw=2)

        # Set up plot limits and labels with larger text
        ax_time.set_xlim(0, t_max)
        ax_time.set_ylim(-2, 2)
        ax_time.set_xlabel('Time', color=case["color"], fontsize=14)
        ax_time.set_ylabel('x(t)', color=case["color"], fontsize=14)
        ax_time.set_title('Position over Time', color=case["color"], fontsize=16)

        ax_phase.set_xlim(-2, 2)
        ax_phase.set_ylim(-2, 2)
        ax_phase.set_xlabel('x', color=case["color"], fontsize=14)
        ax_phase.set_ylabel("x'", color=case["color"], fontsize=14)
        ax_phase.set_title('Phase Diagram', color=case["color"], fontsize=16)

        ax_system.set_xlim(-1, 1)
        ax_system.set_ylim(-2, 2)
        # Removed xlabel from the system plot
        ax_system.set_ylabel('Position (x)', color=case["color"], fontsize=14)
        ax_system.set_xticks([])
        ax_system.set_yticks(np.arange(-2, 3, 1))
        ax_system.set_title('System Visualization', color=case["color"], fontsize=16)

        # Add vector field to the phase diagram
        add_vector_field(ax_phase, case["k"], case["c"])
        
        # Create animation
        ani = FuncAnimation(fig, animate, frames=len(t), fargs=(x_t, x_dot_t), interval=100, blit=True)
        
        # Save the animation
        filename = f"{case['label']}_ic{i+1}_yale.mp4"
        ani.save(filename, writer='ffmpeg', dpi=150)
        print(f"Saved animation for {case['label']} with initial condition set {i+1} as {filename}")

        # Clear the figure for the next animation
        plt.clf()

# # Combined plots for both initial conditions
# for i, ic in enumerate(initial_conditions):
#     plt.figure(figsize=(10, 6), facecolor='black')
#     plt.grid(True, color=yale_gray, alpha=0.2)
#     plt.tick_params(colors=yale_gray)
#     plt.xlim(0, t_max)
#     plt.ylim(-2, 2)
#     plt.xlabel('Time', color='white', fontsize=14)
#     plt.ylabel('x(t)', color='white', fontsize=14)
#     plt.title(f'Combined Damping Cases - Initial Condition {i+1}', color='white', fontsize=16)
    
#     for case in damping_cases:
#         x_t, _ = system_dynamics(t, case["k"], case["c"], ic["x0"], ic["x_dot0"])
#         plt.plot(t, x_t, label=case['label'], color=case["color"])
    
#     plt.legend(loc='upper right', fontsize=12)
#     plt.savefig(f"combined_ic{i+1}_yale.png", dpi=150, facecolor='black')
#     plt.show()
