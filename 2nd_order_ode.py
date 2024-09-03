import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Color Palette
light_blue = "#286DC0"
white = "#FFFFFF"
red = "#990000"
green = "#0E7C7B"
yellow = "#F4D35E"

# Define the system parameters
t_max = 20

# Define distinct damping cases with different colors
damping_cases = [
    {"omega_0": 1.0, "zeta": 2.0, "label": "overdamped", "color": yellow},
    {"omega_0": 1.0, "zeta": 1.0, "label": "critically_damped", "color": green},
    {"omega_0": 1.0, "zeta": 0.2, "label": "underdamped", "color": red},
]

# Initial condition sets
initial_conditions = [
    {"x0": 2.0, "x_dot0": 0.0},  # Non-zero displacement, zero velocity
    {"x0": 0.0, "x_dot0": 2.0},  # Zero displacement, non-zero velocity
]


# Function to compute the system dynamics and characteristic roots
def system_dynamics_and_roots(t, omega_0, zeta, x0, x_dot0):
    if zeta < 1:
        damping_situation = "Underdamped"
        omega_d = omega_0 * np.sqrt(1 - zeta**2)  # Damped natural frequency
        A = x0
        B = (x_dot0 + zeta * omega_0 * x0) / omega_d
        x_t = np.exp(-zeta * omega_0 * t) * (
            A * np.cos(omega_d * t) + B * np.sin(omega_d * t)
        )
        x_dot_t = np.exp(-zeta * omega_0 * t) * (
            (omega_d * B - zeta * omega_0 * A) * np.cos(omega_d * t)
            + (-zeta * omega_0 * B - omega_d * A) * np.sin(omega_d * t)
        )
        roots = [-zeta * omega_0 + 1j * omega_d, -zeta * omega_0 - 1j * omega_d]
    elif zeta == 1:  # Critically damped
        damping_situation = "Critically Damped"
        A = x0
        B = x_dot0 + omega_0 * x0
        x_t = (A + B * t) * np.exp(-omega_0 * t)
        x_dot_t = B * np.exp(-omega_0 * t) - omega_0 * (A + B * t) * np.exp(
            -omega_0 * t
        )
        roots = [-omega_0, -omega_0]
    else:  # Overdamped
        damping_situation = "Overdamped"
        r1 = -omega_0 * (zeta + np.sqrt(zeta**2 - 1))
        r2 = -omega_0 * (zeta - np.sqrt(zeta**2 - 1))
        A = (x_dot0 - r2 * x0) / (r1 - r2)
        B = x0 - A
        x_t = A * np.exp(r1 * t) + B * np.exp(r2 * t)
        x_dot_t = A * r1 * np.exp(r1 * t) + B * r2 * np.exp(r2 * t)
        roots = [r1, r2]

    return x_t, x_dot_t, roots, damping_situation


# Function to add vector field to the phase diagram
def add_vector_field(ax, omega_0, zeta):
    X, X_Dot = np.meshgrid(np.linspace(-2, 2, 20), np.linspace(-2, 2, 20))
    U = X_Dot
    V = -(2 * zeta * omega_0) * X_Dot - (omega_0**2) * X
    ax.quiver(X, X_Dot, U, V, color=white, alpha=0.7, scale=50, width=0.003)


# Animation function
def animate(i, x_t, x_dot_t):
    # Update time plot
    line_time.set_data(t[:i], x_t[:i])
    dot_time.set_data([t[i]], [x_t[i]])  # Current position as red dot

    # Update phase plot
    line_phase.set_data(x_t[:i], x_dot_t[:i])
    dot_phase.set_data([x_t[i]], [x_dot_t[i]])  # Current position as red dot

    # Update mass position (moving vertically now)
    mass.set_data([0], [x_t[i]])  # Move the mass vertically
    spring.set_data([0, 0], [0, x_t[i]])  # Spring is now vertical

    return line_time, dot_time, line_phase, dot_phase, mass, spring


# Generate and save the animation for each case and each initial condition
for case in damping_cases:
    for i, ic in enumerate(initial_conditions):
        dt = 0.1
        t = np.arange(0, t_max + dt, dt)
        x_t, x_dot_t, _, damping_situation = system_dynamics_and_roots(
            t, case["omega_0"], case["zeta"], ic["x0"], ic["x_dot0"]
        )

        # Print damping situation
        print(
            f"Generating animation for {case['label']} case with initial condition set {i+1}: {damping_situation}"
        )

        # Create figure and axes for each animation
        fig, axs = plt.subplots(1, 3, figsize=(18, 6), facecolor="black")
        ax_system, ax_time, ax_phase = axs
        for ax in axs:
            ax.set_facecolor("black")
            ax.grid(True, color=white, alpha=0.2)
            ax.tick_params(colors=white)
            ax.set_yticks([-2, -1, 0, 1, 2])

        # Initialize plots
        (line_time,) = ax_time.plot([], [], color=light_blue)
        (dot_time,) = ax_time.plot([], [], "o", color=red)
        (line_phase,) = ax_phase.plot([], [], color=light_blue)
        (dot_phase,) = ax_phase.plot([], [], "o", color=red)
        (mass,) = ax_system.plot([], [], "o", color=red, markersize=10)
        (spring,) = ax_system.plot([], [], color=light_blue, lw=2)

        # Set up plot limits and labels with larger text
        ax_time.set_xlim(0, t_max)
        ax_time.set_ylim(-2.1, 2.1)
        ax_time.set_xticks([0, 5, 10, 15, 20])
        ax_time.set_xlabel("Time", color=light_blue, fontsize=14)
        ax_time.set_ylabel("x(t)", color=light_blue, fontsize=14)
        ax_time.set_title("Position over Time", color=light_blue, fontsize=16)

        ax_phase.set_xlim(-2.1, 2.1)
        ax_phase.set_ylim(-2.1, 2.1)
        ax_phase.set_xticks([-2, -1, 0, 1, 2])
        ax_phase.set_aspect("equal")
        ax_phase.set_xlabel("x", color=light_blue, fontsize=14)
        ax_phase.set_ylabel("x'", color=light_blue, fontsize=14)
        ax_phase.set_title("Phase Diagram", color=light_blue, fontsize=16)

        ax_system.set_xlim(-1, 1)
        ax_system.set_ylim(-2.1, 2.1)
        ax_system.set_xticks([])
        ax_system.set_aspect("equal")
        ax_system.set_ylabel("Position (x)", color=light_blue, fontsize=14)
        ax_system.set_yticks(np.arange(-2, 3, 1))
        ax_system.set_title("System Visualization", color=light_blue, fontsize=16)

        # Add vector field to the phase diagram
        add_vector_field(ax_phase, case["omega_0"], case["zeta"])

        # Create animation
        ani = FuncAnimation(
            fig, animate, frames=len(t), fargs=(x_t, x_dot_t), interval=100, blit=True
        )

        # Save the animation
        filename = f"{case['label']}_ic{i+1}.mp4"
        ani.save(filename, writer="ffmpeg", dpi=150)

        # Close the figure after saving
        plt.close(fig)

# Generate and save the combined plots
for i, ic in enumerate(initial_conditions):
    fig, axs = plt.subplots(1, 3, figsize=(24, 6), facecolor="black")
    ax_plot, ax_phase, ax_roots = axs
    for ax in axs:
        ax.set_facecolor("black")
        ax.grid(True, color=white, alpha=0.2)
        ax.tick_params(colors=white)
        ax.set_yticks([-2, -1, 0, 1, 2])
        ax.set_ylim(-2.1, 2.1)

    # Time evolution plot
    ax_plot.set_xlim(0, t_max)
    ax_plot.set_xticks([0, 5, 10, 15, 20])
    ax_plot.axhline(0, color=white, lw=0.5)  # x-axis at x=0
    ax_plot.set_xlabel("Time", color="white", fontsize=14)
    ax_plot.set_ylabel("x(t)", color="white", fontsize=14)
    ax_plot.set_title(
        f"Combined Damping Cases - Initial Condition {i+1}", color="white", fontsize=16
    )

    for case in damping_cases:
        t = np.linspace(0, t_max, 1000)
        x_t, x_dot_t, roots, damping_situation = system_dynamics_and_roots(
            t, case["omega_0"], case["zeta"], ic["x0"], ic["x_dot0"]
        )
        ax_plot.plot(t, x_t, label=damping_situation, color=case["color"])

        # Plot roots in the complex plane
        ax_roots.plot(
            [r.real for r in roots],
            [r.imag for r in roots],
            "o",
            color=case["color"],
            markersize=10,
            label=damping_situation,
        )

        # Plot phase plane trajectories
        ax_phase.plot(x_t, x_dot_t, label=damping_situation, color=case["color"])

    # Setup for the phase plane plot
    ax_phase.set_xlim(-2.1, 2.1)
    ax_phase.set_xticks([-2, -1, 0, 1, 2])
    ax_phase.set_aspect("equal")
    ax_phase.axhline(0, color=white, lw=0.5)
    ax_phase.axvline(0, color=white, lw=0.5)
    ax_phase.set_xlabel("x", color="white", fontsize=14)
    ax_phase.set_ylabel("x'", color="white", fontsize=14)
    ax_phase.set_title("Phase Plane Trajectories", color="white", fontsize=16)

    # Setup for the roots plot
    ax_roots.set_xlim(-4, 1)
    ax_roots.axhline(0, color=white, lw=0.5)
    ax_roots.axvline(0, color=white, lw=0.5)
    ax_roots.set_aspect("equal")
    ax_roots.set_xlabel("Re", color="white", fontsize=14)
    ax_roots.set_ylabel("Im", color="white", fontsize=14)
    ax_roots.set_title("Roots of Characteristic Polynomial", color="white", fontsize=16)

    # Add legend to the time evolution plot and phase plane plot
    for ax in [ax_plot, ax_phase, ax_roots]:
        legend = ax.legend(
            loc="best", fontsize=12, facecolor="black", edgecolor="white"
        )
        for text in legend.get_texts():
            text.set_color("white")

    # Save the combined plot
    plt.savefig(f"combined_ic{i+1}.png", dpi=150, facecolor="black")
    plt.close(fig)


# Compare vector fields for all three damping cases
fig, axs = plt.subplots(1, 3, figsize=(24, 6), facecolor="black")
for ax, case in zip(axs, damping_cases):
    ax.set_facecolor("black")
    ax.grid(True, color=white, alpha=0.2)
    ax.tick_params(colors=white)
    ax.set_xticks([-2, -1, 0, 1, 2])
    ax.set_yticks([-2, -1, 0, 1, 2])

    # Add vector field to each subplot
    add_vector_field(ax, case["omega_0"], case["zeta"])

    # Evaluate system
    _, _, _, damping_situation = system_dynamics_and_roots(
        t, case["omega_0"], case["zeta"], ic["x0"], ic["x_dot0"]
    )

    # Set up the axis labels and titles
    ax.set_xlim(-2.1, 2.1)
    ax.set_ylim(-2.1, 2.1)
    ax.set_aspect("equal")
    ax.axhline(0, color=white, lw=0.5)
    ax.axvline(0, color=white, lw=0.5)
    ax.set_xlabel("x", color="white", fontsize=14)
    ax.set_ylabel("x'", color="white", fontsize=14)
    ax.set_title(damping_situation, color="white", fontsize=16)

    add_vector_field(ax, case["omega_0"], case["zeta"])


# Save the vector field comparison plot
plt.savefig("vector_field_comparison.png", dpi=150, facecolor="black")
plt.close(fig)
