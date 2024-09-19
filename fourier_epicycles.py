import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec

# Parameters (common for all N)
t_max = 2 * np.pi     # Duration of one full cycle
dt = 0.01             # Time step for the animation
t_window = 2 * np.pi  # Window length for the moving plot

# Time array covering one full cycle
t = np.arange(0, t_max, dt)

# Convert CMYK (90, 57, 0, 0) to RGB
C, M, Y, K = 0.90, 0.57, 0.0, 0.0
R = 1 - min(1, C + K)
G = 1 - min(1, M + K)
B = 1 - min(1, Y + K)
custom_color = (R, G, B)  # (0.09999999999999998, 0.43, 1.0)

# Set figure DPI
dpi = 120

# Loop over N from 1 to 25
for N in range(1, 26):
    # Fourier coefficients for a square wave
    n = np.arange(1, 2*N, 2)        # Odd harmonics
    amplitudes = 4 / (np.pi * n)    # Amplitude of each harmonic
    frequencies = n                 # Frequency of each harmonic
    phases = np.zeros_like(n)       # Phase shifts (zero for sine terms)

    # Set up the figure and axes with adjusted width ratios
    fig = plt.figure(figsize=(16, 9), dpi=dpi)  # 16:9 aspect ratio
    fig.patch.set_facecolor('black')  # Set figure background to black

    # Create a GridSpec with adjusted width ratios
    gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1])

    # Left subplot: Rotating circles (epicycles)
    ax1 = fig.add_subplot(gs[0])
    ax1.set_aspect('equal')
    ax1.set_xlim(-3, 3)               # Set x-limits to provide more space
    ax1.set_ylim(-3, 3)               # Set y-limits to match x-limits
    ax1.axis('off')                   # Remove axes
    ax1.set_facecolor('black')        # Set axes background to black

    # Right subplot: Time domain approximation
    ax2 = fig.add_subplot(gs[1])
    ax2.set_xlim(0, t_window)         # Set x-limits for the moving plot
    ax2.set_ylim(-3, 3)               # Set y-limits to match ax1
    ax2.axis('off')                   # Remove axes
    ax2.set_facecolor('black')        # Set axes background to black

    # Adjust positions of subplots to fit the new aspect ratio
    ax1.set_position([0.05, 0.1, 0.7, 0.8])  # [left, bottom, width, height]
    ax2.set_position([0.8, 0.1, 0.15, 0.8])  # Adjusted to match ax1

    # Add text above both plots showing the number of Fourier modes
    fig.text(0.5, 0.95, f'Number of Fourier Modes: {N}', ha='center', va='center', color='white', fontsize=18)

    # Initialize lines and circles
    lines = []
    circles = []

    # Colors for lines
    line_color = 'white'
    circle_edge_color = 'gray'
    connector_color = custom_color     # Use custom color
    trace_color = custom_color         # Use custom color

    # Create lines and circles for each harmonic
    for i in range(len(n)):
        line, = ax1.plot([], [], 'o-', lw=2, color=line_color)
        lines.append(line)
        circle = plt.Circle(
            (0, 0),
            amplitudes[i],
            edgecolor=circle_edge_color,
            facecolor='none',
            linestyle='-',
            linewidth=1
        )
        ax1.add_patch(circle)
        circles.append(circle)

    # Line to plot the Fourier approximation in ax2
    trace2, = ax2.plot([], [], color=trace_color)
    lines.append(trace2)

    # Line to connect the tip of the epicycles to the signal across subplots (horizontal line)
    connector_line = Line2D(
        [],
        [],
        linestyle='--',
        lw=1,
        color=connector_color,
        transform=fig.transFigure
    )
    fig.add_artist(connector_line)

    # Lists to store trajectory data
    time_data = []
    func_data = []

    def init():
        """Initialize the animation."""
        for line in lines:
            line.set_data([], [])
        for circle in circles:
            circle.center = (0, 0)
        return lines + circles + [connector_line]

    def update(frame):
        """Update the animation for each frame."""
        time = t[frame]
        x = [0]
        y = [0]

        for i in range(len(n)):
            prev_x, prev_y = x[-1], y[-1]
            theta = frequencies[i] * time + phases[i]
            dx = amplitudes[i] * np.cos(theta)
            dy = amplitudes[i] * np.sin(theta)
            x_new = prev_x + dx
            y_new = prev_y + dy
            x.append(x_new)
            y.append(y_new)
            # Update vector lines
            lines[i].set_data([prev_x, x_new], [prev_y, y_new])
            # Update circles
            circles[i].center = (prev_x, prev_y)

        # The y-coordinate of the tip of the epicycles
        y_tip = y[-1]
        func_value = y_tip  # Use the same value to ensure alignment

        # Update time_data and func_data
        time_data.append(time)
        func_data.append(func_value)

        # Remove old data outside the window
        while time_data and time_data[-1] - time_data[0] > t_window:
            time_data.pop(0)
            func_data.pop(0)

        # Shift x_data so that the current time is at x=0 and plot moves to the right
        x_data = np.array(time_data) - time  # Shift so that current time is at x=0
        x_data = -x_data  # Make x_data positive and increasing

        # Update the line in ax2
        lines[-1].set_data(x_data, func_data)

        # Transform the coordinates for the horizontal connector line
        x_tip = x[-1]
        y_common = y_tip  # Use the y-coordinate of the tip

        # Transform the left point (epicycle tip)
        coord1 = ax1.transData.transform((x_tip, y_common))
        # Transform the right point (fixed at x=0 in ax2)
        coord2 = ax2.transData.transform((0, y_common))

        # Convert to figure coordinates
        inv = fig.transFigure.inverted()
        fig_coord1 = inv.transform(coord1)
        fig_coord2 = inv.transform(coord2)

        # Update the connector line
        connector_line.set_data([fig_coord1[0], fig_coord2[0]], [fig_coord1[1], fig_coord2[1]])

        return lines + circles + [connector_line]

    # Create the animation with blit=False
    ani = FuncAnimation(
        fig,
        update,
        frames=len(t),
        init_func=init,
        blit=False,
        interval=20
    )

    # Save the animation as an MP4 file
    filename = f'fourier_series_animation_N{N:02d}.mp4'
    ani.save(filename, writer='ffmpeg', fps=30, dpi=dpi)

    # Close the figure to free memory
    plt.close(fig)

    print(f'Saved animation for N={N} as {filename}')
