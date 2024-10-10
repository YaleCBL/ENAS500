import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Create a figure and axis with black background
fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=100)
fig.patch.set_facecolor('black')
ax.set_facecolor('black')

# Set up the axis labels and ticks
ax.grid(True)
ax.set_xlim(0, 6*np.pi)  # Expanded domain by a factor of 3
ax.set_ylim(-2.5, 2.5)  # Adjusted to accommodate new y-ticks
ax.set_xticks([0, 2*np.pi, 4*np.pi, 6*np.pi])
ax.set_xticklabels(['0', '2π', '4π', '6π'], color='white', fontsize=20)
ax.set_yticks([2, 1, 0, -1, -2])  # Added +2 and -2 to y-ticks
ax.set_yticklabels(['2', '1', '0', '-1', '-2'], color='white', fontsize=20)

# Set labels and title
ax.set_xlabel('x', color='white', fontsize=20)
ax.set_ylabel('Amplitude', color='white', fontsize=20)

# Initialize lines for the animations
line1, = ax.plot([], [], color='blue', lw=2, label='Wave 1')
line2, = ax.plot([], [], color='yellow', lw=2, label='Wave 2')
line3, = ax.plot([], [], color='red', lw=2, label='Standing Wave')

# Initialization function
def init():
    line1.set_data([], [])
    line2.set_data([], [])
    line3.set_data([], [])
    return line1, line2, line3

# Animation function
def animate(t):
    x = np.linspace(0, 6*np.pi, 1000)  # Adjust x to the expanded domain
    wave1 = np.sin(x - 0.025*t)  # Quarter speed compared to the original
    wave2 = np.sin(x + 0.025*t)  # Quarter speed compared to the original
    standing_wave = wave1 + wave2  # Standing wave
    line1.set_data(x, wave1)
    line2.set_data(x, wave2)
    line3.set_data(x, standing_wave)
    return line1, line2, line3

# Number of frames to match the wave period for periodicity
fps = 40  # Frames per second
period_time = 2*np.pi  # Time for one period of the wave
total_frames = int(period_time * fps)  # Total frames for one complete period

# Create the animation
ani = animation.FuncAnimation(fig, animate, init_func=init, frames=total_frames, interval=1000/fps, blit=True)

# Save the animation
ani.save('standing_wave.mp4', writer='ffmpeg', dpi=200, fps=fps)

plt.close()  # Close the plot to avoid displaying it here
