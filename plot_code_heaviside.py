
import numpy as np
import matplotlib.pyplot as plt

# Define the time domain and shift value
t = np.linspace(0, 4*np.pi, 500)
a = 3 * np.pi / 4  # Shift

# Define the functions
sin_t = np.sin(t)
sin_t_a = np.sin(t - a)
H_t_a = np.heaviside(t - a, 1)
H_t_a_sin_t_a = H_t_a * np.sin(t - a)

# CMYK color in RGB format (approximation for CMYK 90, 57, 0, 0)
cmyk_color = (0.1, 0.43, 1.0)  # Cyan dominance

# Create the plot with shared y-axis and updated parameters
plt.figure(figsize=(19.2, 10.8))
plt.style.use('dark_background')

# Larger text size for labels and titles
label_size = 14
title_size = 16

# Plot sin(t)
plt.subplot(2, 2, 1)
plt.plot(t, sin_t, color=cmyk_color, label=r'$\sin(t)$')
plt.title(r'$\sin(t)$', color='white', fontsize=title_size)
plt.grid(True, color='gray')
plt.xticks(ticks=[0, np.pi, 2*np.pi, 3*np.pi, 4*np.pi], 
           labels=['0', r'$\pi$', r'$2\pi$', r'$3\pi$', r'$4\pi$'], color='white', fontsize=label_size)
plt.yticks(ticks=[-1, 0, 1], color='white', fontsize=label_size)
plt.xlabel('t', color='white', fontsize=label_size)
plt.ylim([-1.2, 1.2])

# Plot sin(t - a)
plt.subplot(2, 2, 2)
plt.plot(t, sin_t_a, color=cmyk_color, label=r'$\sin(t - a)$')
plt.title(r'$\sin(t - \frac{3\pi}{4})$', color='white', fontsize=title_size)
plt.grid(True, color='gray')
plt.xticks(ticks=[0, np.pi, 2*np.pi, 3*np.pi, 4*np.pi], 
           labels=['0', r'$\pi$', r'$2\pi$', r'$3\pi$', r'$4\pi$'], color='white', fontsize=label_size)
plt.yticks(ticks=[-1, 0, 1], color='white', fontsize=label_size)
plt.xlabel('t', color='white', fontsize=label_size)
plt.ylim([-1.2, 1.2])

# Plot H(t - a)
plt.subplot(2, 2, 3)
plt.plot(t, H_t_a, color=cmyk_color, label=r'$H(t - a)$')
plt.title(r'$H(t - \frac{3\pi}{4})$', color='white', fontsize=title_size)
plt.grid(True, color='gray')
plt.xticks(ticks=[0, np.pi, 2*np.pi, 3*np.pi, 4*np.pi], 
           labels=['0', r'$\pi$', r'$2\pi$', r'$3\pi$', r'$4\pi$'], color='white', fontsize=label_size)
plt.yticks(ticks=[-1, 0, 1], color='white', fontsize=label_size)
plt.xlabel('t', color='white', fontsize=label_size)
plt.ylim([-1.2, 1.2])

# Plot H(t - a) * sin(t - a)
plt.subplot(2, 2, 4)
plt.plot(t, H_t_a_sin_t_a, color=cmyk_color, label=r'$H(t - a) \sin(t - a)$')
plt.title(r'$H(t - \frac{3\pi}{4}) \sin(t - \frac{3\pi}{4})$', color='white', fontsize=title_size)
plt.grid(True, color='gray')
plt.xticks(ticks=[0, np.pi, 2*np.pi, 3*np.pi, 4*np.pi], 
           labels=['0', r'$\pi$', r'$2\pi$', r'$3\pi$', r'$4\pi$'], color='white', fontsize=label_size)
plt.yticks(ticks=[-1, 0, 1], color='white', fontsize=label_size)
plt.xlabel('t', color='white', fontsize=label_size)
plt.ylim([-1.2, 1.2])

# Adjust layout and show the plot with shared y-axis and larger text
plt.tight_layout()
plt.show()
