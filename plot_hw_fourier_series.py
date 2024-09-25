import numpy as np
import matplotlib.pyplot as plt

# Define the time variable t
t = np.linspace(-np.pi, np.pi, 1000)

# Define the original functions
def f(t):
    return 1 - np.abs(t)

def g(t):
    return t**2

def h(t):
    return np.abs(np.sin(t))

def k(t):
    return np.sin(t)**3

# Correct Fourier Series approximations based on provided series
def fourier_series_f(t, n):
    a0 = 1 - np.pi/2
    series = a0
    for i in range(1, 2*n+1, 2):  # sum only over odd n
        series += (4 / (np.pi * i**2)) * np.cos(i * t)
    return series

def fourier_series_g(t, n):
    a0 = np.pi**2 / 3
    series = a0
    for i in range(1, n+1):
        series += (4 / i**2) * (-1)**i * np.cos(i * t)
    return series

def fourier_series_h(t, n):
    a0 = 2 / np.pi
    series = a0
    for i in range(2, 2*n+1, 2):  # sum only over even n
        series += (4 / np.pi) * (1 / (1 - i**2)) * np.cos(i * t)
    return series

def fourier_series_k(t, n):
    # Fixed series for k(t) = 3/4 sin(t) - 1/4 sin(3t)
    if n >= 1:
        return 3/4 * np.sin(t) - 1/4 * np.sin(3*t)
    return 0

# Fourier modes to plot
fourier_modes = [1, 3, 10]

# Define a function to format x-axis with -π, 0, and π
def set_pi_ticks(ax):
    ax.set_xticks([-np.pi, 0, np.pi])
    ax.set_xticklabels([r'$-\pi$', '0', r'$\pi$'])

# Define the plotting function (but do not display)
def plot_function_and_fourier(t, func, fourier_func, title, filename):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=True)  # Share y-axis between plots
    
    for idx, n in enumerate(fourier_modes):
        axs[idx].plot(t, func(t), 'k', label='Original')
        axs[idx].plot(t, fourier_func(t, n), 'r--', label='Fourier Series')  # Dotted line for Fourier
        term_label = 'term' if n == 1 else 'terms'  # Dynamically adjust "term" or "terms"
        axs[idx].set_title(f'{title} ({n} {term_label})')
        axs[idx].grid(True)  # Add grid
        set_pi_ticks(axs[idx])
        axs[idx].legend()
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()  # Close the plot to avoid displaying it

# Plot and save each function and its Fourier series
plot_function_and_fourier(t, f, fourier_series_f, 'f(t) = 1 - |t|', 'fourier_series_function_f.png')
plot_function_and_fourier(t, g, fourier_series_g, 'g(t) = t^2', 'fourier_series_function_g.png')
plot_function_and_fourier(t, h, fourier_series_h, 'h(t) = |sin t|', 'fourier_series_function_h.png')
plot_function_and_fourier(t, k, fourier_series_k, 'k(t) = sin^3 t', 'fourier_series_function_k.png')
