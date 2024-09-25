
import numpy as np
import matplotlib.pyplot as plt

# Define larger font size
plt.rcParams.update({'font.size': 22})
plt.style.use('dark_background')  # Ensure black background

# Define the rectangular pulse function
a = 1  # Pulse width
t_rect = np.linspace(-2*a, 2*a, 400)
rect_pulse = np.where(np.abs(t_rect) <= a, 1, 0)
omega_rect = np.linspace(-10, 10, 400)
fourier_rect = 2 * a * np.sinc(omega_rect * a / np.pi)

# Define the one-sided exponential function
t_one_sided = np.linspace(0, 5, 400)
one_sided_exp = np.exp(-t_one_sided)
fourier_one_sided = 1 / np.sqrt(omega_rect**2 + 1)

# Define the two-sided exponential function
t_two_sided = np.linspace(-5, 5, 400)
two_sided_exp = np.exp(-np.abs(t_two_sided))
fourier_two_sided = 2 / (omega_rect**2 + 1)

# Function to create and save the plot
def plot_exponential_and_fourier(t, exp_function, fourier_function, title, file_name):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(19.2, 10.8))  # Adjusted figure size
    ax1.plot(t, exp_function, 'r', label=title + ' in Time Domain')
    ax1.set_title(title + ' in Time Domain', color='white')
    ax1.set_xlabel('t', color='white')
    ax1.set_ylabel('Amplitude', color='white')
    ax1.grid(True, color='gray')
    ax1.tick_params(colors='white')

    ax2.plot(omega_rect, fourier_function, 'r', label=title + ' Fourier Transform')
    ax2.set_title(title + ' Fourier Transform', color='white')
    ax2.set_xlabel(r'$\omega$', color='white')
    ax2.set_ylabel('Magnitude', color='white')
    ax2.grid(True, color='gray')
    ax2.tick_params(colors='white')

    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()

# Create and save plots for the rectangular pulse
plot_exponential_and_fourier(t_rect/a, rect_pulse, fourier_rect, 
                             'Rectangular Pulse', 
                             'rectangular_pulse_fourier_transform.pdf')

# Create and save plots for the one-sided exponential
plot_exponential_and_fourier(t_one_sided, one_sided_exp, fourier_one_sided, 
                             'One-sided Exponential', 
                             'one_sided_exponential_fourier_transform.pdf')

# Create and save plots for the two-sided exponential
plot_exponential_and_fourier(t_two_sided, two_sided_exp, fourier_two_sided, 
                             'Two-sided Exponential', 
                             'two_sided_exponential_fourier_transform.pdf')

