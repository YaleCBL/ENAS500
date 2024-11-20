# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt

# Define the systems
systems = [
    {
        'name': 'Problem 5',
        'filename': 'section_14-1_problem5_vector_field.pdf',
        'matrix': np.array([[7, -17],
                            [2, 1]]),
    },
    {
        'name': 'Problem 9',
        'filename': 'section_14-1_problem9_vector_field.pdf',
        'matrix': np.array([[-2, -1],
                            [3, -2]]),
    },
    {
        'name': 'Problem 11',
        'filename': 'section_14-1_problem11_vector_field.pdf',
        'matrix': np.array([[2, 1],
                            [1, -2]]),
    }
]

# Loop over the systems and plot each vector field
for sys in systems:
    A = sys['matrix']
    name = sys['name']
    filename = sys['filename']
    
    # Define the system of differential equations
    def system(X):
        x, y = X
        dxdt = A[0, 0]*x + A[0, 1]*y
        dydt = A[1, 0]*x + A[1, 1]*y
        return dxdt, dydt
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    # Create a grid of points
    x = np.linspace(-5, 5, 25)
    y = np.linspace(-5, 5, 25)
    X_grid, Y_grid = np.meshgrid(x, y)

    # Compute the derivatives at each grid point
    DX, DY = system([X_grid, Y_grid])

    # Normalize the vectors for better visualization
    M = np.hypot(DX, DY)
    M[M == 0] = 1  # Avoid division by zero
    DX_norm = DX / M
    DY_norm = DY / M

    # Plot the vector field
    plt.figure(figsize=(8, 8))
    plt.quiver(X_grid, Y_grid, DX_norm, DY_norm, pivot='mid', color='black', alpha=0.8)

    # Plot eigenvectors only if eigenvalues are real
    if np.isreal(eigenvalues).all():
        # Plot the eigenvectors as red lines
        for i in range(len(eigenvalues)):
            eigenvalue = eigenvalues[i]
            eigenvector = eigenvectors[:, i].real  # Ensure it's real
            eigenvector /= np.linalg.norm(eigenvector)
            # Create points along the eigenvector direction
            t = np.linspace(-5, 5, 10)
            x_ev = t * eigenvector[0]
            y_ev = t * eigenvector[1]
            plt.plot(x_ev, y_ev, 'r-', linewidth=2, label=f'Eigenvector {i+1}' if i == 0 else "")
        # plt.legend(loc='upper left')
    # else:
    #     plt.text(-4.5, 4.5, 'Complex eigenvalues\nEigenvectors not plotted', fontsize=10, color='red')

    # plt.title(f"Vector Field for {name}")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.axis('equal')  # Make axes equal
    plt.grid(True)

    # Save the plot as a PDF
    plt.savefig(filename, format='pdf')
    # plt.show()
