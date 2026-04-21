"""
Module's docstring to add
"""
from typing import Callable, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from numanalysislib.basis._abstract import PolynomialBasis
from numanalysislib.basis.tensor import TensorProductBasis

class Plotter:
    """
    A generic visualization tool for PolynomialBasis subclasses.
    TODO: improve docs, consider passing the PolynomialBasis to the constructor
    """
    def plot_basis(self, basis: PolynomialBasis, domain: Tuple[float, float] = (0, 1),
                   title: str = None):
        """
        Plots all basis functions phi_i(x) over the given domain.
        TODO: improve docs, consider defaulting to PolynomialBasis.domain
        """
        x = np.linspace(domain[0], domain[1], 500)
        plt.figure(figsize=(10, 6))
        
        # Color map for distinct lines
        colors = plt.cm.viridis(np.linspace(0, 1, basis.n_dofs))

        for i in range(basis.n_dofs):
            y = basis.evaluate_basis(i, x)
            plt.plot(x, y, label=fr'$\phi_{{{i}}}(x)$', color=colors[i], linewidth=2)
            
        plt.title(title or f"Basis Functions (Degree {basis.degree})")
        plt.xlabel("x")
        plt.ylabel("Value")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

    def plot_fit(self, basis: PolynomialBasis, coefficients: np.ndarray, 
                 x_nodes: np.ndarray, y_nodes: np.ndarray, 
                 domain: Tuple[float, float], 
                 true_func: Optional[Callable[[np.ndarray], np.ndarray]] = None):
        """
        Plots the interpolated polynomial against the data nodes and (optionally) the true function.
        TODO: improve docs
        """
        x_dense = np.linspace(domain[0], domain[1], 500)
        y_approx = basis.evaluate(coefficients, x_dense)
        
        plt.figure(figsize=(10, 6))
        
        # plot true function (if provided)
        if true_func:
            y_true = true_func(x_dense)
            plt.plot(x_dense, y_true, 'k--', label=r'True Function $f(x)$', alpha=0.6)
            
        # plot approximation
        plt.plot(x_dense, y_approx, 'b-', linewidth=2.5,
                 label=f'Approximation (Deg {basis.degree})')
        
        # plot data nodes
        plt.scatter(x_nodes, y_nodes, color='red', s=80, zorder=5, edgecolors='black',
                    label='Interpolation Nodes')
        
        plt.title(f"Polynomial Interpolation with {basis.__class__.__name__}")
        plt.xlabel("x")
        plt.legend()
        plt.show()

    def plot_basis_tensor(self, basis: TensorProductBasis, 
                          domain_x: Tuple[float, float], 
                          domain_y: Tuple[float, float],
                          title: str = None):
        """
        Plots all 2D tensor product basis functions as 3D surface plots.
        
        Args:
            basis: A TensorProductBasis instance.
            domain_x: Tuple (ax, bx) specifying the x-domain.
            domain_y: Tuple (ay, by) specifying the y-domain.
            title: Optional overall title for the figure.
        
        Raises:
            TypeError: If basis is not a TensorProductBasis instance.
        """
        if not isinstance(basis, TensorProductBasis):
            raise TypeError(f"Expected TensorProductBasis, got {type(basis).__name__}")
        
        n_dofs = basis.n_dofs
        nx = basis.nx
        ny = basis.ny
        
        # Compute subplot grid layout: ceil(sqrt(n_dofs)) rows and columns
        grid_size = int(np.ceil(np.sqrt(n_dofs)))
        
        # Create dense grid for evaluation
        x_dense = np.linspace(domain_x[0], domain_x[1], 35)
        y_dense = np.linspace(domain_y[0], domain_y[1], 35)
        X, Y = np.meshgrid(x_dense, y_dense)
        
        fig = plt.figure(figsize=(15, 12))
        
        for flat_idx in range(n_dofs):
            ax = fig.add_subplot(grid_size, grid_size, flat_idx + 1, projection='3d')
            
            # Evaluate basis function
            Z = basis.evaluate_basis(flat_idx, x_dense, y_dense)
            
            # Plot wireframe surface
            ax.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.8, edgecolor='none')
            
            # Set labels and title
            i, j = basis._unflatten_index(flat_idx)
            ax.set_title(fr'$\phi_{{{i},{j}}}(x,y)$', fontsize=10)
            ax.set_xlabel('x', fontsize=8)
            ax.set_ylabel('y', fontsize=8)
            ax.set_zlabel('z', fontsize=8)
            ax.tick_params(labelsize=7)
        
        fig.suptitle(title or f"Tensor Product Basis Functions (nx={nx}, ny={ny})", fontsize=14)
        plt.tight_layout()
        plt.show()

    def plot_fit_tensor(self, basis: TensorProductBasis, 
                        coefficients: np.ndarray,
                        x_nodes: np.ndarray, 
                        y_nodes: np.ndarray,
                        domain_x: Tuple[float, float],
                        domain_y: Tuple[float, float],
                        true_func: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
                        title: str = None):
        """
        Plots a fitted 2D tensor product surface and optionally the true function.
        
        Args:
            basis: A TensorProductBasis instance.
            coefficients: 2D array of shape (nx, ny) with basis coefficients.
            x_nodes: 1D array of x node locations used for fitting.
            y_nodes: 1D array of y node locations used for fitting.
            domain_x: Tuple (ax, bx) specifying the x-domain for plotting.
            domain_y: Tuple (ay, by) specifying the y-domain for plotting.
            true_func: Optional callable(x, y) → z for comparison overlay.
            title: Optional title for the figure.
        
        Raises:
            TypeError: If basis is not a TensorProductBasis instance.
            ValueError: If coefficients shape doesn't match (nx, ny).
        """
        if not isinstance(basis, TensorProductBasis):
            raise TypeError(f"Expected TensorProductBasis, got {type(basis).__name__}")
        
        if coefficients.shape != (basis.nx, basis.ny):
            raise ValueError(f"Coefficients shape {coefficients.shape} must match (nx={basis.nx}, ny={basis.ny})")
        
        # Determine number of subplots
        n_subplots = 2 if true_func is not None else 1
        
        # Create dense grid for evaluation
        x_dense = np.linspace(domain_x[0], domain_x[1], 40)
        y_dense = np.linspace(domain_y[0], domain_y[1], 40)
        X, Y = np.meshgrid(x_dense, y_dense)
        
        # Evaluate fitted surface
        Z_fit = basis.evaluate(coefficients, x_dense, y_dense)
        
        fig = plt.figure(figsize=(14, 6) if n_subplots == 2 else (7, 6))
        
        # Plot fitted surface (left or only subplot)
        ax1 = fig.add_subplot(1, n_subplots, 1, projection='3d')
        ax1.plot_surface(X, Y, Z_fit, cmap=cm.viridis, alpha=0.8, edgecolor='none')
        
        # Overlay scatter points of fitting nodes
        Z_nodes = basis.evaluate(coefficients, x_nodes, y_nodes)
        # Extract diagonal values that correspond to the node grid
        X_nodes, Y_nodes = np.meshgrid(x_nodes, y_nodes)
        ax1.scatter(X_nodes, Y_nodes, Z_nodes, color='red', s=50, marker='o', 
                   edgecolors='black', linewidth=1, label='Fitting Nodes', zorder=5)
        
        ax1.set_xlabel('x', fontsize=10)
        ax1.set_ylabel('y', fontsize=10)
        ax1.set_zlabel('z', fontsize=10)
        ax1.set_title('Fitted Surface', fontsize=12)
        ax1.legend(fontsize=9)
        
        # Plot true function if provided
        if true_func is not None:
            ax2 = fig.add_subplot(1, 2, 2, projection='3d')
            Z_true = true_func(X, Y)
            ax2.plot_surface(X, Y, Z_true, cmap=cm.plasma, alpha=0.8, edgecolor='none')
            ax2.set_xlabel('x', fontsize=10)
            ax2.set_ylabel('y', fontsize=10)
            ax2.set_zlabel('z', fontsize=10)
            ax2.set_title('True Function', fontsize=12)
        
        fig.suptitle(title or f"Tensor Product Surface Fit (nx={basis.nx}, ny={basis.ny})", fontsize=14)
        plt.tight_layout()
        plt.show()
