"""
Module's docstring to add
"""
from typing import Callable, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from numanalysislib.basis._abstract import PolynomialBasis

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
