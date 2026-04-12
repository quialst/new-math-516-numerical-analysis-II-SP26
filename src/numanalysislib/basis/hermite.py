"""
Hermite cubic polynomial basis functions.

p(x) = v_a * h_0(x) + d_a * h_1(x) * (b-a) + v_b * h_2(x) + d_b * h_3(x) * (b-a)
"""

import numpy as np

from numanalysislib.basis._abstract import PolynomialBasis


class HermiteBasis(PolynomialBasis):

    def __init__(self):
        super().__init__(degree=3, a=0.0, b=1.0)

    def evaluate_basis(self, index: int, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the i-th Hermite basis function at point x. 
            h_0 = 2x^3 - 3x^2 + 1    
            h_1 = x^3 - 2x^2 + x     
            h_2 = -2x^3 + 3x^2       
            h_3 = x^3 - x^2    

        Parameters
        ----------
        index: the index associated to the basis you want to evaluate
        x: The value of x at which to evaluate the polynomial basis.            

        Returns 
        ----------   
        The value of the basis at x.      
        """
        if index < 0 or index > self.degree:
            raise ValueError(f"Basis index {index} out of range for degree {self.degree}")

        x = np.asarray(x, dtype=float)
        x2 = x * x
        x3 = x2 * x

        if index == 0:
            return 2.0 * x3 - 3.0 * x2 + 1.0
        if index == 1:
            return x3 - 2.0 * x2 + x
        if index == 2:
            return -2.0 * x3 + 3.0 * x2
        if index == 3:
            return x3 - x2

    def fit(self, x_nodes: np.ndarray, y_nodes: np.ndarray, physical_interval: tuple = None) -> np.ndarray:
        """
        Compute coefficients from endpoint data [v_a, d_a, v_b, d_b].

        Parameters
        ----------
        x_nodes: [a, b] defining the interval.
        y_nodes: [v_a, d_a, v_b, d_b], where v_a and v_b are values at two endpoints, d_a and d_b are the derivatives.
        physical_interval: Optional tuple (a, b) to specify the physical interval for scaling derivative coefficients.
        
        Returns
        -------
        Coefficients: [v_a, (b - a) * d_a, v_b, (b - a) * d_b]. 
        """
        x_nodes = np.asarray(x_nodes, dtype=float)
        y_nodes = np.asarray(y_nodes, dtype=float)

        if x_nodes.shape[0] != 2:
            raise ValueError("HermiteBasis.fit expects the interval as [a, b].")
        if y_nodes.shape[0] != self.n_dofs:
            raise ValueError(f"HermiteBasis.fit expects {self.n_dofs} data entries [v_a, d_a, v_b, d_b].")

        # Determine scaling for derivative coefficients. 
        # If a physical interval is provided (from an Affine wrapper), use that; otherwise infer from x_nodes.
        if physical_interval is not None:
            a, b = physical_interval
        else:
            a, b = x_nodes

        scale = b - a
        if scale <= 0.0:
            raise ValueError("Interval endpoints must satisfy b > a.")

        v_a, d_a, v_b, d_b = y_nodes

        coefficients = np.array(
            [v_a, scale * d_a, v_b, scale * d_b],
            dtype=float,
        )

        return coefficients
    
    # Use the default evaluate method from the abstract class
