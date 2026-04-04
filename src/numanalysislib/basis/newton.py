from numanalysislib.basis._abstract import PolynomialBasis
import numpy as np


class NewtonPolynomialBasis(PolynomialBasis):
    def __init__(self, x_nodes: np.ndarray):
        
        # Newton polynomial basis defined by interpolation nodes.
        
        self.x_nodes = np.asarray(x_nodes)
        degree = len(x_nodes) - 1

        super().__init__(degree=degree,
                         a=np.min(x_nodes),
                         b=np.max(x_nodes))


    # Basis functions
    # n_i(x) = Π_{j=0}^{i-1} (x - x_j)

    def evaluate_basis(self, index: int, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)

        if index == 0:
            return np.ones_like(x, dtype=float)

        result = np.ones_like(x, dtype=float)
        
        for j in range(index):
            result *= (x - self.x_nodes[j])

        return result


    # Divided Differences (fit)

    def fit(self, x_nodes: np.ndarray, y_nodes: np.ndarray) -> np.ndarray:
        x_nodes = np.asarray(x_nodes)
        y_nodes = np.asarray(y_nodes)

        if len(x_nodes) != self.n_dofs:
            raise ValueError("Number of nodes must match basis dimension")

        coef = y_nodes.astype(float).copy()
        n = len(coef)

        # In-place divided differences
        for j in range(1, n):
            coef[j:n] = (
                (coef[j:n] - coef[j-1:n-1]) /
                (x_nodes[j:n] - x_nodes[0:n-j])
            )

        return coef


    # Efficient evaluation (Horner for Newton form)

    def evaluate(self, coefficients: np.ndarray, x: np.ndarray) -> np.ndarray:
        if len(coefficients) != self.n_dofs:
            raise ValueError(
                f"Expected {self.n_dofs} coefficients, got {len(coefficients)}"
            )

        x = np.asarray(x, dtype=float)

        # Horner scheme for Newton form
        result = np.zeros_like(x) + coefficients[-1]

        for k in range(self.n_dofs - 2, -1, -1):
            result = result * (x - self.x_nodes[k]) + coefficients[k]

        return result