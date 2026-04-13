import numpy as np
from numanalysislib.basis._abstract import PolynomialBasis

class TensorProductBasis(PolynomialBasis):
    def __init__(self, bx: PolynomialBasis, by: PolynomialBasis):
        self.bx = bx
        self.by = by
        self.nx = bx.n_dofs
        self.ny = by.n_dofs
        total_dofs = self.nx * self.ny

        super().__init__(total_dofs - 1, bx.a, by.a)

    def _unflatten_index(self, index: int):
        """Helper function to loop through vectors"""
        return divmod(index, self.ny)

    def evaluate_basis(self, index: int, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Override: Evaluates a single 2D basis function defined by the tensor product 
        of 1D basis functions.

            Args:
                index: The global flat index of the 2D basis function to evaluate.
                x: An array of x-coordinates for evaluation.
                y: An array of y-coordinates for evaluation.

            Returns:
                A 2D array of shape (len(x), len(y)) containing the values of 
                Phi_ij(x, y) = phi_i(x) * psi_j(y).
        """
        if index < 0 or index >= self.n_dofs:
            raise ValueError(f"Basis index {index} out of range")
        
        i, j = self._unflatten_index(index)
        
        phi_i = self.bx.evaluate_basis(i, x)
        psi_j = self.by.evaluate_basis(j, y)
        
        return np.outer(phi_i, psi_j)

    def evaluate(self, coefficients: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Override: Evaluates the 2D tensor product surface at the given grid of points.

            Args:
                coefficients: A 2D array of shape (nx, ny) containing the basis 
                              coefficients Cij.
                x: An array of x-coordinates (length M) for evaluation.
                y: An array of y-coordinates (length N) for evaluation.

            Returns:
                A 2D array of shape (M, N) representing the evaluated surface 
                values at the grid defined by the Cartesian product of x and y.
        """
        if coefficients.size != self.n_dofs:
            raise ValueError(f"Expected {self.n_dofs} coefficients, got {coefficients.size}")
        
        res = np.zeros((len(x), len(y)))

        for i in range(self.nx):
            for j in range(self.ny):
                c = coefficients[i, j]
                if c != 0:
                    res += c * self.evaluate_basis(i * self.ny + j, x, y)
        return res

    def fit(self, x_nodes: np.ndarray, y_nodes: np.ndarray, z_values: np.ndarray) -> np.ndarray:
        """
        Override: Computes the 2D basis coefficients by performing successive 1D fits 
        along each dimension.

            Args:
                x_nodes: 1D array of coordinates along the x-axis.
                y_nodes: 1D array of coordinates along the y-axis.
                z_values: 2D array of shape (len(x_nodes), len(y_nodes)) containing 
                          the target values to be fitted at each (x, y) node pair.

            Returns:
                A 2D array of coefficients of shape (nx, ny) that reconstructs 
                the input grid when passed to the evaluate method.
        """
        x_nodes = np.asarray(x_nodes)
        y_nodes = np.asarray(y_nodes)
        z_values = np.asarray(z_values)

        if z_values.shape != (len(x_nodes), len(y_nodes)):
            raise ValueError(f"z_values shape {z_values.shape} must match nodes grid.")

        intermediate_coeffs = np.zeros((self.nx, len(y_nodes)))
        for j in range(len(y_nodes)):
            intermediate_coeffs[:, j] = self.bx.fit(x_nodes, z_values[:, j])

        final_coeffs_matrix = np.zeros((self.nx, self.ny))
        for i in range(self.nx):
            final_coeffs_matrix[i, :] = self.by.fit(y_nodes, intermediate_coeffs[i, :])

        return final_coeffs_matrix