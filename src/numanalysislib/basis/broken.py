from numanalysislib.basis._abstract import PolynomialBasis
from numanalysislib.basis.affine import AffinePolynomialBasis
import numpy as np


class BrokenPolynomialBasis(PolynomialBasis):
    def __init__(self, basis_type: PolynomialBasis, mesh: np.ndarray):
        """
        Discontinuous Galerkin (broken) polynomial basis.

        Each element has its own independent polynomial basis.
        No continuity is enforced between elements.

        DOFs = N_elements * (degree + 1)
        """
        self.basis_type = basis_type

        # Build mesh as list of intervals
        # xmpl: [0,1,2] → [(0,1), (1,2)]
        self.mesh = [(mesh[i], mesh[i+1]) for i in range(len(mesh)-1)]

        # Base basis info
        self.degree = basis_type.degree
        self.local_dofs = basis_type.n_dofs

        # Total DOFs
        self.n_elements = len(self.mesh)
        total_dofs = self.n_elements * self.local_dofs

        super().__init__(degree=self.degree,
                         a=mesh[0],
                         b=mesh[-1])

        self.n_dofs = total_dofs  # override

        # Create affine bases per element
        self.bases = {}
        for element in self.mesh:
            a, b = element
            self.bases[element] = AffinePolynomialBasis(basis_type, a, b)

    # Basis evaluation (global index → local element)
    def evaluate_basis(self, index: int, x: np.ndarray) -> np.ndarray:
        """
        Evaluate global basis function.

        Global index → (element, local index)
        """
        x = np.asarray(x)

        element_id = index // self.local_dofs
        local_index = index % self.local_dofs

        element = self.mesh[element_id]
        a, b = element

        # Indicator: only active on its element
        mask = (x >= a) & (x <= b)

        result = np.zeros_like(x, dtype=float)
        result[mask] = self.bases[element].evaluate_basis(local_index, x[mask])

        return result

    #  L2 Projection (fit)
    def fit(self, f, quad_order: int = 5) -> dict:
        """
        L2 projection of function f onto broken polynomial space.

        Returns:
            dict: element -> coefficients
        """
        coeffs = {}

        for element in self.mesh:
            a, b = element
            basis = self.bases[element]

            n = self.local_dofs

            # Quadrature points (simple Gauss-Legendre)
            xi, wi = np.polynomial.legendre.leggauss(quad_order)

            # Map to [a, b]
            xq = 0.5*(b - a)*xi + 0.5*(a + b)
            wq = 0.5*(b - a)*wi

            # Build mass matrix M_ij = ∫ φ_i φ_j
            M = np.zeros((n, n))
            for i in range(n):
                phi_i = basis.evaluate_basis(i, xq)
                for j in range(n):
                    phi_j = basis.evaluate_basis(j, xq)
                    M[i, j] = np.sum(wq * phi_i * phi_j)

            # RHS: b_i = ∫ f φ_i
            b_vec = np.zeros(n)
            f_vals = f(xq)

            for i in range(n):
                phi_i = basis.evaluate_basis(i, xq)
                b_vec[i] = np.sum(wq * f_vals * phi_i)

            # Solve local system
            coeffs[element] = np.linalg.solve(M, b_vec)

        return coeffs


    def float_evaluate(self, coefficients: dict, x: float) -> float:
        """
        Evaluate DG polynomial at a single point.
        """
        for element in self.mesh:
            a, b = element
            if a <= x <= b:
                return self.bases[element].evaluate(coefficients[element], np.array([x]))[0]

        return 0.0  # outside domain

    def evaluate(self, coefficients: dict, x: np.ndarray) -> np.ndarray:
        """
        Vectorized evaluation
        """
        x = np.asarray(x)
        return np.vectorize(lambda xi: self.float_evaluate(coefficients, xi))(x)