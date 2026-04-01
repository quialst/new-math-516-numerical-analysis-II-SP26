import numpy as np
from numanalysislib.basis._abstract import PolynomialBasis
from numanalysislib.basis.power import PowerBasis
from numanalysislib.basis.affine import AffinePolynomialBasis

def differentiate(basis, coefficients):
    if basis.degree == 0:
        return PowerBasis(0), np.array([0.0])

    if isinstance(basis, PowerBasis):
        new_coeffs = np.array([i * coefficients[i] for i in range(1, len(coefficients))])
        new_basis = PowerBasis(basis.degree - 1)
    else:
        n = basis.degree
        x_pts = np.linspace(basis.a, basis.b, n + 1)
        y_pts = basis.evaluate(coefficients, x_pts)

        temp_basis = PowerBasis(n)
        temp_coeffs = temp_basis.fit(x_pts, y_pts)

        new_coeffs = np.array([i * temp_coeffs[i] for i in range(1, len(temp_coeffs))])
        new_basis = PowerBasis(n - 1)

    return new_basis, new_coeffs

def evaluate_derivative(basis, coefficients, x, h=1e-7):
    f_next = basis.evaluate(coefficients, x + h)
    f_prev = basis.evaluate(coefficients, x - h)
    deriv = (f_next - f_prev) / (2 * h)

    return deriv