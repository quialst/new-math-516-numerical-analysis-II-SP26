import numpy as np
import pytest
from numanalysislib.basis.power import PowerBasis
from numanalysislib.basis.affine import AffinePolynomialBasis
from numanalysislib.calculus.differentiator import differentiate, evaluate_derivative


class TestDifferentiate:

    def test_constant(self):
        # derivative of 2 is 0
        basis = PowerBasis(0)
        coeffs = np.array([2.0])
        new_basis, new_coeffs = differentiate(basis, coeffs)
        assert new_basis.degree == 0
        np.testing.assert_allclose(new_coeffs, [0.0])

    def test_linear(self):
        # derivative of 2x + 2 is 2
        basis = PowerBasis(1)
        coeffs = np.array([2.0, 2.0])
        new_basis, new_coeffs = differentiate(basis, coeffs)
        assert new_basis.degree == 0
        np.testing.assert_allclose(new_coeffs, [2.0])

    def test_cubic(self):
        # derivative of x^3 + 4x^2 + 5x + 12 is 3x^2 + 8x + 5
        basis = PowerBasis(3)
        coeffs = np.array([12.0, 5.0, 4.0, 1.0])
        new_basis, new_coeffs = differentiate(basis, coeffs)
        assert new_basis.degree == 2
        np.testing.assert_allclose(new_coeffs, [5.0, 8.0, 3.0])

    def test_affine_derivative(self):
        # f(x) = x^2 on [0, 4], derivative is 2x
        # fit on physical points so coefficients are correct
        inner = PowerBasis(2)
        basis = AffinePolynomialBasis(inner, a=0.0, b=4.0)
        x_nodes = np.array([0.0, 2.0, 4.0])
        y_nodes = x_nodes**2
        coeffs = basis.fit(x_nodes, y_nodes)
        new_basis, new_coeffs = differentiate(basis, coeffs)
        x_test = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        expected = 2.0 * x_test
        result = new_basis.evaluate(new_coeffs, x_test)
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_evaluate_derivative(self):
        # derivative of x^3 is 3x^2
        basis = PowerBasis(3)
        coeffs = np.array([0.0, 0.0, 0.0, 1.0])
        x = np.array([0.0, 1.0, 2.0, -1.0])
        expected = 3.0 * x**2
        result = evaluate_derivative(basis, coeffs, x)
        np.testing.assert_allclose(result, expected, atol=1e-5)

    def test_evaluate_affine_derivative(self):
        # same test as analytic derivative test for affine but using evaluate_derivative instead of differentiate
        inner = PowerBasis(2)
        basis = AffinePolynomialBasis(inner, a=0.0, b=4.0)
        x_nodes = np.array([0.0, 2.0, 4.0])
        y_nodes = x_nodes**2
        coeffs = basis.fit(x_nodes, y_nodes)
        x_test = np.array([1.0, 2.0, 3.0])
        expected = 2.0 * x_test
        result = evaluate_derivative(basis, coeffs, x_test)
        np.testing.assert_allclose(result, expected, atol=1e-5)