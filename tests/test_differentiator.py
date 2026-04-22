import numpy as np
from numanalysislib.basis.power import PowerBasis
from numanalysislib.basis.affine import AffinePolynomialBasis
from numanalysislib.calculus.differentiator import differentiate, evaluate_derivative
from numanalysislib.basis.chebyshev import ChebyshevBasis


class TestDifferentiate:

    def test_constant(self):
        """Test if derivative of a constant is zero."""
        basis = PowerBasis(0)
        coeffs = np.array([2.0])
        new_basis, new_coeffs = differentiate(basis, coeffs)
        assert new_basis.degree == 0
        np.testing.assert_allclose(new_coeffs, [0.0])

    def test_linear(self):
        """Test if derivative of a linear polynomial is correct."""
        basis = PowerBasis(1)
        coeffs = np.array([2.0, 2.0])
        new_basis, new_coeffs = differentiate(basis, coeffs)
        assert new_basis.degree == 0
        np.testing.assert_allclose(new_coeffs, [2.0])

    def test_cubic(self):
        """Test if derivative of a cubic polynomial is correct."""
        # derivative of x^3 + 4x^2 + 5x + 12 is 3x^2 + 8x + 5
        basis = PowerBasis(3)
        coeffs = np.array([12.0, 5.0, 4.0, 1.0])
        new_basis, new_coeffs = differentiate(basis, coeffs)
        assert new_basis.degree == 2
        np.testing.assert_allclose(new_coeffs, [5.0, 8.0, 3.0])

    def test_affine_derivative(self):
        """Test if derivative of an AffinePolynomialBasis polynomial is correct."""
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
        """Test if evaluate_derivative gives correct numerical approximation."""
        basis = PowerBasis(3)
        coeffs = np.array([0.0, 0.0, 0.0, 1.0])
        x = np.array([0.0, 1.0, 2.0, -1.0])
        expected = 3.0 * x**2
        result = evaluate_derivative(basis, coeffs, x)
        np.testing.assert_allclose(result, expected, atol=1e-5)

    def test_evaluate_affine_derivative(self):
        """Test if evaluate_derivative gives correct numerical approximation for an AffinePolynomialBasis polynomial."""
        inner = PowerBasis(2)
        basis = AffinePolynomialBasis(inner, a=0.0, b=4.0)
        x_nodes = np.array([0.0, 2.0, 4.0])
        y_nodes = x_nodes**2
        coeffs = basis.fit(x_nodes, y_nodes)
        x_test = np.array([1.0, 2.0, 3.0])
        expected = 2.0 * x_test
        result = evaluate_derivative(basis, coeffs, x_test)
        np.testing.assert_allclose(result, expected, atol=1e-5)

    def test_second_derivative(self):
        """Test if second derivative of x^4 is 12x^2."""
        basis = PowerBasis(4)
        coeffs = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
        new_basis, new_coeffs = differentiate(basis, coeffs, k=2)
        assert new_basis.degree == 2
        x = np.array([0.0, 1.0, 2.0, -1.0])
        expected = 12.0 * x**2
        result = new_basis.evaluate(new_coeffs, x)
        np.testing.assert_allclose(result, expected)

    def test_evaluate_second_derivative(self):
        """Test numerical second derivative of x^4 is 12x^2."""
        basis = PowerBasis(4)
        coeffs = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
        x = np.array([0.0, 1.0, 2.0, -1.0])
        expected = 12.0 * x**2
        result = evaluate_derivative(basis, coeffs, x, k=2)
        np.testing.assert_allclose(result, expected, atol=1e-3)

    def test_forward_scheme(self):
        """Test forward difference scheme on x^3."""
        basis = PowerBasis(3)
        coeffs = np.array([0.0, 0.0, 0.0, 1.0])
        x = np.array([0.0, 1.0, 2.0])
        expected = 3.0 * x**2
        result = evaluate_derivative(basis, coeffs, x, scheme="forward")
        np.testing.assert_allclose(result, expected, atol=1e-3)

    def test_backward_scheme(self):
        """Test backward difference scheme on x^3."""
        basis = PowerBasis(3)
        coeffs = np.array([0.0, 0.0, 0.0, 1.0])
        x = np.array([0.0, 1.0, 2.0])
        expected = 3.0 * x**2
        result = evaluate_derivative(basis, coeffs, x, scheme="backward")
        np.testing.assert_allclose(result, expected, atol=1e-3)

    def test_chebyshev_derivative(self):
        """Test differentiation of a Chebyshev polynomial."""
        # Represent f(x) = x^2 as a Chebyshev polynomial, derivative is 2x
        cheb_basis = ChebyshevBasis(2)
        x_nodes = cheb_basis.chebyshev_nodes(3, kind="roots")
        y_nodes = x_nodes**2
        coeffs = cheb_basis.fit(x_nodes, y_nodes)
        new_basis, new_coeffs = differentiate(cheb_basis, coeffs)
        x_test = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        expected = 2.0 * x_test
        result = new_basis.evaluate(new_coeffs, x_test)
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_affine_chebyshev_derivative(self):
        """Test differentiation of a Chebyshev polynomial on [0, 10]."""
        cheb = ChebyshevBasis(2)
        basis = AffinePolynomialBasis(cheb, a=0.0, b=10.0)
        x_nodes = np.array([0.0, 5.0, 10.0])
        y_nodes = x_nodes**2
        coeffs = basis.fit(x_nodes, y_nodes)
        new_basis, new_coeffs = differentiate(basis, coeffs)
        x_test = np.array([0.0, 2.5, 5.0, 7.5, 10.0])
        expected = 2.0 * x_test
        result = new_basis.evaluate(new_coeffs, x_test)
        np.testing.assert_allclose(result, expected, atol=1e-10)

    # Rishi Note: Test with non-PowerBasis (e.g. Lagrange, Newton) once those modules are merged
