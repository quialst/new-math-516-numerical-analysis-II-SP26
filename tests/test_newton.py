import pytest
import numpy as np
from numanalysislib.basis.newton import NewtonPolynomialBasis



# 1. Test basis functions

def test_newton_basis_functions():
    
    """
    Check that n_i(x) = Π_{j=0}^{i-1} (x - x_j)
    """
    
    x_nodes = np.array([0.0, 1.0, 2.0])
    basis = NewtonPolynomialBasis(x_nodes)

    x = np.array([3.0])

    # n0(x) = 1
    assert np.allclose(basis.evaluate_basis(0, x), 1.0)

    # n1(x) = (x - x0) = (3 - 0) = 3
    assert np.allclose(basis.evaluate_basis(1, x), 3.0)

    # n2(x) = (x - x0)(x - x1) = 3 * 2 = 6
    assert np.allclose(basis.evaluate_basis(2, x), 6.0)



# 2. Test divided differences (fit)

def test_newton_fit_simple_polynomial():
    
    """
    Fit p(x) = x^2 and verify coefficients are correct.
    """
    
    x_nodes = np.array([0.0, 1.0, 2.0])
    y_nodes = x_nodes**2  # [0, 1, 4]

    basis = NewtonPolynomialBasis(x_nodes)
    coeffs = basis.fit(x_nodes, y_nodes)

    """
    Expected Newton coefficients:
    c0 = 0
    c1 = 1
    c2 = 1
    
    """
    expected = np.array([0.0, 1.0, 1.0])

    np.testing.assert_allclose(coeffs, expected, rtol=1e-14)



# 3. Test interpolation property

def test_newton_interpolation_property():
    
    """
    The fitted polynomial must satisfy p(x_i) = y_i
    """
    
    x_nodes = np.array([0.0, 1.0, 2.0, 3.0])
    y_nodes = np.sin(x_nodes)

    basis = NewtonPolynomialBasis(x_nodes)
    coeffs = basis.fit(x_nodes, y_nodes)

    y_eval = basis.evaluate(coeffs, x_nodes)

    np.testing.assert_allclose(y_eval, y_nodes, rtol=1e-12)



# 4. Test Horner evaluation (vector input)

def test_newton_evaluate_vector():
    
    """
    Check evaluation on multiple points.
    """
    
    x_nodes = np.array([0.0, 1.0, 2.0])
    y_nodes = x_nodes**2

    basis = NewtonPolynomialBasis(x_nodes)
    coeffs = basis.fit(x_nodes, y_nodes)

    x_test = np.array([0.0, 0.5, 1.5, 3.0])
    y_expected = x_test**2

    y_eval = basis.evaluate(coeffs, x_test)

    np.testing.assert_allclose(y_eval, y_expected, rtol=1e-12)



# 5. Test coefficient size mismatch

def test_newton_coefficient_shape_mismatch():
    basis = NewtonPolynomialBasis(np.array([0.0, 1.0, 2.0]))

    wrong_coeffs = np.array([1.0, 2.0])  # should be 3
    x = np.array([0.0])

    with pytest.raises(ValueError):
        basis.evaluate(wrong_coeffs, x)



# 6. Test node size mismatch in fit

def test_newton_fit_node_mismatch():
    basis = NewtonPolynomialBasis(np.array([0.0, 1.0, 2.0]))

    x_nodes = np.array([0.0, 1.0])  # wrong size
    y_nodes = np.array([0.0, 1.0])

    with pytest.raises(ValueError):
        basis.fit(x_nodes, y_nodes)



# 7. Test exact match


def test_newton_matches_polynomial_exactly():
    """
    Higher-degree exact polynomial recovery.
    """
    x_nodes = np.linspace(-1, 1, 5)
    y_nodes = x_nodes**3 - 2*x_nodes + 1

    basis = NewtonPolynomialBasis(x_nodes)
    coeffs = basis.fit(x_nodes, y_nodes)

    x_test = np.linspace(-1, 1, 20)
    y_expected = x_test**3 - 2*x_test + 1

    y_eval = basis.evaluate(coeffs, x_test)

    np.testing.assert_allclose(y_eval, y_expected, rtol=1e-12)
