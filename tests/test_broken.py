import pytest
import numpy as np
from numanalysislib.basis.broken import BrokenPolynomialBasis
from numanalysislib.basis.power import PowerBasis


def test_broken_basis_local_support(): 
    """
    Each basis function should be nonzero only on its element.
    """
    basis_type = PowerBasis(degree=1)
    mesh = np.array([0.0, 1.0, 2.0])
    basis = BrokenPolynomialBasis(basis_type, mesh)

    x = np.array([0.5, 1.5])

    # global index 0 → first element
    vals = basis.evaluate_basis(0, x)

    # should be nonzero on first element only
    assert vals[0] != 0
    assert vals[1] == 0


def test_broken_dof_count():
    """
    Check DOFs = N_elements * (k+1)
    """ 
    basis_type = PowerBasis(degree=2)  # 3 DOFs per element
    mesh = np.array([0.0, 1.0, 2.0, 3.0])  # 3 elements

    basis = BrokenPolynomialBasis(basis_type, mesh)

    expected_dofs = 3 * 3  # 3 elements * 3 DOFs
    assert basis.n_dofs == expected_dofs


def test_broken_l2_projection_constant():  
    """
    Project f(x)=1 → should recover constant exactly.
    """   
    basis_type = PowerBasis(degree=1)
    mesh = np.array([0.0, 1.0, 2.0])
    basis = BrokenPolynomialBasis(basis_type, mesh)

    f = lambda x: np.ones_like(x)

    coeffs = basis.fit(f)

    x_test = np.linspace(0, 2, 20)
    y_eval = basis.evaluate(coeffs, x_test)

    np.testing.assert_allclose(y_eval, 1.0, atol=1e-10)


def test_broken_l2_projection_linear():
    """
    Project f(x)=x → should match exactly for degree 1 basis.
    """
    basis_type = PowerBasis(degree=1)
    mesh = np.array([0.0, 1.0, 2.0])
    basis = BrokenPolynomialBasis(basis_type, mesh)

    f = lambda x: x

    coeffs = basis.fit(f)

    x_test = np.linspace(0, 2, 20)
    y_eval = basis.evaluate(coeffs, x_test)

    np.testing.assert_allclose(y_eval, x_test, atol=1e-10)


def test_broken_discontinuity(): 
    """
    DG allows discontinuities across element boundaries.
    """ 
    basis_type = PowerBasis(degree=1)
    mesh = np.array([0.0, 1.0, 2.0])
    basis = BrokenPolynomialBasis(basis_type, mesh)

    coeffs = {}

    # element 1 → 0, element 2 → 1
    coeffs[(0.0, 1.0)] = np.array([0.0, 0.0])
    coeffs[(1.0, 2.0)] = np.array([1.0, 0.0])

    x_left = 1.0 - 1e-6
    x_right = 1.0 + 1e-6

    y_left = basis.evaluate(coeffs, np.array([x_left]))[0]
    y_right = basis.evaluate(coeffs, np.array([x_right]))[0]

    assert y_left != y_right


def test_broken_coeff_structure():
    """
    fit() should return dict with coefficients per element.
    """
    basis_type = PowerBasis(degree=1)
    mesh = np.array([0.0, 1.0, 2.0])
    basis = BrokenPolynomialBasis(basis_type, mesh)

    f = lambda x: x**2

    coeffs = basis.fit(f)

    assert isinstance(coeffs, dict)
    assert len(coeffs) == len(basis.mesh)

    for element in basis.mesh:
        assert element in coeffs
        assert len(coeffs[element]) == basis_type.n_dofs


def test_broken_evaluate_vector():
    """
    Ensure vectorized evaluation works correctly.
    """ 
    basis_type = PowerBasis(degree=1)
    mesh = np.array([0.0, 1.0, 2.0])
    basis = BrokenPolynomialBasis(basis_type, mesh)

    f = lambda x: x
    coeffs = basis.fit(f)

    x_test = np.array([0.25, 0.75, 1.25, 1.75])
    y_expected = x_test

    y_eval = basis.evaluate(coeffs, x_test)

    np.testing.assert_allclose(y_eval, y_expected, atol=1e-10)


def test_broken_mesh_with_repeated_points():
    basis_type = PowerBasis(degree=1)

    mesh = np.array([0.0, 1.0, 1.0, 2.0])

    with pytest.raises(ValueError):
        BrokenPolynomialBasis(basis_type, mesh)

def test_broken_mesh_not_sorted():
    basis_type = PowerBasis(degree=1)

    mesh = np.array([0.0, 2.0, 1.0])

    with pytest.raises(ValueError):
        BrokenPolynomialBasis(basis_type, mesh)


def test_broken_mesh_too_short():
    basis_type = PowerBasis(degree=1)

    mesh = np.array([1.0])

    with pytest.raises(ValueError):
        BrokenPolynomialBasis(basis_type, mesh)


def test_broken_basis_negative_index():

    mesh = np.array([0.0, 1.0, 2.0])

    basis_type = PowerBasis(degree=2)
    basis = BrokenPolynomialBasis(basis_type, mesh)

    x = np.array([0.5, 1.5])

    with pytest.raises(ValueError):
        basis.evaluate_basis(-1, x)


def test_broken_basis_index_too_large():

    mesh = np.array([0.0, 1.0, 2.0])

    basis_type = PowerBasis(degree=2)
    basis = BrokenPolynomialBasis(basis_type, mesh)

    x = np.array([0.5, 1.5])

    with pytest.raises(ValueError):
        basis.evaluate_basis(basis.n_dofs, x)


