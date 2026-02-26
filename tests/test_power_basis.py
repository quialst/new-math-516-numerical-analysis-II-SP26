import numpy as np
import pytest
from numanalysislib.basis.power import PowerBasis

class TestPowerBasis:
    
    def test_initialization(self):
        """Verify degree and DOF setup."""
        basis = PowerBasis(degree=3)
        assert basis.degree == 3
        assert basis.n_dofs == 4

    def test_evaluate_basis_scalar(self):
        """Check individual monomial evaluation."""
        basis = PowerBasis(degree=2)
        x = np.array([2.0])
        
        # Index 0 -> x^0 = 1
        val0 = basis.evaluate_basis(0, x)
        assert val0[0] == 1.0
        
        # Index 2 -> x^2 = 4
        val2 = basis.evaluate_basis(2, x)
        assert val2[0] == 4.0

    def test_evaluate_basis_vector(self):
        """Check vectorized evaluation."""
        basis = PowerBasis(degree=2)
        x = np.array([0.0, 1.0, 2.0, 3.0])
        
        # x^2 should be [0, 1, 4, 9]
        y = basis.evaluate_basis(2, x)
        expected = np.array([0.0, 1.0, 4.0, 9.0])
        
        np.testing.assert_array_equal(y, expected)

    def test_fit_exact_reconstruction(self):
        """
        Check that we can recover a known polynomial
        Target: y = 1 + 2x + 3x^2
        Coefficients should be [1, 2, 3]
        """
        degree = 2
        basis = PowerBasis(degree)
        
        # We need 3 points to determine a parabola uniquely
        x_nodes = np.array([-1.0, 0.0, 1.0])
        # Calculate y based on target equation
        y_nodes = 1 + 2*x_nodes + 3*(x_nodes**2) 
        
        coeffs = basis.fit(x_nodes, y_nodes)
        
        # Check coefficients [1, 2, 3]
        expected_coeffs = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(coeffs, expected_coeffs, atol=1e-12)

    def test_fit_identity(self):
        """Test fitting a simple line y=x."""
        basis = PowerBasis(degree=1)
        x = np.array([0.0, 10.0])
        y = x
        
        coeffs = basis.fit(x, y)
        # Expected: 0 + 1*x -> [0, 1]
        np.testing.assert_allclose(coeffs, [0.0, 1.0], atol=1e-12)

    def test_evaluate_full_polynomial(self):
        """Test the evaluate() method using Horner's scheme."""
        basis = PowerBasis(degree=2)
        # P(x) = 2 + 0x + 1x^2  (i.e., x^2 + 2)
        coeffs = np.array([2.0, 0.0, 1.0])
        
        x = np.array([2.0])
        result = basis.evaluate(coeffs, x)
        
        # 2^2 + 2 = 6
        assert result[0] == 6.0

    def test_vandermonde_singularity(self):
        """Ensure specific error is raised for duplicate nodes."""
        basis = PowerBasis(degree=1)
        # Duplicate nodes -> Singular matrix
        x = np.array([1.0, 1.0])
        y = np.array([1.0, 2.0])
        
        with pytest.raises(ValueError, match="Singular matrix"):
            basis.fit(x, y)
            
    def test_invalid_basis_index(self):
        basis = PowerBasis(degree=2)
        x = np.array([1.0])
        with pytest.raises(ValueError):
            # Index > degree
            basis.evaluate_basis(3, x)

