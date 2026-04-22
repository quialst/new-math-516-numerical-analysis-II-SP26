import numpy as np
import pytest

from numanalysislib.basis.affine import AffinePolynomialBasis
from numanalysislib.basis.hermite import HermiteBasis


class TestHermiteBasis:
    def test_evaluate_basis_endpoints(self):
        """
        Verify that the Hermite basis functions satisfy the expected endpoint conditions:
            h_0 = 2t^3 - 3t^2 + 1
            h_2 = -2t^3 + 3t^2
            h_1 = t^3 - 2t^2 + t
            h_3 = t^3 - t^2
        At t=0: h_0=1, h_1=0, h_2=0, h_3=0
        At t=1: h_0=0, h_1=0, h_2=1, h_3=0
        """

        basis = HermiteBasis()
        t = np.array([0.0, 1.0])

        h0 = basis.evaluate_basis(0, t)
        h1 = basis.evaluate_basis(1, t)
        h2 = basis.evaluate_basis(2, t)
        h3 = basis.evaluate_basis(3, t)

        np.testing.assert_allclose(h0, [1.0, 0.0]) 
        np.testing.assert_allclose(h1, [0.0, 0.0]) 
        np.testing.assert_allclose(h2, [0.0, 1.0]) 
        np.testing.assert_allclose(h3, [0.0, 0.0])

    def test_fit_scales_derivatives(self):
        """
        Verify the fit method correctly scales derivative coefficients based on the interval length.
        """
        basis = HermiteBasis()
        x_nodes = np.array([2.0, 5.0])  
        y_nodes = np.array([1.0, 2.0, 3.0, 4.0])  # v_a, d_a, v_b, d_b

        coeffs = basis.fit(x_nodes, y_nodes)
        # Expected coefficients: [v_a, (b - a) * d_a, v_b, (b - a) * d_b]
        expected = np.array([1.0, 6.0, 3.0, 12.0])  

        np.testing.assert_allclose(coeffs, expected)

    def test_evaluate_polynomial(self):
        """
        Verify the evaluate method.
        """
        basis = HermiteBasis()
        coeffs = np.array([1.0, 0.0, 2.0, 0.0])  # linear shape from 1 to 2

        t = np.array([0.0, 0.5, 1.0])
        values = basis.evaluate(coeffs, t)

        # Expected values: at t=0 -> 1.0, at t=0.5 -> 1.5, at t=1.0 -> 2.0
        np.testing.assert_allclose(values, [1.0, 1.5, 2.0])

    def test_invalid_basis_index(self):
        basis = HermiteBasis()
        with pytest.raises(ValueError):
            # index 4 is out of range
            basis.evaluate_basis(4, np.array([0.0]))

    def test_coefficient_length(self):
        basis = HermiteBasis()
        # example of wrong coefficient array: only 3 coefficients instead of 4
        bad_coeffs = np.array([1.0, 2.0, 3.0])  
        with pytest.raises(ValueError):
            basis.evaluate(bad_coeffs, np.array([0.0]))

    def test_affine_coefficients(self):
        """
        Hermite coefficients computed on a physical interval should evaluate
        correctly when wrapped by AffinePolynomialBasis.
        """
        basis = HermiteBasis()
        # Physical interval
        a = 2.0
        b = 5.0
        

        # Linear function on [2, 5]: f(x) = 1 + (x-2)/3
        v_a = 1.0
        v_b = 2.0
        d_a = d_b = 1.0 / 3.0  # physical derivatives

        affine = AffinePolynomialBasis(basis, a=a, b=b)
        coeffs_hat = affine.fit(np.array([a, b]), np.array([v_a, d_a, v_b, d_b]))

        np.testing.assert_allclose(
            coeffs_hat,
            np.array([1.0, 1.0, 2.0, 1.0]),
        )

    def test_affine_evaluate_points(self):
        """
        Test that the affine wrapper correctly evaluates the Hermite polynomial at points in the physical interval.
        """
        
        basis = HermiteBasis()
        # Physical interval
        a = 2.0
        b = 5.0
        # Linear function on [2, 5]: f(x) = 1 + (x-2)/3
        v_a = 1.0
        v_b = 2.0
        d_a = d_b = 1.0 / 3.0  # physical derivatives

        affine = AffinePolynomialBasis(basis, a=a, b=b)
        coeffs_hat = affine.fit(np.array([a, b]), np.array([v_a, d_a, v_b, d_b]))

        x_phys = np.array([a, 3.0, 4.0, b])
        expected = np.array([1.0, 1+1.0/3.0, 1+2.0/3.0, 2.0])
        values = affine.evaluate(coeffs_hat, x_phys)
        np.testing.assert_allclose(values, expected)