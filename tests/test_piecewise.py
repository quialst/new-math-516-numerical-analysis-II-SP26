import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from numanalysislib.basis._abstract import PolynomialBasis
from numanalysislib.basis.affine import AffinePolynomialBasis
from numanalysislib.basis.piecewise import PiecewisePolynomial
from numanalysislib.basis.power import PowerBasis
from numanalysislib.plotting import Plotter


class TestPiecewise():
    @pytest.fixture
    def simple_pw_poly(self):
        basis = PowerBasis(degree = 1)
        mesh = [0.0, 1.0]
        return PiecewisePolynomial(basis, mesh)

    @pytest.fixture
    def pw_poly(self):
        basis = PowerBasis(degree=1)
        h = 0.5
        mesh = [k*h for k in range(5)]
        return PiecewisePolynomial(basis, mesh)
    
    def test_evaluate_basis_exact(self, simple_pw_poly):
        "check basis evaluation for simple case"
        x_vals = np.array([0.0, 0.5, 1.0])
        constant_y_vals = simple_pw_poly.evaluate_basis(0, x_vals)
        linear_y_vals = simple_pw_poly.evaluate_basis(1, x_vals)
        np.testing.assert_allclose(constant_y_vals, np.array([1.0, 1.0, 1.0]), atol = 1e-12)
        np.testing.assert_allclose(linear_y_vals, np.array([0.0, 0.5, 1.0]), atol = 1e-12)

    
    def test_fit_evaluate_exact(self, pw_poly):
        "check the process of fit and evaluating"
        y_nodes = [np.array([0.0, 0.5]), np.array([0.5, 0.0]), np.array([0.0, 1.0]), np.array([1.0, 1.25])]
        h = 0.5
        x_vals = np.array([k*h for k in range(5)])
        y_vals = np.array([0.0, 0.5, 0.0, 1.0, 1.25])
        coeffs = pw_poly.fit(y_nodes)
        pred_y_vals = pw_poly.evaluate(coeffs, x_vals)
        np.testing.assert_allclose(pred_y_vals, y_vals)
    
    def test_fit_failure_continuity(self, pw_poly):
        "ensure error is thrown for discontinuous fit points"
        with pytest.raises(ValueError):
            pw_poly.fit([np.array([0.0, 0.5]), np.array([0.6, 0.0]), np.array([0.0, 1.0]), np.array([1.0, 1.25])])

    def test_fit_failure_dofs_low(self, pw_poly):
        "ensure error is thrown for not enough y_vals"
        with pytest.raises(ValueError):
            pw_poly.fit([np.array([0.0]), np.array([0.0, 1.0]), np.array([1.0, 1.0]), np.array([1.0, 1.25])])

    def test_fit_failure_dofs_high(self, pw_poly):
        "ensure error is thrown for too many y_vals"
        with pytest.raises(ValueError):
            pw_poly.fit([np.array([0.0, 1.0, 2.0]), np.array([2.0, 1.0]), np.array([1.0, 1.0]), np.array([1.0, 1.25])])

    @patch("matplotlib.pyplot.show")
    def test_plotter_plot_basis(self, mock_show, pw_poly):
        "check for plotting functionality on the plot_basis function"
        plotter = Plotter()
        plotter.plot_basis(pw_poly, (0, 2))
    
    @patch("matplotlib.pyplot.show")
    def test_plotter_plot_fit(self, mock_show, pw_poly):
        "check for plotting functionality on the plot_fit function"
        plotter = Plotter()
        coeffs = pw_poly.fit([np.array([0.0, 0.5]), np.array([0.5, 0.0]), np.array([0.0, 1.0]), np.array([1.0, 1.25])])
        x_nodes = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        y_nodes = [0.0, 0.5, 0.0, 1.0, 1.25]
        plotter.plot_fit(pw_poly, coeffs, x_nodes, y_nodes, (0.0, 2.0))
