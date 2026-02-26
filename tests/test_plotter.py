import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from numanalysislib.plotting import Plotter
from numanalysislib.basis.power import PowerBasis

class TestPlotter:
    
    @pytest.fixture
    def basis(self):
        return PowerBasis(degree=2)

    @patch("matplotlib.pyplot.show")
    def test_plot_basis_smoke(self, mock_show, basis):
        """
        Smoke Test: Does plot_basis run without error?
        We mock plt.show() so no window actually pops up during testing.
        """
        plotter = Plotter()
        try:
            plotter.plot_basis(basis)
        except Exception as e:
            pytest.fail(f"plot_basis raised an exception: {e}")
            
        # Verify show() was called exactly once
        mock_show.assert_called_once()

    @patch("matplotlib.pyplot.show")
    def test_plot_fit_smoke(self, mock_show, basis):
        """Smoke Test: Does plot_fit run without error?"""
        plotter = Plotter()
        
        # Dummy data
        x = np.array([0, 0.5, 1])
        y = x**2
        coeffs = np.array([0, 0, 1])
        
        try:
            plotter.plot_fit(basis, coeffs, x, y, domain=(0, 1))
        except Exception as e:
            pytest.fail(f"plot_fit raised an exception: {e}")
            
        mock_show.assert_called_once()

    @patch("matplotlib.pyplot.show")
    def test_plot_fit_with_true_func(self, mock_show, basis):
        """Test the optional true_function argument."""
        plotter = Plotter()
        x = np.array([0, 1, 2])
        y = np.array([0, 1, 4])
        coeffs = np.array([0, 0, 1])
        
        # Lambda function for true values
        true_func = lambda x: x**2
        
        try:
            plotter.plot_fit(basis, coeffs, x, y, domain=(0, 2), true_func=true_func)
        except Exception as e:
            pytest.fail(f"plot_fit with true_func raised exception: {e}")
            
        mock_show.assert_called_once()
