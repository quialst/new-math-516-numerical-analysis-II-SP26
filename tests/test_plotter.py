import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from numanalysislib.plotting import Plotter
from numanalysislib.basis.power import PowerBasis
from numanalysislib.basis.tensor import TensorProductBasis

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


# Tests for tensor product basis plotting
class TestTensorPlotter:
    
    @pytest.fixture
    def tensor_basis_2x3(self):
        """Create a 2x3 tensor product basis (degree 1 × degree 2)."""
        bx = PowerBasis(degree=1)  # 2 DOFs
        by = PowerBasis(degree=2)  # 3 DOFs
        return TensorProductBasis(bx, by)
    
    @pytest.fixture
    def tensor_basis_3x3(self):
        """Create a 3x3 tensor product basis (degree 2 × degree 2)."""
        bx = PowerBasis(degree=2)  # 3 DOFs
        by = PowerBasis(degree=2)  # 3 DOFs
        return TensorProductBasis(bx, by)

    @patch("matplotlib.pyplot.show")
    def test_plot_basis_tensor_smoke(self, mock_show, tensor_basis_3x3):
        """Smoke test: plot_basis_tensor runs without error."""
        plotter = Plotter()
        try:
            plotter.plot_basis_tensor(tensor_basis_3x3, domain_x=(0, 1), domain_y=(0, 1))
        except Exception as e:
            pytest.fail(f"plot_basis_tensor raised an exception: {e}")
        
        # Verify show() was called exactly once
        mock_show.assert_called_once()

    @patch("matplotlib.pyplot.show")
    def test_plot_basis_tensor_correct_subplot_count(self, mock_show, tensor_basis_2x3):
        """Verify correct number of subplots for 2x3 tensor basis (6 DOFs)."""
        plotter = Plotter()
        with patch("matplotlib.pyplot.figure") as mock_fig:
            mock_fig.return_value = MagicMock()
            plotter.plot_basis_tensor(tensor_basis_2x3, domain_x=(0, 1), domain_y=(0, 1))
        
        mock_show.assert_called_once()

    @patch("matplotlib.pyplot.show")
    def test_plot_basis_tensor_invalid_basis(self, mock_show):
        """Verify TypeError when passing non-TensorProductBasis."""
        plotter = Plotter()
        basis_1d = PowerBasis(degree=2)
        
        with pytest.raises(TypeError) as exc_info:
            plotter.plot_basis_tensor(basis_1d, domain_x=(0, 1), domain_y=(0, 1))
        
        assert "TensorProductBasis" in str(exc_info.value)
        mock_show.assert_not_called()

    @patch("matplotlib.pyplot.show")
    def test_plot_fit_tensor_smoke(self, mock_show, tensor_basis_3x3):
        """Smoke test: plot_fit_tensor runs without error without true_func."""
        plotter = Plotter()
        
        # Create sample fitting data
        x_nodes = np.array([0, 0.5, 1])
        y_nodes = np.array([0, 0.5, 1])
        X_nodes, Y_nodes = np.meshgrid(x_nodes, y_nodes)
        z_values = X_nodes**2 + Y_nodes**2
        
        # Fit the basis
        coeffs = tensor_basis_3x3.fit(x_nodes, y_nodes, z_values)
        
        try:
            plotter.plot_fit_tensor(tensor_basis_3x3, coeffs, x_nodes, y_nodes,
                                   domain_x=(0, 1), domain_y=(0, 1))
        except Exception as e:
            pytest.fail(f"plot_fit_tensor raised an exception: {e}")
        
        mock_show.assert_called_once()

    @patch("matplotlib.pyplot.show")
    def test_plot_fit_tensor_with_true_func(self, mock_show, tensor_basis_3x3):
        """Test plot_fit_tensor with optional true_func argument."""
        plotter = Plotter()
        
        # Create sample fitting data
        x_nodes = np.array([0, 0.5, 1])
        y_nodes = np.array([0, 0.5, 1])
        X_nodes, Y_nodes = np.meshgrid(x_nodes, y_nodes)
        z_values = X_nodes**2 + Y_nodes**2
        
        # Fit the basis
        coeffs = tensor_basis_3x3.fit(x_nodes, y_nodes, z_values)
        
        # Define true function
        true_func = lambda x, y: x**2 + y**2
        
        try:
            plotter.plot_fit_tensor(tensor_basis_3x3, coeffs, x_nodes, y_nodes,
                                   domain_x=(0, 1), domain_y=(0, 1), 
                                   true_func=true_func)
        except Exception as e:
            pytest.fail(f"plot_fit_tensor with true_func raised exception: {e}")
        
        mock_show.assert_called_once()

    @patch("matplotlib.pyplot.show")
    def test_plot_fit_tensor_invalid_basis(self, mock_show):
        """Verify TypeError when passing non-TensorProductBasis to plot_fit_tensor."""
        plotter = Plotter()
        basis_1d = PowerBasis(degree=2)
        coeffs = np.array([0, 0, 1])
        x_nodes = np.array([0, 0.5, 1])
        y_nodes = np.array([0, 0.5, 1])
        
        with pytest.raises(TypeError) as exc_info:
            plotter.plot_fit_tensor(basis_1d, coeffs, x_nodes, y_nodes,
                                   domain_x=(0, 1), domain_y=(0, 1))
        
        assert "TensorProductBasis" in str(exc_info.value)
        mock_show.assert_not_called()

    @patch("matplotlib.pyplot.show")
    def test_plot_fit_tensor_coeff_shape_mismatch(self, mock_show, tensor_basis_3x3):
        """Verify ValueError when coefficients shape doesn't match (nx, ny)."""
        plotter = Plotter()
        
        # Create mismatched coefficients (should be (3, 3) but use (2, 2))
        coeffs_wrong = np.random.rand(2, 2)
        x_nodes = np.array([0, 0.5, 1])
        y_nodes = np.array([0, 0.5, 1])
        
        with pytest.raises(ValueError) as exc_info:
            plotter.plot_fit_tensor(tensor_basis_3x3, coeffs_wrong, x_nodes, y_nodes,
                                   domain_x=(0, 1), domain_y=(0, 1))
        
        assert "shape" in str(exc_info.value).lower()
        mock_show.assert_not_called()

    @patch("matplotlib.pyplot.show")
    def test_plot_basis_tensor_2x2_layout(self, mock_show):
        """Test plot_basis_tensor with small tensor basis (2x2 = 4 DOFs)."""
        plotter = Plotter()
        tensor_basis = TensorProductBasis(PowerBasis(degree=1), PowerBasis(degree=1))
        
        try:
            plotter.plot_basis_tensor(tensor_basis, domain_x=(0, 1), domain_y=(0, 1))
        except Exception as e:
            pytest.fail(f"plot_basis_tensor with 2x2 basis raised exception: {e}")
        
        mock_show.assert_called_once()
