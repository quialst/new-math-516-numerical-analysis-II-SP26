from numanalysislib.basis._abstract import PolynomialBasis
from numanalysislib.basis.affine import AffinePolynomialBasis
import numpy as np
import warnings


class PiecewisePolynomial(PolynomialBasis):
    def __init__(self, basis_type: PolynomialBasis, mesh: list):
        """
        Initalize PiecewisePolynomial

        Args:
            - basis_type: Polynomial basis to be used for each element
            - mesh: list of tuples specifying element endpoints
        """
        self.basis_type = basis_type

        # inheret attributes
        self.basis = basis_type
        self.a_hat = basis_type.a
        self.b_hat = basis_type.b
        self.degree = basis_type.degree
        self.n_dofs = basis_type.n_dofs

        #order mesh
        mesh.sort()
        self.mesh = mesh
        
        #check that mesh has no overlaps
        for index in range(len(self.mesh)-1):
            element1 = self.mesh[index]
            element2 = self.mesh[index + 1]

            if element1[1] != element2[0]:
                raise ValueError("mesh must be have connected elements")

        self.bases = {}
        for element in self.mesh:
            a = element[0]
            b = element[1]
            self.bases[element] = AffinePolynomialBasis(basis_type, a, b)

    #overwrites output type
    def fit(self, y_mesh: list) -> dict:
        """
        Override of fit method with dictionary.

        Args:
            - y_mesh: list of arrays. Each array should have enough points for the degrees of freedom of the basis_type
        
        Returns:
            - bases_coeffs: dictionary with kw for each mesh element and corresponding value of the coefficients for that element
        """
        bases_coeffs = {}

        last_right_endpoint = y_mesh[0][0] #initializer tracker for last encountered right endpoint

        for index, y_element in enumerate(y_mesh):
            # check dofs
            if len(y_element) != self.n_dofs:
                raise ValueError("y_mesh must have as many points as dofs for basis_type")
            
            # check continuity
            if y_element[0] != last_right_endpoint:
                raise ValueError(f"y_mesh must induce continuous piecewise polynomial. Ensure endpoints are consistent")
            last_right_endpoint = y_element[-1]

            # extract info from element
            element = self.mesh[index]
            a = element[0]
            b = element[1]
            n_dofs = self.bases[element].n_dofs

            # create equidistant x_nodes to fit with
            x_nodes = []
            for i in range(n_dofs):
                x_nodes.append((1-i/(n_dofs-1))*a + i/(n_dofs-1)*b)
            x_nodes = np.asarray(x_nodes)
            #fit 
            bases_coeffs[element] = self.bases[element].fit(x_nodes, y_element)
        

        return bases_coeffs
    
    def float_evaluate_basis(self, index: int, x: float) -> float:
        """
        single float x implementation for basis evaluation

        Args:
            - index: index of basis element to evaluate
            - x: point to evaluate at

        Returns:
            - y: value of basis element at x
        """
        min_index = 0
        max_index = len(self.mesh)
        current_index = min_index + (max_index - min_index)//2
        for _ in range(len(self.mesh)):
            element = self.mesh[current_index]
            a = element[0]
            b = element[1]
            if x < a:
                max_index = current_index
            elif x > b:
                min_index = current_index
            else:
                return self.bases[element].evaluate_basis(index, x) 
            
            current_index = min_index + (max_index - min_index)//2
        # throw a warning if the loop finishes
        warnings.warn("no interval found")
    
    def evaluate_basis(self, index: int, x: np.array) -> np.array:
        """
        vectorized x implementation for basis evaluation

        Args:
            - index: index of basis element to evaluate
            - x: points to evaluate at

        Returns:
            - y: values of basis element at x
        """
        return np.vectorize(lambda y: self.float_evaluate_basis(index, y))(x)

    def float_evaluate(self, coefficients: dict, x: float) -> float:
        """
        single float x implementation for evaluation

        Args:
            - coeffs: coefficients for basis functions
            - x: point to evaluate at

        Returns:
            - y: value of basis at x
        """

        min_index = 0
        max_index = len(self.mesh)
        current_index = min_index + (max_index - min_index)//2
        for _ in range(len(self.mesh)):
            element = self.mesh[current_index]
            a = element[0]
            b = element[1]
            if x < a:
                max_index = current_index
            elif x > b:
                min_index = current_index
            else:
                return self.bases[element].evaluate(coefficients[element], x) 
            
            current_index = min_index + (max_index - min_index)//2
        # throw a warning if the loop finishes
        warnings.warn("no interval found")
    
    #overwrites output type
    def evaluate(self, coefficients: dict, x: np.array):
        """
        vectorized x implementation for evaluation

        Args:
            - coeffs: coefficients for basis functions
            - x: points to evaluate at

        Returns:
            - y: values of basis at x
        """
        return np.vectorize(lambda y: self.float_evaluate(coefficients, y))(x)
