from numanalysislib.basis._abstract import PolynomialBasis
from numanalysislib.basis.affine import AffinePolynomialBasis
from numanalysislib.basis.power import PowerBasis #DELETE THIS
import numpy as np


class PiecewisePolynomial:
    def __init__(self, basis_type: PolynomialBasis, mesh: list):
        self.basis_type = basis_type

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

    
    def fit(self, y_mesh: list):
        # y_mesh is list of arrays with enough pts per element for required degree

        bases_coeffs = {}
        for index, y_element in enumerate(y_mesh):
            # extract info from element
            element = self.mesh[index]
            a = element[0]
            b = element[1]
            n_dofs = self.bases[element].n_dofs

            # create equidistant x_nodes to fit with
            x_nodes = []
            for i in range(n_dofs):
                x_nodes.append((1-i/n_dofs)*a + i/n_dofs*b)
            x_nodes = np.asarray(x_nodes)

            #fit 
            bases_coeffs[element] = self.bases[element].fit(x_nodes, y_element)

        return bases_coeffs

    def evaluate(self, coefficients: dict, x: np.array):
        # find element containing x
        for element in self.mesh:
            a = element[0]
            b = element[1]
            # evaluate element
            if a <= x and x <= b:
                return self.bases[element].evaluate(coefficients[element], x)
            

if __name__ == "__main__":
    power = PowerBasis(2)
    mesh = [(0.0, 0.5), (0.5, 1.0)]
    pw = PiecewisePolynomial(power, mesh)
    pw.fit([[0.1, 0.2, 0.1], [0.3, 0.1, 0.4]])
    print("worked")
