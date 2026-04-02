from numanalysislib.basis._abstract import PolynomialBasis
import numpy as np


class PiecewisePolynomial:
    def __init__(self, basis_type: PolynomialBasis, mesh: list):
        self.basis_type = basis_type
        
        #check that mesh has no overlaps

        #order mesh


        self.mesh = mesh #ordered by left endpoints

        bases = {}
        for element in self.mesh:
            a = element[0]
            b = element[1]
            bases[element] = self.AffinePolynomialBasis(basis_type(), a, b)

    
    def fit(self, y_mesh):
        # y_mesh is list of lists with enough pts per element for required degree


    def evaluate(self, x):
        # find element containing x
        current_index = len(self.mesh)//2
        while True:
            element = self.mesh[current_index]
            a = element[0]
            b = element[1]
            if x < a:

        # return 

"""
Example usage:
mesh = [[0,1], [1, 2]]
basis_type = Lagrange



"""