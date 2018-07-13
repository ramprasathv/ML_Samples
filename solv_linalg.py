import numpy as np
#from numpy import linalg


def solve(a,b):
    """
    Solves linear equations a x  = b, and returns the value as array
    :param a: Array a of dim m x n
    :param b: Array b of dim n x p
    :return: Returns solution for x in equation a x = b
    """

    return np.linalg.solve(a,b)

"""
Solve the equations: 3x0 + x1  =0 and x0 + 2x1 = 8
"""
a = np.array([[3,1],[1,2]])
b = np.array([9,8])

x = solve(a,b)
print("Result x0={}, x1={}".format(x[0], x[1]))
print("Check result: ", np.allclose(np.dot(a,x), b))
