import numpy as np
from numpy.linalg import svd 

def dot_product(vector1, vector2):
    """ Implement dot product of the two vectors.
    Args:
        vector1: numpy array of shape (x, n)
        vector2: numpy array of shape (n, x)

    Returns:
        out: numpy array of shape (x,x) (scalar if x = 1)
    """
    if (vector1.ndim == 1):
        vector1 = vector1.reshape((len(vector1), 1))
    if (vector2.ndim == 1):
        vector2 = vector2.reshape((len(vector2), 1))
    if (vector1.shape[1] != vector2.shape[0]):
        print("You can't multiply these two vectors")
        return None
    out = np.zeros((vector1.shape[0], vector2.shape[1]))
    for i in np.arange(vector1.shape[0]):
        for j in np.arange(vector2.shape[1]):
            for z in np.arange(vector1.shape[1]):
                out[i][j] += vector1[i][z]*vector2[z][j]
    return out

def matrix_mult(M, vector1, vector2):
    """ Implement (vector1.T * vector2) * (M * vector1)
    Args:
        M: numpy matrix of shape (x, n)
        vector1: numpy array of shape (1, n)
        vector2: numpy array of shape (n, 1)

    Returns:
        out: numpy matrix of shape (1, x)
    """
    out = None
    aDotB = dot_product(vector1, vector2)
    MdotA = dot_product(vector1, M.T)
    return aDotB[0][0] * MdotA 

def get_singular_values(matrix, n):
    """Return top n singular values of matrix
    Args:
        matrix: numpy matrix of shape (m, w)
        n: number of singular values to output 
    
    Returns:
        singular_values: array of shape (n)
    """
    u, s, v = svd(matrix) 
    s.sort()
    return s[len(s)-n:]


def get_eigen_values_and_vectors(matrix, num_values):
    """ Return top n eigen values and corresponding vectors of matrix
    Args:
        matrix: numpy matrix of shape (m, m)
        num_values: number of eigen values and respective vectors to return
        
    Returns:
        eigen_values: array of shape (n)
        eigen_vectors: array of shape (m, n)
    """
    w, v = np.linalg.eig(matrix)
    sort_index = w.argsort()
    eigen_values = w[sort_index[len(sort_index)-num_values:]]
    eigen_vectors =v[sort_index[len(sort_index)-num_values:]]
    return eigen_values, eigen_vectors
