from s1.svd import *
import numpy as np

movieRatings = np.array([
        [2, 5, 3],
        [1, 2, 1],
        [4, 1, 1],
        [3, 5, 2],
        [5, 3, 1],
        [4, 5, 5],
        [2, 4, 2],
        [2, 2, 5],
    ], dtype='float64');

theSVD = svd(movieRatings);

npSVD = np.linalg.svd(movieRatings, full_matrices=False)

print(1)