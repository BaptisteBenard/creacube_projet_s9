import numpy as np

#####################################################################
################ Functions to help to create test data ##############

def positions_noiser(positions):
    """
    Add noise to positions for tests
    :positions: (nb_samples, 4, 3) numpy array that contains positions of the 4 cubes (x, y, z) coordinates for each sample
    """
    for i in range(len(positions)):
        for j in range(len(positions[0])):
            positions[i][j] = positions[i][j] + np.random.normal(0, 5, 3)
    return positions