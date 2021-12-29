import numpy as np

#####################################################################
################ Functions to help to create test data ##############

def positions_noiser(positions):
    for i in range(len(positions)):
        for j in range(len(positions[0])):
            positions[i][j] = positions[i][j] + np.random.normal(0, 5, 3)
    return positions