import numpy as np

#####################################################################
################ Functions to help to create test data ##############

def positions_noiser(positions):
    for time_stamp in positions:
        for cube in time_stamp:
            cube = cube + np.random.normal(0, 5, 3)
    return positions