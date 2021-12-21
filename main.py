import numpy as np
import math

CUBE_DIM = 400
DIST_MARGIN = 20

def in_hand(positions, is_connected):
    # Check if length are similar and if data had the right dimensions
    pos_shape = positions.shape
    conn_shape = is_connected.shape
    if pos_shape[0] != conn_shape[0] or \
       pos_shape[1] != 4 or pos_shape[2] != 3 or \
       conn_shape[1] != 4 or conn_shape[2] != 6:
        return -1

    
    
    is_in_hand = []

    for i in range(len(positions)):
        # Initialize state
        if i != 0:
            state = is_in_hand[-1]
        else :
            state = [False, False, False, False]
        
        # Connection matrix
        connections_mat = get_connection_mat(positions[i], is_connected[i])

        # Apply rules

        # Propagate decision

        # Add current state
        is_in_hand.append(state)

    is_in_hand = np.array([[False, False, False, False] for _ in range(pos_shape[0])])
    return is_in_hand

def get_connection_mat(position, is_connected):
    # Initialize without any connection
    connection_mat = np.full((4, 4), False)

    if (is_connected == np.full((4, 6), False)).all():
        return connection_mat

    connected_cubes = []
    for cube in range(4):
        if np.sum(is_connected[cube]) != 0:
            connected_cubes.append(cube)

    for cube in connected_cubes:
        # check is face are connected for each cube
        for other_cube in connected_cubes:
            if cube != other_cube:
                if compute_distance(position[cube], position[other_cube]) < CUBE_DIM + DIST_MARGIN:
                    connection_mat[cube][other_cube] = True
                    connection_mat[other_cube][cube] = True

    return connection_mat

def compute_distance(pos1, pos2):
    dist = np.linalg.norm(pos1-pos2)
    return dist

