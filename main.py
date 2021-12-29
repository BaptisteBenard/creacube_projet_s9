import numpy as np
import math
import copy

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

    for time_step in range(len(positions)):
        state = []
        # Initialize state
        if time_step != 0:
            state = copy.deepcopy(is_in_hand[-1])
        else :
            state = [False, False, False, False]
        
        # Connection matrix
        connections_mat = get_connection_mat(positions[time_step], is_connected[time_step])

        # Apply rules
        # Rule 1: Cube is higher than 4 cubes
        for cube in range(4):
            if positions[time_step][cube][2] > 3.5 * CUBE_DIM + DIST_MARGIN:
                state[cube] = True

        # Rule 2: Cube is moving upward
        for cube in range(4):
            if time_step != 0 and positions[time_step][cube][2] - positions[time_step - 1][cube][2] > 2*DIST_MARGIN:
                state[cube] = True
        
        # Rule 3: Cube on mat and not moving
        if time_step > 0:
            for cube in range(4):
                if on_ground(positions[time_step][cube]) and not is_moving(positions[time_step-1][cube],positions[time_step][cube]):
                    state[cube] = False
        
        # Rule 4 : z constant and speed is constant 
        if time_step > 1:
            for cube in range(4):
                if z_constant(positions[time_step-2][cube],positions[time_step-1][cube],positions[time_step][cube]) and \
                   is_moving_constantly(positions[time_step-2][cube],positions[time_step-1][cube],positions[time_step][cube]):
                    state[cube] = False

        # Propagate decision
        state = propagate(state, connections_mat)

        # Add current state
        is_in_hand.append(state)

    is_in_hand = np.array(is_in_hand)
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

# true if the cube is moving between pos1 and pos2
# pos1 and pos2 are np.array : [x,y,z]
def is_moving(pos1,pos2):
    for i in range(3):
        if abs(pos1[i]-pos2[i]) > 2 * DIST_MARGIN:
            return True
    return False

def is_moving_constantly(pos1, pos2, pos3):
    # we first check that the cube is moving between pos1 and pos3
    if is_moving(pos1,pos3):
        for i in range(3):
            #if the difference between pos1 and pos2 according to the coordinate x is not the same compared to the difference between pos2 and pos3 according to the coordinate x, 
#the cube cannot have a constant speed. 
            if abs(abs(pos1[i]-pos2[i])-abs(pos2[i]-pos3[i])) > 2 * DIST_MARGIN:
                print(pos1, pos2, pos3, i)
                return False
        return True
    return False

def on_ground(pos):
    return pos[2] > CUBE_DIM/2 - DIST_MARGIN and pos[2] < CUBE_DIM/2 + DIST_MARGIN

def z_constant(pos1, pos2, pos3):
    return abs(pos1[2] - pos2[2]) < 2 * DIST_MARGIN and abs(pos1[2] - pos3[2]) < 2 * DIST_MARGIN and abs(pos2[2] - pos3[2]) < 2 * DIST_MARGIN

def propagate(states, connection_mat):
    state = copy.deepcopy(states)
    tmp = [False, False, False, False]
    while np.sum(state != tmp) > 0:
        tmp = copy.deepcopy(state)
        for cube1 in range(4):
            for cube2 in range(cube1, 3):
                if connection_mat[cube1][cube2]:
                    if not state[cube1] or not state[cube2]:
                        state[cube1], state[cube2] = False, False
    return state