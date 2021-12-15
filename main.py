import numpy as np


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

    return is_in_hand

def get_connection_mat(position, is_connected):
    # Initialize without any connection
    connection_mat = np.full((4, 4), False)

    print(connection_mat)

    return connection_mat