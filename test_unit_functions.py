import numpy as np
from main import get_connection_mat, compute_distance, is_moving, is_moving_constantly,\
                 on_ground, z_constant, propagate
from utils import positions_noiser

#####################################################################
################ Tests on connection matrix #########################

def test_conn_mat_no_connections():
    """ Check that no connection are found if the are no connected face"""
    positions = np.array([[[700, 700, 200],[700,-700,800],[-700,700,200],[-700,-700,200]]])
    positions = positions_noiser(positions)
    positions = positions[0]
    connections = np.array([[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]])
    
    result = get_connection_mat(positions, connections)
    assert (result == np.full((4, 4), False)).all()
    

def test_conn_mat_one_connection_1():
    """ Check that one connection is found"""
    positions = np.array([[[700, 700, 200],[700,-700,800],[1100,700,200],[-700,-700,200]]])
    positions = positions_noiser(positions)
    positions = positions[0]
    connections = np.array([[False, True, False, False, False, False], [False, False, False, False, False, False], [True, False, False, False, False, False], [False, False, False, False, False, False]])
    
    result = get_connection_mat(positions, connections)
    expected = np.array([[False, False, True, False],
                        [False, False, False, False],
                        [True, False, False, False],
                        [False, False, False, False]])

    assert (result == expected).all()
    assert (result == result.T).all()

def test_conn_mat_one_connection_2():
    """ Check that one connection is found"""
    positions = np.array([[[700, 700, 200],[700,-700,800],[900,1046.41,200],[-700,-700,200]]])
    positions = positions_noiser(positions)
    positions = positions[0]
    connections = np.array([[False, False, True, False, False, False], [False, False, False, False, False, False], [False, True, False, False, False, False], [False, False, False, False, False, False]])
    
    result = get_connection_mat(positions, connections)
    expected = np.array([[False, False, True, False],
                        [False, False, False, False],
                        [True, False, False, False],
                        [False, False, False, False]])
    
    assert (result == expected).all()
    assert (result == result.T).all()

def test_conn_mat_two_connections_1():
    """ Check that two connections are found (one line)"""
    positions = np.array([[[700, 700, 200],[700,-700,800],[900,1046.41,200],[1100,1328.3,200]]])
    positions = positions_noiser(positions)
    positions = positions[0]
    connections = np.array([[False, False, True, False, False, False], [False, False, False, False, False, False], [False, True, False, False, True, False], [False, False, True, False, False, False]])
    
    result = get_connection_mat(positions, connections)
    expected = np.array([[False, False, True, False],
                        [False, False, False, False],
                        [True, False, False, True],
                        [False, False, True, False]])

    assert (result == expected).all()
    assert (result == result.T).all()

def test_conn_mat_two_connections_2():
    """ Check that two connections are found (two blocks)"""
    positions = np.array([[[700, 700, 200],[700,-700,200],[900,1046.41,200],[700,-700,600]]])
    positions = positions_noiser(positions)
    positions = positions[0]
    connections = np.array([[False, False, True, False, False, False], [False, True, False, False, False, False], [False, True, False, False, False, False], [False, False, True, False, False, False]])
    
    result = get_connection_mat(positions, connections)
    expected = np.array([[False, False, True, False],
                        [False, False, False, True],
                        [True, False, False, False],
                        [False, True, False, False]])

    assert (result == expected).all()
    assert (result == result.T).all()

#####################################################################
################ Tests on distance computing ########################

def test_compute_dist_x():
    """ Test distance computng along axe x."""
    pos1 = np.array([700, 700, 700])
    pos2 = np.array([700, 800, 700])
    assert compute_distance(pos1, pos2) == 100

def test_compute_dist_xy():
    """ Test distance computng along axes x and y."""
    pos1 = np.array([750, 750, 700])
    pos2 = np.array([700, 700, 700])
    dist = compute_distance(pos1, pos2)
    assert dist > 70.71
    assert dist < 70.72

def test_compute_dist_xyz():
    """ Test distance computng along axes x, y and z."""
    pos1 = np.array([750, 750, 750])
    pos2 = np.array([700, 700, 700])
    dist = compute_distance(pos1, pos2)
    assert dist > 86.60
    assert dist < 86.61

#####################################################################
################## Tests on cube is_moving ##########################

def test_is_moving_false():
    pos1 = np.array([700,700,200])
    pos2 = np.array([700,700,200])
    assert(is_moving(pos1,pos2)==False)

def test_is_moving_true():
    pos1 = np.array([700,700,200])
    pos2 = np.array([700,700,400])
    assert(is_moving(pos1,pos2)==True)

def test_is_moving_true_2():
    pos1 = np.array([700,700,200])
    pos2 = np.array([700,500,200])
    assert(is_moving(pos1,pos2)==True)

#####################################################################
############ Tests on cube is_moving_constantly #####################

def test_is_not_moving():
    pos1 = np.array([700,700,200])
    pos2 = np.array([700,700,200])
    pos3 = np.array([700,700,200])
    assert(is_moving_constantly(pos1,pos2,pos3)==False)

def test_is_moving_constantly_true():
    pos1 = np.array([700,700,200])
    pos2 = np.array([700,700,400])
    pos3 = np.array([700,700,600])
    assert(is_moving_constantly(pos1,pos2,pos3)==True)

def test_is_moving_constantly_true_2():
    pos1 = np.array([700,700,200])
    pos2 = np.array([700,800,400])
    pos3 = np.array([700,900,600])
    assert(is_moving_constantly(pos1,pos2,pos3)==True)

def test_is_not_moving_constantly():
    pos1 = np.array([700,700,200])
    pos2 = np.array([700,700,600])
    pos3 = np.array([700,700,300])
    assert(is_moving_constantly(pos1,pos2,pos3)==False)

#####################################################################
############ Tests on on_ground function ############################

def test_on_groud():
    """ Check that a cube on mat is detected on ground."""
    pos = np.array([700,700,200])
    assert on_ground(pos,2) == True

def test_on_groud_noise():
    """ Check that a cube on mat is detected on ground with noise."""
    pos = np.array([700,700,200],dtype=float)
    pos = pos + np.random.normal(0, 5, 3)
    assert on_ground(pos,2) == True

def test_not_on_groud():
    """ Check that a cube not on mat is not detected on ground."""
    pos = np.array([700,700,300])
    assert on_ground(pos,2) == False

def test_not_on_groud_noise():
    """ Check that a cube not on mat is not detected on ground with noise."""
    pos = np.array([700,700,300],dtype=float)
    pos = pos + np.random.normal(0, 5, 3)
    assert on_ground(pos,2) == False

def test_on_groud_wheel():
    """ Check that a cube on mat is detected on ground with wheels size."""
    pos = np.array([700,700,250])
    assert on_ground(pos,3) == True

def test_on_groud_wheel_noise():
    """ Check that a cube on mat is detected on ground with wheels size and noise."""
    pos = np.array([700,700,250],dtype=float)
    pos = pos + np.random.normal(0, 5, 3)
    assert on_ground(pos,3) == True


#####################################################################
############ Tests on z_constant function ###########################

def test_z_constant():
    """ Check is z_constant work with z constant."""
    pos1 = np.array([700,700,200])
    pos2 = np.array([256,400,200])
    pos3 = np.array([-700,-700,200])
    assert z_constant(pos1, pos2, pos3) == True

def test_z_constant_noise():
    """ Check is z_constant work with z constant and noise."""
    pos1 = np.array([700,700,200])
    pos2 = np.array([256,400,200])
    pos3 = np.array([-700,-700,200])
    pos1 = pos1 + np.random.normal(0, 5, 3)
    pos2 = pos2 + np.random.normal(0, 5, 3)
    pos2 = pos2 + np.random.normal(0, 5, 3)
    assert z_constant(pos1, pos2, pos3) == True

def test_z_not_constant():
    """ Check is z_constant work with z not constant."""
    pos1 = np.array([700,700,200])
    pos2 = np.array([256,400,250])
    pos3 = np.array([-700,-700,181])
    assert z_constant(pos1, pos2, pos3) == False

def test_z_not_constant_noise():
    """ Check is z_constant work with z not constant and noise."""
    pos1 = np.array([700,700,200])
    pos2 = np.array([256,400,250])
    pos3 = np.array([-700,-700,181])
    pos1 = pos1 + np.random.normal(0, 5, 3)
    pos2 = pos2 + np.random.normal(0, 5, 3)
    pos2 = pos2 + np.random.normal(0, 5, 3)
    assert z_constant(pos1, pos2, pos3) == False

#####################################################################
############ Tests on propagate function ############################

def test_propagate_all_connected():
    """ Check is propagation work with one structure of 4 cubes."""
    conn_mat = np.array([[False, True, False, True],
                         [True, False, True, False],
                         [False, True, False, True],
                         [True, False, True, False]])
    state = [False, True, True, True]
    assert propagate(state, conn_mat) == [False, False, False, False]

def test_propagate_all_connected2():
    """ Check is propagation work with one structure of 4 cubes."""
    conn_mat = np.array([[False, True, False, True],
                         [True, False, True, False],
                         [False, True, False, True],
                         [True, False, True, False]])
    state = [True, False, True, True]
    assert propagate(state, conn_mat) == [False, False, False, False]

def test_propagate_1_block():
    """ Check is propagation work with one structure of 2 cubes."""
    conn_mat = np.array([[False, False, False, False],
                         [False, False, False, True],
                         [False, False, False, False],
                         [False, True, False, False]])
    state = [True, False, True, True]
    assert propagate(state, conn_mat) == [True, False, True, False]

def test_propagate_1_block2():
    """ Check is propagation work with one structure of 2 cubes."""
    conn_mat = np.array([[False, False, False, False],
                         [False, False, False, True],
                         [False, False, False, False],
                         [False, True, False, False]])
    state = [True, True, True, True]
    assert propagate(state, conn_mat) == [True, True, True, True]

def test_propagate_no_connection():
    """ Check is propagation work without ay connections."""
    conn_mat = np.array([[False, False, False, False],
                         [False, False, False, False],
                         [False, False, False, False],
                         [False, False, False, False]])
    state = [True, False, False, True]
    assert propagate(state, conn_mat) == [True, False, False, True]

def test_propagate_no_connection2():
    """ Check is propagation work without ay connections."""
    conn_mat = np.array([[False, False, False, False],
                         [False, False, False, False],
                         [False, False, False, False],
                         [False, False, False, False]])
    state = [True, False, True, False]
    assert propagate(state, conn_mat) == [True, False, True, False]