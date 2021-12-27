import numpy as np
from numpy.lib.function_base import disp
from main import in_hand, get_connection_mat, compute_distance

#####################################################################
################ Tests on data dimensions ###########################


def test_different_size():
    """ Check that an error is return if data don't have the same length """
    assert in_hand(np.zeros((10, 4, 3)), np.zeros((12, 4, 6))) == -1


def test_wrong_prop_1():
    """ Check that an error is return if data don't have the right dimensions"""
    assert in_hand(np.zeros((12, 4, 2)), np.zeros((12, 4, 6))) == -1


def test_wrong_prop_2():
    """ Check that an error is return if data don't have the right dimensions"""
    assert in_hand(np.zeros((12, 4, 3)), np.zeros((12, 5, 6))) == -1


def test_same_size():
    """ Check that something is done"""
    assert (in_hand(np.zeros((12, 4, 3)), np.zeros((12, 4, 6))) != -1).all()


#####################################################################
################ Tests on differents possibles cases ################

# # Tests on case of cubes on the tables
# def test_cube_table():
#     """ Check that cube immobile is detected as on the table"""
#     positions = np.array([[[700, 700, 200],[700,-700,200],[-700,700,200],[-700,-700,200]],
#                           [[700, 700, 200],[700,-700,200],[-700,700,200],[-700,-700,200]]])
#     connections = np.array([[[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
#                            [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]]])
#     assert (in_hand(positions, connections) == np.array([[False, False, False, False], [False, False, False, False]])).all()

# def test_cube_table_noise():
#     """ Check that cube immobile is detected as on the table with noise"""
#     positions = np.array([[[700, 700, 200],[700,-700,200],[-700,700,200],[-700,-700,200]],
#                           [[700, 700, 200],[700,-700,200],[-700,700,200],[-700,-700,200]]],
#                          dtype="float64")
#     positions = positions_noiser(positions)
#     connections = np.array([[[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
#                            [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]]])
#     assert (in_hand(positions, connections) == np.array([[False, False, False, False], [False, False, False, False]])).all()

# def test_cube_tabel_moving():
#     """ Check that cube is not immobile"""
#     positions = np.array([[[700, 700, 200],[700,-700,200],[-700,700,200],[-700,-700,200]],
#                           [[700, 700, 200],[700,-700,200],[-700,700,400],[-700,-700,200]]],
#                          dtype="float64")
#     connections = np.array([[[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
#                            [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]]])
#     assert (in_hand(positions, connections) != np.array([[False, False, False, False], [False, False, False, False]])).all()

# def test_cube_in_hand():
#     """ Check that cube immobile is not detected as on the table"""
#     positions = np.array([[[700, 700, 200],[700,-700,200],[-700,700,200],[-700,-700,300]],
#                           [[700, 700, 200],[700,-700,200],[-700,700,200],[-700,-700,200]]],
#                          dtype="float64")
#     connections = np.array([[[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
#                            [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]]])
#     assert (in_hand(positions, connections) != np.array([[False, False, False, False], [False, False, False, False]])).all()

# def test_cube_in_hand_noise():
#     """ Check that cube immobile is not detected as on the table"""
#     positions = np.array([[[700, 700, 200],[700,-700,200],[-700,700,200],[-700,-700,300]],
#                           [[700, 700, 200],[700,-700,200],[-700,700,200],[-700,-700,200]]],
#                          dtype="float64")
#     positions = positions_noiser(positions)
#     connections = np.array([[[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
#                            [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]]])
#     assert (in_hand(positions, connections) != np.array([[False, False, False, False], [False, False, False, False]])).all()

# # Tests on case of cubes on the tables
# def test_cube_moving_table_x():
#     """ Check that cube moving with a constant speed on the table is detected as on table (move following x)"""
#     positions = np.array([[[700, 700, 200],[700,-700,200],[-700,700,200],[-700,-700,200]],
#                           [[700, 700, 200],[800,-700,200],[-700,700,200],[-700,-700,200]],
#                           [[700, 700, 200],[900,-700,200],[-700,700,200],[-700,-700,200]]])
#     connections = np.array([[[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
#                             [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
#                             [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]]])
#     assert (in_hand(positions, connections) == np.array([[False, False, False, False], [False, False, False, False], [False, False, False, False]])).all()

# def test_cube_moving_table_x_noise():
#     """ Check that cube moving with a constant speed on the table is detected as on table (move following x) with noise"""
#     positions = np.array([[[700, 700, 200],[700,-700,200],[-700,700,200],[-700,-700,200]],
#                           [[700, 700, 200],[800,-700,200],[-700,700,200],[-700,-700,200]],
#                           [[700, 700, 200],[900,-700,200],[-700,700,200],[-700,-700,200]]])
#     positions = positions_noiser(positions)
#     connections = np.array([[[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
#                             [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
#                             [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]]])
#     assert (in_hand(positions, connections) == np.array([[False, False, False, False], [False, False, False, False], [False, False, False, False]])).all()


# def test_cube_moving_table_x_negative():
#     """ Check that cube moving with a constant speed on the table is detected as on table (move following -x)"""
#     positions = np.array([[[700, 700, 200],[700,-700,200],[-700,700,200],[-700,-700,200]],
#                           [[700, 700, 200],[700,-700,200],[-800,700,200],[-700,-700,200]],
#                           [[700, 700, 200],[700,-700,200],[-900,700,200],[-700,-700,200]]])
#     connections = np.array([[[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
#                             [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
#                             [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]]])
#     assert (in_hand(positions, connections) == np.array([[False, False, False, False], [False, False, False, False], [False, False, False, False]])).all()

# def test_cube_moving_table_x_negative():
#     """ Check that cube moving with a constant speed on the table is detected as on table (move following -x) with noise"""
#     positions = np.array([[[700, 700, 200],[700,-700,200],[-700,700,200],[-700,-700,200]],
#                           [[700, 700, 200],[700,-700,200],[-800,700,200],[-700,-700,200]],
#                           [[700, 700, 200],[700,-700,200],[-900,700,200],[-700,-700,200]]])
#     positions = positions_noiser(positions)
#     connections = np.array([[[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
#                             [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
#                             [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]]])
#     assert (in_hand(positions, connections) == np.array([[False, False, False, False], [False, False, False, False], [False, False, False, False]])).all()

# def test_cube_moving_table_y():
#     """ Check that cube moving with a constant speed on the table is detected as on table (move following y)"""
#     positions = np.array([[[700, 700, 200],[700,-700,200],[-700,700,200],[-700,-700,200]],
#                           [[700, 800, 200],[700,-700,200],[-700,700,200],[-700,-700,200]],
#                           [[700, 900, 200],[700,-700,200],[-700,700,200],[-700,-700,200]]])
#     connections = np.array([[[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
#                             [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
#                             [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]]])
#     assert (in_hand(positions, connections) == np.array([[False, False, False, False], [False, False, False, False], [False, False, False, False]])).all()

# def test_cube_moving_table_y_noise():
#     """ Check that cube moving with a constant speed on the table is detected as on table (move following y) with noise"""
#     positions = np.array([[[700, 700, 200],[700,-700,200],[-700,700,200],[-700,-700,200]],
#                           [[700, 800, 200],[700,-700,200],[-700,700,200],[-700,-700,200]],
#                           [[700, 900, 200],[700,-700,200],[-700,700,200],[-700,-700,200]]])
#     positions = positions_noiser(positions)
#     connections = np.array([[[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
#                             [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
#                             [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]]])
#     assert (in_hand(positions, connections) == np.array([[False, False, False, False], [False, False, False, False], [False, False, False, False]])).all()


# def test_cube_moving_table_y_negative():
#     """ Check that cube moving with a constant speed on the table is detected as on table (move following -y)"""
#     positions = np.array([[[700, 700, 200],[700,-700,200],[-700,700,200],[-700,-700,200]],
#                           [[700, 700, 200],[700,-700,200],[-700,700,200],[-700,-800,200]],
#                           [[700, 700, 200],[700,-700,200],[-700,700,200],[-700,-900,200]]])
#     connections = np.array([[[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
#                             [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
#                             [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]]])
#     assert (in_hand(positions, connections) == np.array([[False, False, False, False], [False, False, False, False], [False, False, False, False]])).all()

# def test_cube_moving_table_y_negative():
#     """ Check that cube moving with a constant speed on the table is detected as on table (move following -y) with noise"""
#     positions = np.array([[[700, 700, 200],[700,-700,200],[-700,700,200],[-700,-700,200]],
#                           [[700, 700, 200],[700,-700,200],[-700,700,200],[-700,-800,200]],
#                           [[700, 700, 200],[700,-700,200],[-700,700,200],[-700,-900,200]]])
#     positions = positions_noiser(positions)
#     connections = np.array([[[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
#                             [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
#                             [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]]])
#     assert (in_hand(positions, connections) == np.array([[False, False, False, False], [False, False, False, False], [False, False, False, False]])).all()

# def test_cube_moving_table_diag():
#     """ Check that cube moving with a constant speed on the table is detected as on table (move following 3x + 2y)"""
#     positions = np.array([[[700, 700, 200],[700,-700,200],[-700,700,200],[-700,-700,200]],
#                           [[760, 740, 200],[700,-700,200],[-700,700,200],[-700,-800,200]],
#                           [[820, 780, 200],[700,-700,200],[-700,700,200],[-700,-900,200]]])
#     connections = np.array([[[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
#                             [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
#                             [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]]])
#     assert (in_hand(positions, connections) == np.array([[False, False, False, False], [False, False, False, False], [False, False, False, False]])).all()

# def test_cube_moving_table_diag_noise():
#     """ Check that cube moving with a constant speed on the table is detected as on table (move following 3x + 2y) with noise"""
#     positions = np.array([[[700, 700, 200],[700,-700,200],[-700,700,200],[-700,-700,200]],
#                           [[760, 740, 200],[700,-700,200],[-700,700,200],[-700,-800,200]],
#                           [[820, 780, 200],[700,-700,200],[-700,700,200],[-700,-900,200]]])
#     positions = positions_noiser(positions)
#     connections = np.array([[[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
#                             [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
#                             [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]]])
#     assert (in_hand(positions, connections) == np.array([[False, False, False, False], [False, False, False, False], [False, False, False, False]])).all()


# def test_cube_moving_table_diag_negative():
#     """ Check that cube moving with a constant speed on the table is detected as on table (move following 3x - 2y) with noise"""
#     positions = np.array([[[700, 700, 200],[700,-700,200],[-700,700,200],[-700,-700,200]],
#                           [[760, 660, 200],[700,-700,200],[-700,700,200],[-700,-800,200]],
#                           [[820, 620, 200],[700,-700,200],[-700,700,200],[-700,-900,200]]])
#     connections = np.array([[[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
#                             [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
#                             [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]]])
#     assert (in_hand(positions, connections) == np.array([[False, False, False, False], [False, False, False, False], [False, False, False, False]])).all()

# def test_cube_moving_table_diag_negative():
#     """ Check that cube moving with a constant speed on the table is detected as on table (move following -y) with noise"""
#     positions = np.array([[[700, 700, 200],[700,-700,200],[-700,700,200],[-700,-700,200]],
#                           [[760, 660, 200],[700,-700,200],[-700,700,200],[-700,-800,200]],
#                           [[820, 620, 200],[700,-700,200],[-700,700,200],[-700,-900,200]]])
#     positions = positions_noiser(positions)
#     connections = np.array([[[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
#                             [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
#                             [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]]])
#     assert (in_hand(positions, connections) == np.array([[False, False, False, False], [False, False, False, False], [False, False, False, False]])).all()

# def test_cube_moving_table_z():
#     """ Check that cube moving with a constant speed (move following z)"""
#     positions = np.array([[[700, 700, 200],[700,-700,200],[-700,700,200],[-700,-700,200]],
#                           [[700, 700, 200],[700,-700,300],[-700,700,200],[-700,-800,200]],
#                           [[700, 700, 200],[700,-700,400],[-700,700,200],[-700,-900,200]]])
#     connections = np.array([[[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
#                             [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
#                             [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]]])
#     assert (in_hand(positions, connections) != np.array([[False, False, False, False], [False, False, False, False], [False, False, False, False]])).all()

# def test_cube_moving_table_z_noise():
#     """ Check that cube moving with a constant speed (move following z) with noise"""
#     positions = np.array([[[700, 700, 200],[700,-700,200],[-700,700,200],[-700,-700,200]],
#                           [[700, 700, 200],[700,-700,300],[-700,700,200],[-700,-800,200]],
#                           [[700, 700, 200],[700,-700,400],[-700,700,200],[-700,-900,200]]])
#     positions = positions_noiser(positions)
#     connections = np.array([[[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
#                             [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
#                             [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]]])
#     assert (in_hand(positions, connections) != np.array([[False, False, False, False], [False, False, False, False], [False, False, False, False]])).all()

# def test_cube_moving_table_z_negative():
#     """ Check that cube moving with a constant speed (move following -z)"""
#     positions = np.array([[[700, 700, 200],[700,-700,800],[-700,700,200],[-700,-700,200]],
#                           [[700, 700, 200],[700,-700,700],[-700,700,200],[-700,-800,200]],
#                           [[700, 700, 200],[700,-700,600],[-700,700,200],[-700,-900,200]]])
#     connections = np.array([[[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
#                             [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
#                             [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]]])
#     assert (in_hand(positions, connections) != np.array([[False, False, False, False], [False, False, False, False], [False, False, False, False]])).all()

# def test_cube_moving_table_z_negative_noise():
#     """ Check that cube moving with a constant speed (move following -z) with noise"""
#     positions = np.array([[[700, 700, 200],[700,-700,800],[-700,700,200],[-700,-700,200]],
#                           [[700, 700, 200],[700,-700,700],[-700,700,200],[-700,-800,200]],
#                           [[700, 700, 200],[700,-700,600],[-700,700,200],[-700,-900,200]]])
#     positions = positions_noiser(positions)
#     connections = np.array([[[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
#                             [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
#                             [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]]])
#     assert (in_hand(positions, connections) != np.array([[False, False, False, False], [False, False, False, False], [False, False, False, False]])).all()


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
    pos1 = np.array([700, 700, 700])
    pos2 = np.array([700, 800, 700])
    assert compute_distance(pos1, pos2) == 100

def test_compute_dist_xy():
    pos1 = np.array([750, 750, 700])
    pos2 = np.array([700, 700, 700])
    dist = compute_distance(pos1, pos2)
    assert dist > 70.71
    assert dist < 70.72

def test_compute_dist_xyz():
    pos1 = np.array([750, 750, 750])
    pos2 = np.array([700, 700, 700])
    dist = compute_distance(pos1, pos2)
    assert dist > 86.60
    assert dist < 86.61

#####################################################################
#################### Tests on z > 4*coté ############################

# In_hand is true
def test_z_sup_4_cote():
    #COTE cube = 400
    positions = np.array([700, 700, 2000])
    connections = np.array([False, False, False, False, False, False])
    assert (in_hand(positions,connections))

#####################################################################
#################### Tests on z augmente ############################

# def test_z_augmente():
#     pos_1 = np.array([700, 700, 700])
#     pos_2 = np.array([700, 700, 800])
#     assert (in_hand(p))


#####################################################################
################ Functions to help to create test data ##############

def positions_noiser(positions):
    for time_stamp in positions:
        for cube in time_stamp:
            cube = cube + np.random.normal(0, 5, 3)
    print(positions)
    return positions