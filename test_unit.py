import numpy as np
from main import in_hand
from utils import positions_noiser

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

# Tests on case of cubes on the tables
def test_cube_table():
    """ Check that cube immobile is detected as on the table"""
    positions = np.array([[[700, 700, 200],[700,-700,200],[-700,700,200],[-700,-700,200]],
                          [[700, 700, 200],[700,-700,200],[-700,700,200],[-700,-700,200]]])
    connections = np.array([[[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
                           [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]]])
    assert (in_hand(positions, connections) == np.array([[False, False, False, False], [False, False, False, False]])).all()

def test_cube_table_noise():
    """ Check that cube immobile is detected as on the table with noise"""
    positions = np.array([[[700, 700, 200],[700,-700,200],[-700,700,200],[-700,-700,200]],
                          [[700, 700, 200],[700,-700,200],[-700,700,200],[-700,-700,200]]],
                         dtype="float64")
    positions = positions_noiser(positions)
    connections = np.array([[[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
                           [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]]])
    assert (in_hand(positions, connections) == np.array([[False, False, False, False], [False, False, False, False]])).all()

def test_cube_table_moving():
    """ Check that cube is not immobile"""
    positions = np.array([[[700, 700, 200],[700,-700,200],[-700,700,200],[-700,-700,200]],
                          [[700, 700, 200],[700,-700,200],[-700,700,400],[-700,-700,200]]],
                         dtype="float64")
    connections = np.array([[[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
                           [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]]])
    not_expected = np.array([[False, False, False, False], [False, False, False, False]])
    assert np.sum(in_hand(positions, connections) != not_expected) > 0


def test_cube_in_hand():
    """ Check that cube immobile is not detected as on the table"""
    positions = np.array([[[700, 700, 200],[700,-700,200],[-700,700,200],[-700,-700,200]],
                          [[700, 700, 200],[700,-700,200],[-700,700,200],[-700,-700,300]],
                          [[700, 700, 200],[700,-700,200],[-700,700,200],[-700,-700,200]]],
                         dtype="float64")
    connections = np.array([[[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
                            [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
                            [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]]])
    not_expected = np.array([[False, False, False, False], [False, False, False, True], [False, False, False, True]])
    assert (in_hand(positions, connections) == not_expected).all()

def test_cube_in_hand_noise():
    """ Check that cube immobile is not detected as on the table"""
    positions = np.array([[[700, 700, 200],[700,-700,200],[-700,700,200],[-700,-700,200]],
                          [[700, 700, 200],[700,-700,200],[-700,700,200],[-700,-700,300]],
                          [[700, 700, 200],[700,-700,200],[-700,700,200],[-700,-700,200]]],
                         dtype="float64")
    positions = positions_noiser(positions)
    connections = np.array([[[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
                            [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
                            [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]]])
    not_expected = np.array([[False, False, False, False], [False, False, False, True], [False, False, False, True]])
    assert (in_hand(positions, connections) == not_expected).all()

# Tests on case of cubes on the tables
def test_cube_moving_table_x():
    """ Check that cube moving with a constant speed on the table is detected as on table (move following x)"""
    positions = np.array([[[700, 700, 200],[700,-700,200],[-700,700,200],[-700,-700,200]],
                          [[700, 700, 200],[800,-700,200],[-700,700,200],[-700,-700,200]],
                          [[700, 700, 200],[900,-700,200],[-700,700,200],[-700,-700,200]]])
    connections = np.array([[[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
                            [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
                            [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]]])
    assert (in_hand(positions, connections) == np.array([[False, False, False, False], [False, False, False, False], [False, False, False, False]])).all()

def test_cube_moving_table_x_noise():
    """ Check that cube moving with a constant speed on the table is detected as on table (move following x) with noise"""
    positions = np.array([[[700, 700, 200],[700,-700,200],[-700,700,200],[-700,-700,200]],
                          [[700, 700, 200],[800,-700,200],[-700,700,200],[-700,-700,200]],
                          [[700, 700, 200],[900,-700,200],[-700,700,200],[-700,-700,200]]])
    positions = positions_noiser(positions)
    connections = np.array([[[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
                            [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
                            [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]]])
    assert (in_hand(positions, connections) == np.array([[False, False, False, False], [False, False, False, False], [False, False, False, False]])).all()


def test_cube_moving_table_x_negative():
    """ Check that cube moving with a constant speed on the table is detected as on table (move following -x)"""
    positions = np.array([[[700, 700, 200],[700,-700,200],[-700,700,200],[-700,-700,200]],
                          [[700, 700, 200],[700,-700,200],[-800,700,200],[-700,-700,200]],
                          [[700, 700, 200],[700,-700,200],[-900,700,200],[-700,-700,200]]])
    connections = np.array([[[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
                            [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
                            [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]]])
    assert (in_hand(positions, connections) == np.array([[False, False, False, False], [False, False, False, False], [False, False, False, False]])).all()

def test_cube_moving_table_x_negative():
    """ Check that cube moving with a constant speed on the table is detected as on table (move following -x) with noise"""
    positions = np.array([[[700, 700, 200],[700,-700,200],[-700,700,200],[-700,-700,200]],
                          [[700, 700, 200],[700,-700,200],[-800,700,200],[-700,-700,200]],
                          [[700, 700, 200],[700,-700,200],[-900,700,200],[-700,-700,200]]])
    positions = positions_noiser(positions)
    connections = np.array([[[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
                            [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
                            [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]]])
    assert (in_hand(positions, connections) == np.array([[False, False, False, False], [False, False, False, False], [False, False, False, False]])).all()

def test_cube_moving_table_y():
    """ Check that cube moving with a constant speed on the table is detected as on table (move following y)"""
    positions = np.array([[[700, 700, 200],[700,-700,200],[-700,700,200],[-700,-700,200]],
                          [[700, 800, 200],[700,-700,200],[-700,700,200],[-700,-700,200]],
                          [[700, 900, 200],[700,-700,200],[-700,700,200],[-700,-700,200]]])
    connections = np.array([[[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
                            [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
                            [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]]])
    assert (in_hand(positions, connections) == np.array([[False, False, False, False], [False, False, False, False], [False, False, False, False]])).all()

def test_cube_moving_table_y_noise():
    """ Check that cube moving with a constant speed on the table is detected as on table (move following y) with noise"""
    positions = np.array([[[700, 700, 200],[700,-700,200],[-700,700,200],[-700,-700,200]],
                          [[700, 800, 200],[700,-700,200],[-700,700,200],[-700,-700,200]],
                          [[700, 900, 200],[700,-700,200],[-700,700,200],[-700,-700,200]]])
    positions = positions_noiser(positions)
    connections = np.array([[[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
                            [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
                            [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]]])
    assert (in_hand(positions, connections) == np.array([[False, False, False, False], [False, False, False, False], [False, False, False, False]])).all()


def test_cube_moving_table_y_negative():
    """ Check that cube moving with a constant speed on the table is detected as on table (move following -y)"""
    positions = np.array([[[700, 700, 200],[700,-700,200],[-700,700,200],[-700,-700,200]],
                          [[700, 700, 200],[700,-700,200],[-700,700,200],[-700,-800,200]],
                          [[700, 700, 200],[700,-700,200],[-700,700,200],[-700,-900,200]]])
    connections = np.array([[[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
                            [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
                            [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]]])
    assert (in_hand(positions, connections) == np.array([[False, False, False, False], [False, False, False, False], [False, False, False, False]])).all()

def test_cube_moving_table_y_negative():
    """ Check that cube moving with a constant speed on the table is detected as on table (move following -y) with noise"""
    positions = np.array([[[700, 700, 200],[700,-700,200],[-700,700,200],[-700,-700,200]],
                          [[700, 700, 200],[700,-700,200],[-700,700,200],[-700,-800,200]],
                          [[700, 700, 200],[700,-700,200],[-700,700,200],[-700,-900,200]]])
    positions = positions_noiser(positions)
    connections = np.array([[[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
                            [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
                            [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]]])
    assert (in_hand(positions, connections) == np.array([[False, False, False, False], [False, False, False, False], [False, False, False, False]])).all()

def test_cube_moving_table_diag():
    """ Check that cube moving with a constant speed on the table is detected as on table (move following 3x + 2y)"""
    positions = np.array([[[700, 700, 200],[700,-700,200],[-700,700,200],[-700,-700,200]],
                          [[760, 740, 200],[700,-700,200],[-700,700,200],[-700,-800,200]],
                          [[820, 780, 200],[700,-700,200],[-700,700,200],[-700,-900,200]]])
    connections = np.array([[[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
                            [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
                            [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]]])
    assert (in_hand(positions, connections) == np.array([[False, False, False, False], [False, False, False, False], [False, False, False, False]])).all()

def test_cube_moving_table_diag_noise():
    """ Check that cube moving with a constant speed on the table is detected as on table (move following 3x + 2y) with noise"""
    positions = np.array([[[700, 700, 200],[700,-700,200],[-700,700,200],[-700,-700,200]],
                          [[760, 740, 200],[700,-700,200],[-700,700,200],[-700,-800,200]],
                          [[820, 780, 200],[700,-700,200],[-700,700,200],[-700,-900,200]]])
    positions = positions_noiser(positions)
    connections = np.array([[[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
                            [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
                            [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]]])
    assert (in_hand(positions, connections) == np.array([[False, False, False, False], [False, False, False, False], [False, False, False, False]])).all()


def test_cube_moving_table_diag_negative():
    """ Check that cube moving with a constant speed on the table is detected as on table (move following 3x - 2y) with noise"""
    positions = np.array([[[700, 700, 200],[700,-700,200],[-700,700,200],[-700,-700,200]],
                          [[760, 660, 200],[700,-700,200],[-700,700,200],[-700,-800,200]],
                          [[820, 620, 200],[700,-700,200],[-700,700,200],[-700,-900,200]]])
    connections = np.array([[[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
                            [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
                            [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]]])
    assert (in_hand(positions, connections) == np.array([[False, False, False, False], [False, False, False, False], [False, False, False, False]])).all()

def test_cube_moving_table_diag_negative():
    """ Check that cube moving with a constant speed on the table is detected as on table (move following -y) with noise"""
    positions = np.array([[[700, 700, 200],[700,-700,200],[-700,700,200],[-700,-700,200]],
                          [[760, 660, 200],[700,-700,200],[-700,700,200],[-700,-800,200]],
                          [[820, 620, 200],[700,-700,200],[-700,700,200],[-700,-900,200]]])
    positions = positions_noiser(positions)
    connections = np.array([[[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
                            [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
                            [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]]])
    assert (in_hand(positions, connections) == np.array([[False, False, False, False], [False, False, False, False], [False, False, False, False]])).all()

# Ne marche pas
def test_cube_moving_table_z():
    """ Check that cube moving with a constant speed (move following z)"""
    positions = np.array([[[700, 700, 200],[700,-700,200],[-700,700,200],[-700,-700,200]],
                          [[700, 700, 200],[700,-700,300],[-700,700,200],[-700,-800,200]],
                          [[700, 700, 200],[700,-700,400],[-700,700,200],[-700,-900,200]]])
    connections = np.array([[[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
                            [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
                            [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]]])
    print(in_hand(positions,connections))
    # résultat du print : [[False False False False], [False  True False False],[False  True False False]]
    # cet assert devrait donc être correct non ?
    not_expected = np.array([[False, False, False, False], [False, False, False, False], [False, False, False, False]])
    assert np.sum(in_hand(positions, connections) != not_expected) > 0

def test_cube_moving_table_z_noise():
    """ Check that cube moving with a constant speed (move following z) with noise"""
    positions = np.array([[[700, 700, 200],[700,-700,200],[-700,700,200],[-700,-700,200]],
                          [[700, 700, 200],[700,-700,300],[-700,700,200],[-700,-800,200]],
                          [[700, 700, 200],[700,-700,400],[-700,700,200],[-700,-900,200]]])
    positions = positions_noiser(positions)
    connections = np.array([[[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
                            [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
                            [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]]])
    not_expected = np.array([[False, False, False, False], [False, False, False, False], [False, False, False, False]])
    assert np.sum(in_hand(positions, connections) != not_expected) > 0

def test_cube_moving_table_z_negative():
    """ Check that cube moving with a constant speed (move following -z)"""
    positions = np.array([[[700, 700, 200],[700,-700,200],[-700,700,200],[-700,-700,200]],
                          [[700, 700, 200],[700,-700,800],[-700,700,200],[-700,-700,200]],
                          [[700, 700, 200],[700,-700,700],[-700,700,200],[-700,-800,200]],
                          [[700, 700, 200],[700,-700,600],[-700,700,200],[-700,-900,200]]])
    connections = np.array([[[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
                            [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
                            [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
                            [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]]])
    print(in_hand(positions,connections))
    not_expected = np.array([[False, False, False, False], [False, False, False, False], [False, False, False, False], [False, False, False, False]])
    assert np.sum(in_hand(positions, connections) != not_expected) > 0

def test_cube_moving_table_z_negative():
    """ Check that cube moving with a constant speed (move following -z)"""
    positions = np.array([[[700, 700, 200],[700,-700,200],[-700,700,200],[-700,-700,200]],
                          [[700, 700, 200],[700,-700,800],[-700,700,200],[-700,-700,200]],
                          [[700, 700, 200],[700,-700,700],[-700,700,200],[-700,-800,200]],
                          [[700, 700, 200],[700,-700,600],[-700,700,200],[-700,-900,200]]])
    positions = positions_noiser(positions)
    connections = np.array([[[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
                            [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
                            [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
                            [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]]])
    print(in_hand(positions,connections))
    not_expected = np.array([[False, False, False, False], [False, False, False, False], [False, False, False, False], [False, False, False, False]])
    assert np.sum(in_hand(positions, connections) != not_expected) > 0


#####################################################################
#################### Tests on z > 4*coté ############################

# In_hand is true
def test_z_sup_4_cote():
    positions = np.array([[[700, 700, 2000], [700, -700, 2000], [-700, 700, 2000], [-700, -700, 2000]]])
    connections = np.array([[[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]]])
    expected = np.array([[True, True, True, True]])
    assert (in_hand(positions,connections) == expected).all()

def test_z_sup_4_cote_2():
    positions = np.array([[[700, 700, 200], [700, -700, 200], [-700, 700, 2000], [-700, -700, 200]]])
    connections = np.array([[[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]]])
    expected = np.array([[False, False, True, False]])
    assert (in_hand(positions,connections) == expected).all()

def test_z_sup_4_cote_2_noise():
    positions = np.array([[[700, 700, 200], [700, -700, 200], [-700, 700, 1600], [-700, -700, 200]]])
    positions = positions_noiser(positions)
    connections = np.array([[[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]]])
    expected = np.array([[False, False, True, False]])
    assert (in_hand(positions,connections) == expected).all()

def test_z_sup_4_cote_false():
    positions = np.array([[[700, 700, 200], [700, -700, 600], [-700, 700, 1000], [-700, -700, 1400]]])
    connections = np.array([[[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]]])
    expected = np.array([[False, False, False, False]])
    assert (in_hand(positions,connections) == expected).all()


#####################################################################
#################### Tests on z augmente ############################

# Z augmente -> In_hand is true
def test_z_augmente():
    positions = np.array([[[700, 700, 200],[700,-700,200],[-700,700,200],[-700,-700,200]],
                         [[700, 700, 300],[700,-700,200],[-700,700,200],[-700,-700,200]]])
    connections = np.array([[[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
                            [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]]])
    expected = np.array([[False, False, False, False],[True, False, False, False]])
    # print(in_hand(positions,connections))
    assert (in_hand(positions,connections)==expected).all()

def test_z_augmente_noise():
    positions = np.array([[[700, 700, 200],[700,-700,200],[-700,700,200],[-700,-700,200]],
                         [[700, 700, 300],[700,-700,200],[-700,700,200],[-700,-700,200]]])
    positions = positions_noiser(positions)
    connections = np.array([[[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
                            [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]]])
    expected = np.array([[False, False, False, False],[True, False, False, False]])
    # print(in_hand(positions,connections))
    assert (in_hand(positions,connections)==expected).all()

