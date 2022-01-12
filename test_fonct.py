import numpy as np
from main import in_hand
from utils import positions_noiser

#####################################################################
#################### Fonctionnal tests ##############################

def test_f_1():
    """ Mini scenario in which cube 1 and 2 are manipulated."""
    positions = np.array([[[700, 700, 200], [700, -700, 200], [-700, 700, 200], [-700, -700, 200]],
                          [[740, 780, 800], [700, -700, 200], [-700, 700, 200], [-700, -700, 200]],
                          [[720, 750, 900], [720, -800, 500], [-700, 700, 200], [-700, -700, 200]],
                          [[700, 710, 700], [730, -800, 400], [-700, 700, 200], [-700, -700, 200]],
                          [[690, 760, 500], [720, -750, 500], [-700, 700, 200], [-700, -700, 200]]])
    connections = np.array([[[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
                            [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
                            [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
                            [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
                            [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]]])
    expected = np.array([[False, False, False, False],[True, False, False, False],[True, True, False, False],[True, True, False, False], [True, True, False, False]])
    assert (in_hand(positions,connections) == expected).all()

def test_f_1_noise():
    """ Mini scenario in which cube 1 and 2 are manipulated with noise."""
    positions = np.array([[[700, 700, 200], [700, -700, 200], [-700, 700, 200], [-700, -700, 200]],
                          [[740, 780, 800], [700, -700, 200], [-700, 700, 200], [-700, -700, 200]],
                          [[720, 750, 900], [720, -800, 500], [-700, 700, 200], [-700, -700, 200]],
                          [[700, 710, 700], [730, -800, 400], [-700, 700, 200], [-700, -700, 200]],
                          [[690, 760, 500], [720, -750, 500], [-700, 700, 200], [-700, -700, 200]]])
    positions = positions_noiser(positions)
    connections = np.array([[[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
                            [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
                            [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
                            [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
                            [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]]])
    expected = np.array([[False, False, False, False],[True, False, False, False],[True, True, False, False],[True, True, False, False], [True, True, False, False]])
    assert (in_hand(positions,connections) == expected).all()

def test_f_2():
    """ 
    Mini scenario in which cube 1 and 2 are manipulated.
    Then cube 2 is put on the ground.
    """
    positions = np.array([[[700, 700, 200], [700, -700, 200], [-700, 700, 200], [-700, -700, 200]],
                          [[740, 780, 800], [700, -700, 200], [-700, 700, 200], [-700, -700, 200]],
                          [[720, 750, 900], [720, -800, 500], [-700, 700, 200], [-700, -700, 200]],
                          [[700, 710, 700], [730, -800, 400], [-700, 700, 200], [-700, -700, 200]],
                          [[690, 760, 500], [720, -750, 200], [-700, 700, 200], [-700, -700, 200]],
                          [[690, 760, 500], [720, -750, 200], [-700, 700, 200], [-700, -700, 200]]])
    connections = np.array([[[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
                            [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
                            [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
                            [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
                            [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
                            [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]]])
    expected = np.array([[False, False, False, False],[True, False, False, False],[True, True, False, False],[True, True, False, False], [True, True, False, False], [True, False, False, False]])
    assert (in_hand(positions,connections) == expected).all()

def test_f_2_noise():
    """ 
    Mini scenario in which cube 1 and 2 are manipulated.
    Then cube 2 is put on the ground.
    Test with noise.
    """
    positions = np.array([[[700, 700, 200], [700, -700, 200], [-700, 700, 200], [-700, -700, 200]],
                          [[740, 780, 800], [700, -700, 200], [-700, 700, 200], [-700, -700, 200]],
                          [[720, 750, 900], [720, -800, 500], [-700, 700, 200], [-700, -700, 200]],
                          [[700, 710, 700], [730, -800, 400], [-700, 700, 200], [-700, -700, 200]],
                          [[690, 760, 500], [720, -750, 200], [-700, 700, 200], [-700, -700, 200]],
                          [[690, 760, 500], [720, -750, 200], [-700, 700, 200], [-700, -700, 200]]])
    positions = positions_noiser(positions)
    connections = np.array([[[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
                            [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
                            [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
                            [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
                            [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
                            [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]]])
    expected = np.array([[False, False, False, False],[True, False, False, False],[True, True, False, False],[True, True, False, False], [True, True, False, False], [True, False, False, False]])
    assert (in_hand(positions,connections) == expected).all()

def test_f_3():
    """ 
    Mini scenario in which cube 1, 2 and 3 are manipulated.
    A cube structure is build (2 and 3) and put on the mat.
    The structure move (maybe not possible with only these 2 cubes).
    """
    positions = np.array([[[700, 700, 200], [700, -700, 200], [-700, 700, 200], [-700, -700, 200]],
                          [[740, 780, 800], [700, -700, 200], [-500, 80, 400], [-700, -700, 200]],
                          [[720, 750, 900], [720, -800, 500], [-100, -200, 600], [-700, -700, 200]],
                          [[700, 710, 700], [730, -800, 400], [450, -600, 800], [-700, -700, 200]],
                          [[690, 760, 500], [720, -750, 200], [720, -750, 600], [-700, -700, 200]],
                          [[690, 760, 500], [720, -750, 200], [720, -750, 600], [-700, -700, 200]],
                          [[690, 760, 500], [720, -750, 200], [720, -750, 600], [-700, -700, 200]]])
    connections = np.array([[[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
                            [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
                            [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
                            [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
                            [[False, False, False, False, False, False], [False, False, True, False, False, False], [False, False, False, False, False, True], [False, False, False, False, False, False]],
                            [[False, False, False, False, False, False], [False, False, True, False, False, False], [False, False, False, False, False, True], [False, False, False, False, False, False]],
                            [[False, False, False, False, False, False], [False, False, True, False, False, False], [False, False, False, False, False, True], [False, False, False, False, False, False]]])
    expected = np.array([[False, False, False, False],[True, False, True, False],[True, True, True, False],[True, True, True, False], [True, True, True, False], [True, False, False, False], [True, False, False, False]])
    assert (in_hand(positions,connections) == expected).all()

def test_f_3_noise():
    """ 
    Mini scenario in which cube 1, 2 and 3 are manipulated.
    A cube structure is build (2 and 3) and put on the mat.
    The structure move (maybe not possible with only these 2 cubes).
    Test with noise.
    """
    positions = np.array([[[700, 700, 200], [700, -700, 200], [-700, 700, 200], [-700, -700, 200]],
                          [[740, 780, 800], [700, -700, 200], [-500, 80, 400], [-700, -700, 200]],
                          [[720, 750, 900], [720, -800, 500], [-100, -200, 600], [-700, -700, 200]],
                          [[700, 710, 700], [730, -800, 400], [450, -600, 800], [-700, -700, 200]],
                          [[690, 760, 500], [720, -750, 200], [720, -750, 600], [-700, -700, 200]],
                          [[690, 760, 500], [720, -750, 200], [720, -750, 600], [-700, -700, 200]],
                          [[690, 760, 500], [720, -750, 200], [720, -750, 600], [-700, -700, 200]]])
    positions = positions_noiser(positions)
    connections = np.array([[[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
                            [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
                            [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
                            [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
                            [[False, False, False, False, False, False], [False, False, True, False, False, False], [False, False, False, False, False, True], [False, False, False, False, False, False]],
                            [[False, False, False, False, False, False], [False, False, True, False, False, False], [False, False, False, False, False, True], [False, False, False, False, False, False]],
                            [[False, False, False, False, False, False], [False, False, True, False, False, False], [False, False, False, False, False, True], [False, False, False, False, False, False]]])
    expected = np.array([[False, False, False, False],[True, False, True, False],[True, True, True, False],[True, True, True, False], [True, True, True, False], [True, False, False, False], [True, False, False, False]])
    assert (in_hand(positions,connections) == expected).all()

def test_f_4():
    """ Mini scenario in which cube 1, 2 and 3 are manipulated."""
    positions = np.array([[[700, 700, 200], [700, -700, 200], [-700, 700, 200], [-700, -700, 200]],
                            [[740, 780, 800], [700, -700, 200], [-500, 80, 400], [-700, -700, 200]],
                            [[720, 750, 900], [720, -800, 500], [-100, -200, 600], [-700, -700, 200]],
                            [[700, 710, 700], [730, -800, 400], [450, -600, 800], [-700, -700, 200]],
                            [[690, 760, 500], [720, -750, 200], [720, -750, 600], [-700, -700, 200]],
                            [[690, 760, 500], [780, -750, 200], [780, -750, 600], [-700, -700, 200]],
                            [[690, 760, 500], [840, -750, 200], [840, -750, 600], [-700, -700, 200]],
                            [[690, 760, 500], [900, -750, 200], [900, -750, 600], [-700, -700, 200]]])
    connections = np.array([[[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
                            [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
                            [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
                            [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
                            [[False, False, False, False, False, False], [False, False, True, False, False, False], [False, False, False, False, False, True], [False, False, False, False, False, False]],
                            [[False, False, False, False, False, False], [False, False, True, False, False, False], [False, False, False, False, False, True], [False, False, False, False, False, False]],
                            [[False, False, False, False, False, False], [False, False, True, False, False, False], [False, False, False, False, False, True], [False, False, False, False, False, False]],
                            [[False, False, False, False, False, False], [False, False, True, False, False, False], [False, False, False, False, False, True], [False, False, False, False, False, False]]])
    not_expected = np.array([[False, False, False, False],[True, False, True, False],[True, True, True, False],[True, True, True, False], [True, True, True, False], [True, True, True, False], [True, True, True, False], [True, True, True, False]])
    in_hand(positions,connections)
    assert np.sum(in_hand(positions,connections) != not_expected) > 0

def test_f_4_noise():
    """ Mini scenario in which cube 1, 2 and 3 are manipulated with noise."""
    positions = np.array([[[700, 700, 200], [700, -700, 200], [-700, 700, 200], [-700, -700, 200]],
                            [[740, 780, 800], [700, -700, 200], [-500, 80, 400], [-700, -700, 200]],
                            [[720, 750, 900], [720, -800, 500], [-100, -200, 600], [-700, -700, 200]],
                            [[700, 710, 700], [730, -800, 400], [450, -600, 800], [-700, -700, 200]],
                            [[690, 760, 500], [720, -750, 200], [720, -750, 600], [-700, -700, 200]],
                            [[690, 760, 500], [780, -750, 200], [780, -750, 600], [-700, -700, 200]],
                            [[690, 760, 500], [840, -750, 200], [840, -750, 600], [-700, -700, 200]],
                            [[690, 760, 500], [900, -750, 200], [900, -750, 600], [-700, -700, 200]]])
    positions = positions_noiser(positions)
    connections = np.array([[[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
                            [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
                            [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
                            [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
                            [[False, False, False, False, False, False], [False, False, True, False, False, False], [False, False, False, False, False, True], [False, False, False, False, False, False]],
                            [[False, False, False, False, False, False], [False, False, True, False, False, False], [False, False, False, False, False, True], [False, False, False, False, False, False]],
                            [[False, False, False, False, False, False], [False, False, True, False, False, False], [False, False, False, False, False, True], [False, False, False, False, False, False]],
                            [[False, False, False, False, False, False], [False, False, True, False, False, False], [False, False, False, False, False, True], [False, False, False, False, False, False]]])
    not_expected = np.array([[False, False, False, False],[True, False, True, False],[True, True, True, False],[True, True, True, False], [True, True, True, False], [True, True, True, False], [True, True, True, False], [True, True, True, False]])
    in_hand(positions,connections)
    assert np.sum(in_hand(positions,connections) != not_expected) > 0