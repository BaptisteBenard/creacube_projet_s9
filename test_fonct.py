import numpy as np
from main import in_hand
from utils import positions_noiser

#####################################################################
#################### Fonctionnal tests ##############################

def test_f_1():
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