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

def test_f_2():
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
    positions = np.array([[[700, 700, 200], [700, -700, 200], [-700, 700, 200], [-700, -700, 200]],
                          [[740, 780, 800], [700, -700, 200], [-500, 80, 400], [-700, -700, 200]],
                          [[720, 750, 900], [720, -800, 500], [-100, -200, 600], [-700, -700, 200]],
                          [[700, 710, 700], [730, -800, 400], [450, -600, 800], [-700, -700, 200]],
                          [[690, 760, 500], [720, -750, 200], [720, -750, 600], [-700, -700, 200]],
                          [[690, 760, 500], [760, -750, 200], [760, -750, 600], [-700, -700, 200]],
                          [[690, 760, 500], [800, -750, 200], [800, -750, 600], [-700, -700, 200]]])
    connections = np.array([[[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
                            [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
                            [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
                            [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
                            [[False, False, False, False, False, False], [False, False, True, False, False, False], [False, False, False, False, False, True], [False, False, False, False, False, False]],
                            [[False, False, False, False, False, False], [False, False, True, False, False, False], [False, False, False, False, False, True], [False, False, False, False, False, False]],
                            [[False, False, False, False, False, False], [False, False, True, False, False, False], [False, False, False, False, False, True], [False, False, False, False, False, False]]])
    expected = np.array([[False, False, False, False],[True, False, True, False],[True, True, True, False],[True, True, True, False], [True, True, True, False], [True, True, True, False], [True, False, False, False]])
    assert (in_hand(positions,connections) == expected).all()

def test_f_4():
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