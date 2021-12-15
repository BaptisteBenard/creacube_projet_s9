import numpy as np
from main import in_hand

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
    assert in_hand(np.zeros((12, 4, 3)), np.zeros((12, 4, 6))) != -1