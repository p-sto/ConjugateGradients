"""Tests for ConjugateGradients solvers."""

import pytest
from scripts.ConjugateGradients.utils import get_test_matrix_diagonal, get_solver

import numpy as np


@pytest.fixture
def test_matrix():
    matrix_size = 100
    a_matrix = get_test_matrix_diagonal(matrix_size)
    x_vec = np.vstack([1 for x in range(matrix_size)])
    b_vec = np.vstack([0 for x in range(matrix_size)])
    return a_matrix, x_vec, b_vec


def test_cg(test_matrix):
    a_matrix, x_vec, b_vec = test_matrix
    Solver = get_solver()
    solver = Solver(a_matrix, x_vec, b_vec)
    x_vec, i = solver.solve()
    assert 0.99999 < np.linalg.norm(x_vec) < 1.0001


def test_pcg(test_matrix):
    a_matrix, x_vec, b_vec = test_matrix
    Solver = get_solver('PCG')
    solver = Solver(a_matrix, x_vec, b_vec)
    x_vec, i = solver.solve()
    assert 0.99999 < np.linalg.norm(x_vec) < 1.0001
