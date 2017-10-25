"""Tests for ConjugateGradients solvers."""

import pytest
from scripts.ConjugateGradients.utils import get_solver
from scripts.ConjugateGradients.test_matrices import TestMatrices

import numpy as np


@pytest.fixture
def test_matrix():
    matrix_size = 100
    a_matrix = TestMatrices.get_diagonal_matrix(matrix_size) * 10
    x_vec = np.vstack([1 for x in range(matrix_size)])
    b_vec = np.vstack([0 for x in range(matrix_size)])
    return a_matrix, x_vec, b_vec


def test_cg(test_matrix):
    a_matrix, x_vec, b_vec = test_matrix
    Solver = get_solver()
    solver = Solver(a_matrix, x_vec, b_vec)
    results = solver.solve()
    assert 0.99999 < np.linalg.norm(results[0]) < 1.0001


def test_pcg_jacobi(test_matrix):
    a_matrix, x_vec, b_vec = test_matrix
    Solver = get_solver('PCG')
    solver = Solver(a_matrix, x_vec, b_vec)
    results = solver.solve()
    assert 0.99999 < np.linalg.norm(results[0]) < 1.0001
