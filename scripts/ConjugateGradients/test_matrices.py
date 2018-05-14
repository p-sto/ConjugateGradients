"""Test matrices"""
import random
import re

from numpy.ma import ceil, sqrt
from numpy.ma import floor

import numpy as np
from scipy.sparse.csr import csr_matrix


class TestMatrices:
    """Test matrices generating methods."""

    @classmethod
    def get_diagonal_matrix(cls, size: int = 50) -> np.matrix:
        """Return size by size test diagonal matrix."""
        diagonal = np.diagflat([1] * size)
        return np.matrix(diagonal)

    @classmethod
    def get_diagonal_matrix_csr(cls, size: int = 50) -> csr_matrix:
        """Return diagonal matrix in CSR format."""
        return csr_matrix(cls.get_diagonal_matrix(size=size))

    @classmethod
    def get_matrix_three_diagonal(cls, size: int = 50) -> np.matrix:
        """Return size by size test 3 diagonal matrix."""
        upper_diagonal = np.diagflat([1] * (size - 1), 1)
        lower_diagonal = np.diagflat([1] * (size - 1), -1)
        diagonal = np.diagflat([10] * size)
        return np.matrix(lower_diagonal + diagonal + upper_diagonal)

    @classmethod
    def get_matrix_three_diagonal_csr(cls, size: int = 50) -> csr_matrix:
        """Return three diagonal test matrix in csr format."""
        return csr_matrix(cls.get_matrix_three_diagonal(size=size))

    @staticmethod
    def _curve_mask(size: int):
        """Return curved mask for random matrix"""
        mask = np.zeros((size, size))
        prev = 0
        for ind in range(size):
            _y = size - int(round(float(sqrt(pow(size, 2) - pow(ind, 2)))))
            mask[ind, int(ceil(_y/2))] = 1
            mask[int(ceil(_y/2)), ind] = 1
            mask[ind, _y] = 1
            mask[_y, ind] = 1
            for _i in range(prev, _y):
                mask[ind, _i] = 1
                mask[_i, ind] = 1
                mask[ind, int(ceil(_i/2))] = 1
                mask[int(ceil(_i/2)), ind] = 1
            if ind == size - 1:
                for _i in range(prev, size):
                    mask[ind, _i] = 1
                    mask[_i, ind] = 1
            prev = _y
            mask[ind, ind] = 1
        return mask

    @staticmethod
    def _get_arrow_mask(size: int) -> np.matrix:
        """Return arrow-like mask for random matrix"""
        mask = np.zeros((size, size))
        for ind in range(size):
            mask[ind, ind] = 1
            if ind > 0:
                length = random.randint(1, int(np.log(ind) * 2) + 1) * random.randint(0, 1)    # pylint: disable=no-member
                if ind - length > 0:
                    mask[ind-length:ind, ind] = 1
                    mask[ind, ind-length:ind] = 1
        return mask

    @staticmethod
    def _get_noise_mask(size: int) -> np.matrix:
        """Return random noise"""
        density = 0.1
        mask = np.zeros((size, size))
        for ind in np.random.choice(size-1, int(floor(size * density))):        # pylint: disable=no-member
            how_many_elmns = random.randint(1, int(sqrt(ind)) + 1)
            for var in range(1, how_many_elmns):                                # pylint: disable=unused-variable
                if random.randint(0, 1):
                    length = random.randint(1, int(sqrt(size * density)))
                    rnd_from = random.randint(1, ind) - length
                    rnd_to = rnd_from + length
                    mask[rnd_from:rnd_to, ind] = 1
                    mask[ind, rnd_from:rnd_to] = 1
        return mask

    @staticmethod
    def _get_quadratic_mask(size: int) -> np.matrix:
        """Return mask for triangle test matrix."""
        mask = np.zeros((size, size))
        offset = int(floor(size/150))   # add extra diagonal for each 150elements grown size
        # lets add some randomly distributed diagonals
        extra_diagonals = [x for x in range(1, offset) if random.randint(0, 1)]
        for ind in range(size):
            mask[ind, ind] = 1
            mask[ind, int(ceil(ind/2))] = 1
            mask[int(ceil(ind/2)), ind] = 1
            mask[ind, int(ceil(ind/4))] = 1
            mask[int(ceil(ind/4)), ind] = 1
            # will add extra diagonals randomly
            for add in extra_diagonals:
                if ind + add < size:
                    mask[ind + add, ind] = 1
                    mask[ind + add, int(ceil(ind/2))] = 1
                    mask[int(ceil(ind/2)), ind + add] = 1
                    mask[ind + add, int(ceil(ind/4))] = 1
                    mask[int(ceil(ind/4)), ind + add] = 1
                if ind - add >= 0:
                    mask[ind - add, ind] = 1
                    mask[ind - add, int(ceil(ind/2))] = 1
                    mask[int(ceil(ind/2)), ind - add] = 1
                    mask[ind - add, int(ceil(ind/4))] = 1
                    mask[int(ceil(ind/4)), ind - add] = 1
        for splitter in range(int(np.sqrt(size))):
            matrix_splitter = ceil(float(pow(2, splitter + 1)))
            split_value = size - int(size/matrix_splitter)
            for ind in range(split_value, size):
                for add in [0] + extra_diagonals:
                    if split_value + add < size:
                        mask[ind, split_value + add] = 1
                        mask[split_value + add, ind] = 1
        return mask

    @staticmethod
    def _get_rectangle_mask(size: int) -> np.matrix:    # pylint: disable=too-many-branches
        """Return mask for rectangle test matrix."""
        mask = np.zeros((size, size))
        rnd = lambda: random.randint(0, 1)
        for ind in range(size):
            if ind < 3 * int(size/4):
                for _y in range(0, int(size/4)):
                    val = rnd() * rnd() * rnd()
                    mask[ind, size - _y - 1] = val
                    mask[size - _y - 1, ind] = val
                if ind < int(size/8):
                    for _y in range(0, int(size/8)):
                        if _y > ind:
                            # probability of heaving 1 is 0.5 * 0.5
                            val = rnd() * rnd()
                            mask[ind, _y] = val
                            mask[_y, ind] = val
                elif ind < int(size/4):
                    for _y in range(int(size/8), int(size/4)):
                        if _y > ind:
                            val = rnd() * rnd()
                            mask[ind, _y] = val
                            mask[_y, ind] = val
                elif ind < int(size/2):
                    for _y in range(int(size/4), int(size/2)):
                        if _y > ind:
                            val = rnd() * rnd() * rnd()
                            mask[ind, _y] = val
                            mask[_y, ind] = val
                elif ind < 5*int(size/8):
                    for _y in range(int(size/2), 5*int(size/8)):
                        if _y > ind:
                            val = rnd() * rnd()
                            mask[ind, _y] = val
                            mask[_y, ind] = val
            elif ind > size - int(size/8):
                for _y in range(0, int(size/8)):
                    val = rnd() * rnd()
                    if size - _y - 1 < ind:
                        mask[ind, size - _y - 1] = val
                        mask[size - _y - 1, ind] = val
            mask[ind, ind] = 1
        return mask

    @classmethod
    def get_random_test_matrix(cls, size: int = 50, pattern: str = 'q') -> np.matrix:
        """Return positively defined matrix used for testing purposes.

        Matrix will be positively define if its eigenvalues are positive, to achieve this it can be stated that:
        A = Q'DQ, where Q is random matrix, D diagonal matrix with positive elements on its diagonal.

        :param size: represents size of matrix which as to be returned
        :param pattern: represents pattern of random generated matrix e.g. quadratic, arrow, curve, rectangle, noise
        :returns: generated matrix
        """
        rand_matrix = np.matrix(np.random.rand(size, size))    # pylint: disable=no-member
        # consider making it diagonally dominant
        d_matrix = np.diagflat([1] * size)
        q_matrix = rand_matrix.T * d_matrix * rand_matrix
        # log used to reduce growth of values when size gets greater
        q_matrix = q_matrix / np.log(q_matrix.size)     # pylint: disable=no-member
        shapes = {'q': cls._get_quadratic_mask,
                  'r': cls._get_rectangle_mask,
                  'c': cls._curve_mask,
                  'a': cls._get_arrow_mask,
                  'n': cls._get_noise_mask}
        if re.match(r'[qrcan]', pattern):
            mask = np.zeros((size, size))
            for key in pattern:
                # mypy: begin ignore
                mask = np.logical_or(shapes[key](size), mask)    # pylint: disable=no-member
                # mypy: end ignore
            return np.multiply(q_matrix, mask)
        else:
            raise NotImplementedError

    @classmethod
    def get_random_test_matrix_csr(cls, size: int = 50, pattern: str = 'q') -> csr_matrix:
        """Return random test matrix in csr format."""
        return csr_matrix(cls.get_random_test_matrix(size=size, pattern=pattern))
