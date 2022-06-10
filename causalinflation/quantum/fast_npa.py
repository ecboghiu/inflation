import numpy as np
import scipy

from tqdm import tqdm

from typing import List, Dict, Tuple

try:
    from numba import jit
    from numba.types import bool_
except ImportError:
    def jit(*args, **kwargs):
        return lambda f: f
    bool_ = bool

cache = True


@jit(nopython=True, cache=cache)
def reverse_mon(mon: np.ndarray) -> np.ndarray:
    """Output the monomial reversed, which means reverse the row of the 2d
    matrix representing the monomial. This represents the complex conjugate
    of the monomial, but we assume they are Hermitian.

    Parameters
    ----------
    mon : np.ndarray
        Input monomial as 2d array.

    Returns
    -------
    np.ndarray
        Reversed monomial.

    Examples
    --------
    >>> reverse_mon(np.array([[1,...],[4,...],[3,...]]))
    np.array([[3,...],[4,...],[1,...]])
    """

    return mon[np.arange(mon.shape[0])[::-1]]


@jit(nopython=True, cache=cache)
def mon_sorted_by_parties(mon: np.ndarray) -> np.ndarray:
    """Sort by parties the monomial, i.e., sort by the first column in
    the 2d representation of the monomial.

    Parameters
    ----------
    mon : np.ndarray
        Input monomial as 2d array.

    Returns
    -------
    np.ndarray
        Sorted monomial.

    Examples
    --------
    >>> mon_sorted_by_parties(np.array([[3,...],[1,...],[4,...]]))
    np.array([[1,...],[3,...],[4,...]])

    """

    return mon[np.argsort(mon[:, 0], kind='mergesort')]


@jit(nopython=True, cache=cache)
def dot_mon(mon1: np.ndarray,
            mon2: np.ndarray
            ) -> np.ndarray:
    """Returns ((mon1)^dagger)*mon2.

    For hermitian operators this is the same as reversed(mon1)*mon2.
    Since all parties commute, the output is ordered by parties. We do
    not assume any other commutation rules.


    Parameters
    ----------
    mon1 : np.ndarray
        Monomial as a numpy array.
    mon2 : np.ndarray
        Monomial as a numpy array.

    Returns
    -------
    np.ndarray
        Output monomial ordered by parties.

    Examples
    --------
    >>> mon1 = np.array([[1,2,3],[4,5,6]])
    >>> mon2 = np.array([[7,8,9]])
    >>> dot_mon(mon1, mon2)
    np.array([[4,5,6],[1,2,3],[7,8,9]])

    """

    if mon1.size <= 1:
        return mon2
    if mon2.size <= 1:
        return mon_sorted_by_parties(reverse_mon(mon1))
    return mon_sorted_by_parties(np.concatenate((reverse_mon(mon1), mon2)))

#@jit(nopython=True, cache=cache)
def dot_mon_commuting(mon1: np.ndarray,
                      mon2: np.ndarray
                      ) -> np.ndarray:
    """A faster implementation of ```dot_mon``` that assumes that all
    operators commute. This implies we order everything lexiographically.

    Parameters
    ----------
    mon1 : np.ndarray
        Monomial as a numpy array.
    mon2 : np.ndarray
        Monomial as a numpy array.

    Returns
    -------
    np.ndarray
        Returns (mon1)^\dagger*mon2 with the assumption that
        everything commutes with everything.
    """

    if mon1.size <= 1:
        return mon2
    if mon2.size <= 1:
        return mon_sorted_by_parties(reverse_mon(mon1))
    # So it seems there is no implementation of lexsort in numba, so for now
    # we don't use a precompiled function
    # what we can do is use 'sorted' with the string representation of
    # the monomials which has a lexicographic ordering or implement a
    # lexicographic ordering in numba
    # numba issue: https://github.com/numba/numba/issues/3689
    mon = np.concatenate((reverse_mon(mon1), mon2))
    return mon_lexsorted(mon)


def mon_lexsorted(mon: np.ndarray
                 ) -> np.ndarray:
    """Return a monomial sorted lexicographically.

    The sorting keys are as follows. The first key is the parties, the second
    key is the inflation indices and the last keys are the input and output
    cardinalities. More informally, once we sorted all the parties together,
    within a party group we group all operators with the same inflation-copy
    index for the first source together, and within this group we group all
    operators with the same inflation-copy index for the second source
    together and so on until the last keys.

    Parameters
    ----------
    mon : np.ndarray
        Monomial as a numpy array.

    Returns
    -------
    np.ndarray
        Sorted monomial.
    """

    return mon[np.lexsort(np.rot90(mon))]


@jit(nopython=True, cache=cache)
def remove_projector_squares(mon: np.ndarray) -> np.ndarray:
    """Simplify the monomial by removing the squares. This is because we
    assume projectors, P^2=P.

    Parameters
    ----------
    mon : np.ndarray
        Input monomial as 2d array.

    Returns
    -------
    np.ndarray
        Simplified monomial.
    """

    to_keep = np.array([1]*mon.shape[0], dtype=bool_)  # bool_
    prev_row = mon[0]
    for i in range(1, mon.shape[0]):
        row = mon[i]
        if np.array_equal(row, prev_row):
            to_keep[i] = False
        prev_row = row
    return mon[to_keep]


@jit(nopython=True, cache=cache)
def mon_is_zero(mon: np.ndarray
                ) -> bool_:
    """Function which checks if there is a product of two orthogonal projectors,
    and returns True if so.

    _extended_summary_

    Parameters
    ----------
    mon : np.ndarray
        Input monomial as 2d array.

    Returns
    -------
    bool_
        True if the monomial is zero, False otherwise.
    """

    prev_row = mon[0]
    for i in range(1, mon.shape[0]):
        row = mon[i]
        if row[-1] != prev_row[-1] and np.array_equal(row[:-1], prev_row[:-1]):
            return True
        prev_row = row
    return False

#@jit(nopython=True, cache=cache)
def to_name(monomial_numbers: np.ndarray,
            parties_names: np.ndarray
            ) -> str:
    """Converts the 2d array representation of a monoial to a string.

    _extended_summary_

    Parameters
    ----------
    monomial_numbers : np.ndarray
        Input monomial as 2d array.
    parties_names : np.ndarray
        Array of strings representing the parties.

    Returns
    -------
    str
        String representation of the monomial.

    Examples
    --------
    >>> to_name([[1 1,0,3], [4,1,2,6], [2,3,3,4]], ['a','bb','x','Z'])
    'a_1_0_3*Z_1_2_6*bb_3_3_4'
    """

    # Numba version is 2x slower than without!! Probably not optimized for
    # strings. But we need it to be able to call to_name within a numba
    # function
    if type(monomial_numbers) != np.ndarray:
        monomial_numbers = np.array(monomial_numbers)
    if type(parties_names) != np.ndarray:
        parties_names = np.array(parties_names)

     #np.zeros(monomial_numbers.shape[0], dtype=numba.types.unicode_type)
    components = ['']*monomial_numbers.shape[0]
    for i, monomial in enumerate(monomial_numbers):
        arr1 = [parties_names[monomial[0] - 1]]
        arr2 = [str(i) for i in monomial[1:]]
        components[i] = '_'.join(arr1 + arr2)
    return '*'.join(components)


def pad_with_zeros(cols: List[List]
                    ) -> List[np.ndarray]:
    """Pad the columns with zeros.

    Parameters
    ----------
    cols : List[List]
        Columns as lists of lists.

    Returns
    -------
    List[np.ndarray]
        Returns the inputs but with rows of zeros padded to the same length.
    """
    max_prod_len = max([len(col) for col in cols])
    mon_len = len(cols[-1][0])
    cols_padded = np.zeros((len(cols), max_prod_len, mon_len), dtype=np.uint8)
    for i in range(1, len(cols)):
        cols_padded[i, :, :] = np.pad(cols[i],
                                      ((0, max_prod_len-len(cols[i])), (0, 0)),
                                      'constant', constant_values=(0, 0))
    return cols_padded


@jit(nopython=True, cache=cache)
def remove_zeros(mon: np.ndarray
                 ) -> np.ndarray:
    """If monomial has rows of all zeros, remove them.

    Parameters
    ----------
    mon : np.ndarray
        Monomial as a matrix where each row is an operator.

    Returns
    -------
    np.ndarray
        The monomial without rows of all zeros.
    """

    return mon[mon[:, 0] != 0]


@jit(nopython=True, cache=cache)
def commuting(letter1: np.array,
              letter2: np.array
              ) -> bool_:
    """Determine if two letters/operators commute.

    TODO accept arbitrary commutation rules.
    Currently this only takes into accounts commutation coming of inflation

    Parameters
    ----------
    letter1 : np.array
        Tuple of integers representing an operator.
    letter2 : np.array
        Tuple of integers representing an operator.

    Returns
    -------
    bool_
        If they commute or not.

    Examples
    --------
    A^11_00 commutes with A^22_00
    >>> commuting(np.array([1, 1, 1, 0, 0]), np.array([1, 2, 2, 0, 0]))
    True

    A^11_00 does not commute with A^12_00 as they overlap on source 1.
    >>> commuting(np.array([1, 1, 1, 0, 0]), np.array([1, 1, 2, 0, 0]))
    False
    """

    if letter1[0] != letter2[0]:
        # if exactly same support and input, but then different outputs,
        #  they commute (but the product should be 0 anyway!! so the last
        #  thing shouldnt be relevant)
        return True
    if np.array_equal(letter1[1:-1], letter2[1:-1]):
        return True

    inf1, inf2 = letter1[1:-2], letter2[1:-2]
    inf1, inf2 = inf1[np.nonzero(inf1)], inf2[np.nonzero(inf2)]

    # if at least one in inf1-inf2 is 0, then there is one source in common
    # therefore they don't commute.
    # If all are 0, then this case is covered in the first conditional,
    # they commute regardless of the value of the output
    return True if np.all(inf1-inf2) else False


@jit(nopython=True, cache=cache)
def A_lessthan_B(A: np.array, B: np.array) -> bool_:
    """Compares two letters/measurement operators lexicographically.

    Parameters
    ----------
    A : np.array
        Measurement operator encoded as a 1D array.
    B : np.array
        Measurement operator encoded as a 1D array.

    Returns
    -------
    bool_
        True if A is less than B.
    """
    # if A.size == 0:
    #     return True
    # if B.size == 0:
    #     return False
    # if A[0] < B[0]:
    #     return True
    # if A[0] > B[0]:
    #     return False
    # return A_lessthan_B(A[1:], B[1:])
    # Above is more 'elegant' but im not sure about how efficient recursive
    # functions are
    for i in range(A.shape[0]):
        if A[i] != B[i]:
            return A[i] < B[i]
    return True


@jit(nopython=True, cache=cache)
def mon_lessthan_mon(mon1: np.ndarray,
                     mon2: np.ndarray
                     ) -> bool_:
    """Compares two monomials and returns True if mon1 < mon2 in lexicographic
    order.

    It flattens then 2D array representing the monomial to and uses
    the function A_lessthan_B.

    Parameters
    ----------
    mon1 : np.ndarray
        Input monomial as a 2d array.
    mon2 : np.ndarray
        Input monomial as a 2d array.

    Returns
    -------
    bool_
        True if mon1 < mon2 in lexicographic order.
    """

    return A_lessthan_B(mon1.flatten(), mon2.flatten())


@jit(nopython=True, cache=cache)
def nb_apply_substitutions(mon_in: np.ndarray) -> np.ndarray:
    """Apply substitutions to a monomial.

    Currently it only supports commutations arising from operators having
    completely different support. It goes in a loop applying the substitutions
    until it reaches a fixed point, if it finds two letters that commute and
    are not in lexicographic ordering. This function does a single loop from
    the first row to the last applying all substitutions along the way and
    then it returns.

    Parameters
    ----------
    mon_in : np.ndarray
        Input monomial as a 2d array.

    Returns
    -------
    np.ndarray
        Simplified input monomial.
    """

    if mon_in.shape[0] == 1:
        return mon_in
    mon = mon_in.copy()
    for i in range(1, mon.shape[0]):
        if not A_lessthan_B(mon[i-1], mon[i]):
            if commuting(mon[i-1], mon[i]):
                # More elegant but doesn't work with numba
                #mon[[i-1,i],:] = mon[[i,i-1],:]
                mon[i-1, :], mon[i, :] = mon[i, :].copy(), mon[i-1, :].copy()
                # ?? Is it best to stop after one commutation, or just
                #  keep going until the end of the array?
                #return mon
    return mon


@jit(nopython=True, cache=cache)
def to_canonical(mon: np.ndarray) -> np.ndarray:
    """Apply substitutions to a monomial until it stops changing.

    Parameters
    ----------
    mon : np.ndarray
        Input monomial as a 2d array.

    Returns
    -------
    np.ndarray
        Monomial in canonical form w.r.t some lexicographic
        ordering.
    """

    prev = mon
    while True:
        mon = nb_apply_substitutions(mon)
        if np.array_equal(mon, prev):
            break
        prev = mon
    return remove_projector_squares(mon)


def calculate_momentmatrix(cols: List,
                           names: np.ndarray,
                           verbose: int = 0
                           ) -> Tuple[np.ndarray, Dict]:
    """Calculate the moment matrix.

    Takes as input the generating set {mon_i}_i encoded as a list of monomials.
    Each monomial is a matrix where each row is an operator and the columns
    specify the operator labels/indices. The moment matrix is the inner product
    between all possible pairs of elements from the generating set. The program
    outputs the moment matrix as a 2d array. Entry (i,j) of the moment matrix
    stores the index of the monomial that is the result of the dot product
    mon_i^\dagger * mon_j after applying the substitutions. The program returns
    the moment matrix and the dictionary mapping each monomial in string
    representation to its integer representation.

    Parameters
    ----------
    cols : List
        List of np.ndarray representing the generating set.
    names : np.ndarray
        The string names of each party.
    verbose : int, optional
        _description_, by default 0

    Returns
    -------
    Tuple[np.ndarray, Dict]
        The moment matrix, where each entry (i,j) stores the
        integer representation of a monomial. The Dict is a
        mapping from string representation of monomial to integer
        representation.
    """

    nrcols = len(cols)
    vardic = {}
    # Emi: so np.array([-1],dtype=np.uint16) evaluates to 65535, can we ensure
    # we will never have more than 65535 unique variables??
    # I'm not so convinced, so I go with int32 (~4 billion)
    momentmatrix = scipy.sparse.lil_matrix((nrcols, nrcols), dtype=np.uint32)
    if np.array_equal(cols[0], np.array([0])):
        cols[0] = np.array([cols[0]])  # Some function needs [[0]] not []
    iteration = 0
    varidx = 1  # We start from 1 because 0 is reserved for 0
    for i in tqdm(range(nrcols),
                  disable=not verbose,
                  desc="Calculating moment matrix    "):
        for j in range(i, nrcols):
            mon1, mon2 = cols[i], cols[j]
            if mon1.size <= 1 and mon2.size <= 1:
                #print(mon1,mon2)
                name = ' '
                vardic[name] = varidx
                momentmatrix[i, j] = vardic[name]
                varidx += 1
            else:
                mon = dot_mon(mon1, mon2)
                if mon_is_zero(mon):
                    # If sparse, we don't need this, but for readibility...
                    momentmatrix[i, j] = 0
                else:
                    mon = to_canonical(mon)
                    name = to_name(mon, names)

                    if name not in vardic:
                        vardic[name] = varidx
                        mon_rev = to_canonical(dot_mon(mon2, mon1))
                        # equiv to name2 != name, but faster?
                        if not np.array_equal(mon, mon_rev):
                            name2 = to_name(mon_rev, names)
                            if name2 not in vardic:
                                vardic[name2] = varidx
                        momentmatrix[i, j] = vardic[name]
                        varidx += 1
                    else:
                        momentmatrix[i, j] = vardic[name]
                    if i != j:
                        # NOTE: Assuming a REAL moment matrix!!
                        momentmatrix[j, i] = momentmatrix[i, j]
            iteration += 1
    return momentmatrix, vardic


def calculate_momentmatrix_commuting(cols: np.ndarray,
                                     names: np.ndarray,
                                     verbose: int = 0
                                     ) -> np.ndarray:
    """See description of 'calculate_momentmatrix'. The same, but we further
    assume everything commutes with everything.

    Parameters
    ----------
    cols : np.ndarray
        List of np.ndarray representing the generating set.
    names : np.ndarray
        The string names of each party.
    verbose : int, optional
        How descriptive the prints are, by default 0.

    Returns
    -------
    Tuple[np.ndarray, Dict]
        The moment matrix, where each entry (i,j) stores the
        integer representation of a monomial. The Dict is a a
        mapping from string representation of monomial to integer
        representation.
    """
    nrcols = len(cols)

    vardic = {}
    momentmatrix = scipy.sparse.lil_matrix((nrcols, nrcols), dtype=np.uint32)
    if np.array_equal(cols[0], np.array([0])):
        # Some function doesnt like [0] and needs [[0]]
        cols[0] = np.array([cols[0]])
    iteration = 0
    varidx = 1  # We start from 1 because 0 is reserved for 0
    for i in tqdm(range(nrcols), disable=not verbose,
                  desc="Calculating moment matrix"):
        for j in range(i, nrcols):
            mon1, mon2 = cols[i], cols[j]
            if mon1.size <= 1 and mon2.size <= 1:
                name = ' '
                vardic[name] = varidx
                momentmatrix[i, j] = vardic[name]
                varidx += 1
            else:
                mon = dot_mon_commuting(mon1, mon2)
                if mon_is_zero(mon):
                    # If sparse, we don't need this, but for readibility...
                    momentmatrix[i, j] = 0
                else:
                    name = to_name(remove_projector_squares(mon), names)
                    if name not in vardic:
                        vardic[name] = varidx
                        momentmatrix[i, j] = vardic[name]
                        varidx += 1
                    else:
                        momentmatrix[i, j] = vardic[name]
                    if i != j:
                        # Assuming a REAL moment matrix!!
                        momentmatrix[j, i] = momentmatrix[i, j]
            iteration += 1
    return momentmatrix, vardic


def export_to_file_numba(momentmatrix, monomials_list, filename_label,
                         verbose: int = 0) -> None:
    """
     TODO WARNING: BUGGY!!
    """
    block_idx = 1  # TODO accept any block structure
    filename_momentmatrix = 'momentmatrix+' + filename_label + '.dat-s'
    filename_monomials = 'monomials' + filename_label + '.txt'
    with open(filename_momentmatrix, 'w') as f:
        nrvars = np.max(monomials_list[:, 0].astype(int))
        f.write("file "+filename_momentmatrix +
                " generated by UnifiedInflation\n")
        f.write(str(nrvars) + " = number of variables\n")
        f.write("1 = number of blocs\n")
        f.write("(" + str(momentmatrix.shape[0]) + ") = BlocStructure\n")
        f.write("{")
        for i in range(nrvars):
            f.write("0, ")  # This is the objective function!
        f.write("}\n")

        # Matrix entries are given in the following format:
        # n b i j v
        # n = index of variable (0 stands for the rhs, i.e. the constant terms)
        # b = index of block
        # i j = position in matrix
        # v = coefficient or value
        # See http://www.opt.tu-darmstadt.de/scipsdp/downloads/data_format.txt
        for i in tqdm(range(momentmatrix.shape[0]),
                      disable=not verbose,
                      desc='Writing momentmatrix to file '):
            for j in range(i, momentmatrix.shape[1]):
                if momentmatrix[i, j] != 0:
                    # TODO accept any coefficient, or change the format
                    # in which we enconde semiknowns
                    f.write(str(momentmatrix[i, j]-1) + "\t" + str(block_idx) +
                            "\t" + str(i) + "\t" + str(j) + "\t" + "1\n")

    with open(filename_monomials, 'w') as f:
        for row in tqdm(monomials_list,
                        disable=not verbose,
                        desc='Writing monomials to file    '):
            f.write(" ".join(row) + "\n")
