import numpy as np
import scipy

from collections import defaultdict, deque
from typing import List, Dict, Tuple

try:
    import numba
    from numba import jit
    from numba.types import bool_
    int16_ = numba.int16
except ImportError:
    def jit(*args, **kwargs):
        return lambda f: f
    bool_ = bool
    int16_ = np.uint16

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(*args, **kwargs):
        return args[0]

cache = True

################################################################################
# FUNCTIONS WITH ARRAY OPERATIONS                                              #
################################################################################
@jit(nopython=True, cache=cache)
def A_lessthan_B(A: np.array, B: np.array) -> bool_:
    """Compare two letters/measurement operators lexicographically.

    Parameters
    ----------
    A : np.array
        Measurement operator encoded as a 1D array.
    B : np.array
        Measurement operator encoded as a 1D array.

    Returns
    -------
    bool
        True if A is less than B.
    """
    for i in range(A.shape[0]):
        if A[i] != B[i]:
            return A[i] < B[i]
    return True


@jit(nopython=True)
def nb_first_index(array: np.ndarray, item: float) -> int:
    """Find the first index of an item in an array.

    Parameters
    ----------
    array : np.ndarray
         The array to search.
    item : float
        The item to find.

    Returns
    -------
    int
        The index where the first item is found.

    Examples
    --------
    >>> array = np.array([1, 2, 3, 4, 5, 6])
    >>> nb_first_index(array, 5)
    4
    """
    for idx, val in enumerate(array):
        if abs(val - item) < 1e-10:
            return idx


@jit(nopython=True)
def nb_unique(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Find the unique elements in an array without sorting and in order of
    appearance.

    Parameters
    ----------
    arr : np.ndarray
        The array to search.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        The unique values unsorted and their indices.

    Examples
    --------
    >>> nb_unique(np.array([1, 3, 3, 2, 2, 5, 4]))
    (array([1, 3, 2, 5, 4], dtype=int16), array([0, 1, 3, 5, 6], dtype=int16))
    """
    uniquevals = np.unique(arr)
    nr_uniquevals = uniquevals.shape[0]

    indices = np.zeros(nr_uniquevals).astype(int16_)
    for i in range(nr_uniquevals):
        indices[i] = nb_first_index(arr, uniquevals[i])
    indices.sort()

    uniquevals_unsorted = np.zeros(nr_uniquevals).astype(int16_)
    for i in range(nr_uniquevals):
        # Undo the sorting done by np.unique()
        uniquevals_unsorted[i] = arr[indices[i]]

    return uniquevals_unsorted, indices


################################################################################
# ABSTRACT OPERATIONS ON MONOMIALS                                             #
################################################################################
@jit(nopython=True, cache=cache)
def commuting(letter1: np.array, letter2: np.array) -> bool_:
    """Determine if two letters/operators commute. Currently this only takes
    into account commutations coming of inflation

    Parameters
    ----------
    letter1 : np.array
        Tuple of integers representing an operator.
    letter2 : np.array
        Tuple of integers representing an operator.

    Returns
    -------
    bool
        If they commute or not.

    Examples
    --------
    A^11_00 commutes with A^22_00
    >>> commuting(np.array([1, 1, 1, 0, 0]), np.array([1, 2, 2, 0, 0]))
    True

    A^11_00 does not commute with A^12_00 since they overlap on source 1.
    >>> commuting(np.array([1, 1, 1, 0, 0]), np.array([1, 1, 2, 0, 0]))
    False
    """
    if letter1[0] != letter2[0]:
        return True
    if np.array_equal(letter1[1:-1], letter2[1:-1]):
        return True

    inf1, inf2 = letter1[1:-2], letter2[1:-2]
    inf1, inf2 = inf1[np.nonzero(inf1)], inf2[np.nonzero(inf2)]

    # If at least one in inf1-inf2 is 0, then there is one source in common
    # and therefore the letters don't commute.
    # If all are 0, then this case is covered in the first conditional,
    # they commute regardless of the value of the output
    return True if np.all(inf1-inf2) else False


@jit(nopython=True, cache=cache)
def dot_mon(mon1: np.ndarray, mon2: np.ndarray) -> np.ndarray:
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


def dot_mon_commuting(mon1: np.ndarray, mon2: np.ndarray) -> np.ndarray:
    """A faster implementation of `dot_mon` that assumes that all
    operators commute. This implies we order everything lexicographically.

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
        return mon_lexsorted(mon2)
    if mon2.size <= 1:
        return mon_lexsorted(mon1)
    mon = np.concatenate((mon1, mon2))
    return mon_lexsorted(mon)


@jit(nopython=True, cache=cache)
def mon_equal_mon(mon1: np.ndarray, mon2: np.ndarray) -> bool:
    """Check if two monomials are equal. This is just a numba-ified version of
    numpy.array_equal.

    Parameters
    ----------
    mon1 : np.ndarray
        First monomial as a 2d array.
    mon2 : np.ndarray
        Second monomial as a 2d array.
    Returns
    -------
    bool
        True if the two monomials are equal, False otherwise.
    """
    return np.array_equal(mon1, mon2)


@jit(nopython=True, cache=cache)
def mon_is_zero(mon: np.ndarray) -> bool_:
    """Check if there is a product of two orthogonal projectors, returning True
    if so.

    Parameters
    ----------
    mon : np.ndarray
        Input monomial as 2d array.

    Returns
    -------
    bool
        True if the monomial is zero, False otherwise.
    """
    prev_row = mon[0]
    for i in range(1, mon.shape[0]):
        row = mon[i]
        if row[-1] != prev_row[-1] and np.array_equal(row[:-1], prev_row[:-1]):
            return True
        prev_row = row
    return False


@jit(nopython=True, cache=cache)
def mon_lessthan_mon(mon1: np.ndarray, mon2: np.ndarray) -> bool_:
    """Compare two monomials and returns True if mon1 < mon2 in lexicographic
    order.

    Parameters
    ----------
    mon1 : np.ndarray
        Input monomial as a 2d array.
    mon2 : np.ndarray
        Input monomial as a 2d array.

    Returns
    -------
    bool
        True if mon1 < mon2 in lexicographic order.
    """
    return A_lessthan_B(mon1.flatten(), mon2.flatten())


def mon_lexsorted(mon: np.ndarray) -> np.ndarray:
    """Return a monomial sorted lexicographically. The sorting keys are as
    follows. The first key is the parties, the second key is the inflation
    indices and the last keys are the input and output cardinalities. More
    informally, once we sorted all the parties together, within a party group we
    group all operators with the same inflation-copy index for the first source
    together, and within this group we group all operators with the same
    inflation-copy index for the second source together and so on until the last
    keys.

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
def mon_sorted_by_parties(mon: np.ndarray) -> np.ndarray:
    """Sort the monomial by parties, i.e., sort by the first column in
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


################################################################################
# OPERATIONS ON MONOMIALS RELATED TO INFLATION                                 #
################################################################################
@jit(nopython=True)
def apply_source_swap_monomial(monomial: np.ndarray,
                               source: int,
                               copy1: int,
                               copy2: int
                               ) -> np.ndarray:
    """Applies a swap of two sources to a monomial.

    Parameters
    ----------
    monomial : np.ndarray
        2d array representation of a monomial.
    source : int
         Integer in values [0, ..., nr_sources]
    copy1 : int
        Represents the copy of the source that swaps with copy2
    copy2 : int
        Represents the copy of the source that swaps with copy1

    Returns
    -------
    np.ndarray
         The new monomial with swapped sources.

    Examples
    --------
    >>> monomial = np.array([[1, 2, 3, 0, 0, 0]])
    >>> apply_source_swap_monomial(np.array([[1, 0, 2, 1, 0, 0],
                                             [2, 1, 3, 0, 0, 0]]),
                                             1,  # source
                                             2,  # copy1
                                             3)  # copy2
    array([[1, 0, 3, 1, 0, 0],
           [2, 1, 2, 0, 0, 0]])
    """
    new_factors = monomial.copy()
    for i in range(new_factors.shape[0]):
        copy = new_factors[i, 1 + source]
        if copy > 0:
            if copy == copy1:
                new_factors[i, 1 + source] = copy2
            elif copy == copy2:
                new_factors[i, 1 + source] = copy1
    return new_factors


def factorize_monomial(monomial: np.ndarray) -> np.ndarray:
    """Split a moment/expectation value into products of moments according to
    the support of the operators within the moment.

    The moment is encoded as a 2d array where each row is an operator.
    If monomial=A*B*C*B then row 1 is A, row 2 is B, row 3 is C and row 4 is B.
    In each row, the columns encode the following information:

    First column:       The party index, *starting from 1*.
                        (1 for A, 2 for B, etc.)
    Last two columns:   The input x, starting from zero and then the
                        output a, starting from zero.
    In between:         This encodes the support of the operator. There
                        are as many columns as sources/quantum states.
                        Column j represents source j-1 (-1 because the 1st
                        col is the party). If the value is 0, then this
                        operator does not measure this source. If the value
                        is for e.g. 2, then this operator is acting on
                        copy 2 of source j-1.

    The output is a list of lists, where each list represents another
    monomial s.t. their product is equal to the original monomial.

    Parameters
    ----------
    monomial : np.ndarray
        Monomial encoded as a 2d array where each row is an operator.

    Returns
    -------
    np.ndarray
        A list of lists, where each list represents the monomial factors.

    Examples
    --------
    >>> monomial = np.array([[1, 0, 1, 1, 0, 0],
                             [2, 1, 0, 2, 0, 0],
                             [1, 0, 3, 3, 0, 0],
                             [3, 3, 5, 0, 0, 0],
                             [3, 1, 4, 0, 0, 0],
                             [3, 6, 6, 0, 0, 0],
                             [3, 4, 5, 0, 0, 0]])
    >>> factorised = factorize_monomial(monomial)
    [array([[1, 0, 1, 1, 0, 0]]),
     array([[1, 0, 3, 3, 0, 0]]),
     array([[2, 1, 0, 2, 0, 0],
            [3, 1, 4, 0, 0, 0]]),
     array([[3, 3, 5, 0, 0, 0],
            [3, 4, 5, 0, 0, 0]]),
     array([[3, 6, 6, 0, 0, 0]])]
    """
    monomial = np.array(monomial, dtype=np.ubyte)
    components_indices = np.zeros((len(monomial), 2), dtype=np.ubyte)
    # Add labels to see if the components have been used
    components_indices[:, 0] = np.arange(0, len(monomial), 1)

    inflation_indices = monomial[:, 1:-2]
    disconnected_components = []

    idx = 0
    while idx < len(monomial):
        component = []
        if components_indices[idx, 1] == 0:
            component.append(idx)
            components_indices[idx, 1] = 1
            jdx = 0
            # Iterate over all components that are connected
            while jdx < len(component):
                nonzero_sources = np.nonzero(
                    inflation_indices[component[jdx]])[0]
                for source in nonzero_sources:
                    overlapping = inflation_indices[:,
                           source] == inflation_indices[component[jdx], source]
                    # Add the components that overlap to the lookup list
                    component += components_indices[overlapping &
                                (components_indices[:, 1] == 0)][:, 0].tolist()
                    # Specify that the components that overlap have been used
                    components_indices[overlapping, 1] = 1
                jdx += 1
        if len(component) > 0:
            disconnected_components.append(component)
        idx += 1

    disconnected_components = [
        monomial[np.array(component)] for component in disconnected_components]

    # Order each factor as determined by the input monomial. We store the
    # the positions in the monomial so we can read it off afterwards.
    # Method taken from
    # https://stackoverflow.com/questions/64944815/
    # sort-a-list-with-duplicates-based-on-another-
    # list-with-same-items-but-different
    monomial = monomial.tolist()
    indexes = defaultdict(deque)
    for i, x in enumerate(monomial):
        indexes[tuple(x)].append(i)

    for idx, component in enumerate(disconnected_components):
        ids = sorted([indexes[tuple(x)].popleft() for x in component.tolist()])
        ordered_component = [monomial[id] for id in ids]
        disconnected_components[idx] = np.array(ordered_component)

    # Order independent factors canonically
    disconnected_components = sorted(disconnected_components,
                                     key=lambda x: x[0].tolist())
    return disconnected_components


@jit(nopython=True, cache=cache)
def nb_apply_substitutions(mon_in: np.ndarray) -> np.ndarray:
    """Apply substitutions to a monomial. Currently it only supports
    commutations arising from operators having completely different support.

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
                mon[i-1, :], mon[i, :] = mon[i, :].copy(), mon[i-1, :].copy()
    return mon


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
    to_keep = np.array([1]*mon.shape[0], dtype=bool_)
    prev_row = mon[0]
    for i in range(1, mon.shape[0]):
        row = mon[i]
        if np.array_equal(row, prev_row):
            to_keep[i] = False
        prev_row = row
    return mon[to_keep]


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
    # The two-body commutation rules are not enough in some occasions when the
    # monomial can be factorized. An example is (all indices are inflation
    # indices) A13A33A22 and A22A13A33. The solution below is to decompose
    # in disconnected components, order them canonically, and recombine them.
    mon = np.concatenate(factorize_monomial(remove_projector_squares(mon)))
    mon = np.vstack(sorted(mon, key=lambda x: x[0]))
    return mon


def to_name(monomial: np.ndarray, names: List[str]) -> str:
    """Convert the 2d array representation of a monomial to a string.

    Parameters
    ----------
    monomial : np.ndarray
        Monomial in matrix format.
    names : List[str]
        List of party names.

    Returns
    -------
    str
        String representation of the monomial.

    Examples
    --------
    >>> to_name([[1 1,0,3], [4,1,2,6], [2,3,3,4]], ['a','bb','x','Z'])
    'a_1_0_3*Z_1_2_6*bb_3_3_4'
    """
    if mon_equal_mon(monomial, np.array([0])):
        return '1'

    # It is faster to convert to list of lists than to loop through numpy arrays
    monomial = monomial.tolist()
    return '*'.join(['_'.join([names[letter[0]-1]]+[str(i) for i in letter[1:]])
                     for letter in monomial])


################################################################################
# OPERATIONS ON MOMENT MATRICES                                                #
################################################################################
def calculate_momentmatrix(cols: List,
                           names: np.ndarray,
                           commuting: bool=False,
                           verbose: int=0
                           ) -> Tuple[np.ndarray, Dict]:
    """Calculate the moment matrix. The function takes as input the generating
    set {mon_i}_i encoded as a list of monomials. Each monomial is a matrix
    where each row is an operator and the columns specify the operator
    labels/indices. The moment matrix is the inner product between all possible
    pairs of elements from the generating set. The program outputs the moment
    matrix as a 2d array. Entry (i,j) of the moment matrix stores the index of
    the monomial that is the result of the dot product mon_i^\dagger * mon_j
    after applying the substitutions. The program returns the moment matrix and
    the dictionary mapping each monomial in string representation to its integer
    representation.

    Parameters
    ----------
    cols : List
        List of np.ndarray representing the generating set.
    names : np.ndarray
        The string names of each party.
    commuting: bool, optional
        Whether the variables in the problem commute or not, by default False.
    verbose : int, optional
        How much information to print, by default 0.

    Returns
    -------
    Tuple[np.ndarray, Dict]
        The moment matrix, where each entry (i,j) stores the integer
        representation of a monomial. The Dict is a mapping from string
        representation to integer representation.
    """
    # Choose dot_mon depending on commutation properties
    if commuting:
        dot_fn = dot_mon_commuting
    else:
        dot_fn = dot_mon
    nrcols = len(cols)
    vardic = {}
    momentmatrix = scipy.sparse.lil_matrix((nrcols, nrcols), dtype=np.uint32)
    if np.array_equal(cols[0], np.array([0])):
        cols[0] = np.array([cols[0]])  # Some functions need [[0]], not [0]
    varidx = 1  # Start from 1 because 0 is reserved for the constant 0
    for i in tqdm(range(nrcols),
                  disable=not verbose,
                  desc="Calculating moment matrix    "):
        for j in range(i, nrcols):
            mon1, mon2 = cols[i], cols[j]
            if mon1.size <= 1 and mon2.size <= 1:
                name = ' '
                vardic[name] = varidx
                momentmatrix[i, j] = vardic[name]
                varidx += 1
            else:
                mon = dot_fn(mon1, mon2)
                if not mon_is_zero(mon):
                    mon  = to_canonical(mon)
                    name = to_name(mon, names)
                    if name not in vardic:
                        mon_rev  = to_canonical(dot_mon(mon2, mon1))
                        rev_name = to_name(mon_rev, names)
                        if rev_name not in vardic:
                            vardic[name] = varidx
                            if not np.array_equal(mon, mon_rev):
                                vardic[rev_name] = varidx
                            varidx += 1
                        else:
                            vardic[name] = vardic[rev_name]
                        momentmatrix[i, j] = vardic[name]
                    else:
                        momentmatrix[i, j] = vardic[name]
                    if i != j:
                        # Note: We assume a REAL moment matrix
                        momentmatrix[j, i] = momentmatrix[i, j]
    return momentmatrix, vardic
