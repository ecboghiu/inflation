import numba
import numpy as np

@numba.njit
def nb_ndindex(dims: np.ndarray):
    """Function to replace np.ndindex for numba compatibility

    Parameters
    ----------
    dims : np.ndarray
        Array of dimensions

    Returns
    -------
    np.ndarray
        2D array where each row is a tuple of indices
    """
    dims = dims[::-1]  # Reverse dims to match np.ndindex
    nr_rows = 1
    for dim in dims:
        nr_rows *= dim
    indices = np.zeros((int(nr_rows), len(dims)), dtype=np.int32)
    step_size = 1
    for i, dim in enumerate(dims):
        nr_steps = int(nr_rows/step_size)
        for j in range(nr_steps):
            indices[j*step_size:(j + 1)*step_size, -1 - i] = j % dim
        step_size *= dim
    return indices

@numba.jit(nopython=True)
def nb_expand_moment_normalisation(moment, outcome_cardinalities):
    """Function to expand a moment encoded as a 2D array where some of
    the operators correspond to the last outcome, and expand it
    into a linear combination of moments that do not involve any operators
    with the last outcome.


    Parameters
    ----------
    moment : np.ndarray
        Moment encoded as a 2D array where rows are operators, the order
        of the rows encodes the order of the product, and columns encode 
        different properties of different operators.
    outcome_cardinalities : np.ndarray
        Array of integers specifying the outcome cardinality of each party.

    Returns
    -------
    Tuple
        Tuple(Tuple(int), Tuple(np.ndarray))
        
        It returns a tuple with two elements, both tuples, where the i-th 
        element of the first tuple corresponds to the coefficient of the 
        array in the i-th place in the second tuple in the linear combination.
    """
    # Identify the operators with the last outcome
    ops_w_last_outcome = np.zeros(moment.shape[0], dtype=np.int32)
    for k, op in enumerate(moment):
        if op[-1] == outcome_cardinalities[op[0] - 1] - 1:
            ops_w_last_outcome[k] = 1
    # Also useful to store the row index k of the operators with the last outcome
    nof_ops_w_last_outcome = np.sum(ops_w_last_outcome)
    op_idx_w_last_outcome = np.zeros(nof_ops_w_last_outcome, dtype=np.int32)
    idx = 0
    for k, op in enumerate(moment):
        if op[-1] == outcome_cardinalities[op[0] - 1] - 1:
            op_idx_w_last_outcome[idx] = k
            idx += 1
    
    # Count the total number of terms in the expansion to allocate the arrays.
    # The counting is done in a stupid way, but it is fast enough.
    nof_terms = 0
    for if_ops_present in nb_ndindex(2*np.ones(nof_ops_w_last_outcome)):  
        # if_ops_present contains all possible combinations of present/absent
        # as we are substitutin the last outcome operators with
        # (1-\sum_{a\a_last} Aa)
        aux_counting = 1  # Starts at 1, in case where this is a term with without operators present
        for idx__, present in enumerate(if_ops_present):
            if present:
                # If an operator is present, there are len({a\a_last}) terms
                # in the expansion, this corresponds to \sum_{a\a_last} Aa
                aux_counting *= outcome_cardinalities[moment[op_idx_w_last_outcome[idx__], 0] - 1] - 1
        nof_terms += aux_counting
    nof_terms = int(nof_terms)
    
    signs = np.zeros(nof_terms, dtype=np.int32)
    moments = np.zeros((nof_terms*moment.shape[0], moment.shape[1]), 
                       dtype=np.int32)

    moments_idx = 0
    for if_ops_present in nb_ndindex(2*np.ones(nof_ops_w_last_outcome, dtype=np.int32)):
        # Example:
        # A0 B2 C0 D2 F0 with cardinality 3 has B2 and D2 as operators with last
        # outcome. Then, if_ops_present = [0, 0] means that B2 and D2 are not
        # present, if_ops_present = [0, 1] means that B2 is not present and D2
        # is present, etc.
        nof_present_ops = np.sum(if_ops_present)
        if np.sum(if_ops_present) == 0:
            moments[moments_idx*moment.shape[0]:(moments_idx+1)*moment.shape[0], :] = moment
            for _i in op_idx_w_last_outcome:
                moments[moments_idx*moment.shape[0] + _i, 0] = -1  # Mark to delete
            signs[moments_idx] = (-1)**np.sum(if_ops_present)
            moments_idx += 1
        else:
            moments_template = moment.copy().astype(np.int32)
            # If there is some operator we are expanding, write down their 
            # cardinality
            cards_of_present_ops = np.zeros(nof_present_ops, dtype=np.int32)
            idx_of_present_ops = np.zeros(nof_present_ops, dtype=np.int32)
            _i = 0
            for __i__, op_present in enumerate(if_ops_present):
                op_idx = op_idx_w_last_outcome[__i__]
                if op_present:
                    # Write down the row index of the operator we are expanding
                    # and how many terms there are in the expansion
                    idx_of_present_ops[_i] = op_idx
                    cards_of_present_ops[_i] = outcome_cardinalities[moment[op_idx, 0] - 1]
                    _i += 1
                else:
                    # Mark the operators we are not expanding to delete them at
                    # the end
                    moments_template[op_idx, 0] = -1

            for outcome_tuple in nb_ndindex(cards_of_present_ops - 1):
                moments[moments_idx*moment.shape[0]:(moments_idx+1)*moment.shape[0], :] = moments_template
                for op_idx, outcome in zip(idx_of_present_ops, outcome_tuple):
                    moments[moments_idx*moment.shape[0] + op_idx, -1] = outcome
                signs[moments_idx] = (-1)**np.sum(if_ops_present)
                moments_idx += 1

    # Remove rows starting with '-1' as they corresponds to missing operators
    # print(moments)
    terms = []
    for i in range(nof_terms):
        aux_moment = moments[i*moment.shape[0]:(i + 1)*moment.shape[0]]
        aux_moment = aux_moment[aux_moment[:, 0] > 0]
        terms.append(aux_moment.astype(np.uint8))
 
    return signs, terms

if __name__ == '__main__':
    case = "A1 = 1 - A0"
    signs, terms = nb_expand_moment_normalisation(np.array([[1,0,0,1]]),
                                                np.array([2]))
    assert np.allclose(signs, np.array([1, -1])), f"Signs are wrong in {case} expansion"
    assert np.allclose(terms[0], np.empty((0, 4))), f"Arrays are wrong in {case} expansion"
    assert np.allclose(terms[1], np.array([[1,0,0,0]])), f"Arrays are wrong in {case} expansion"

    case = "A1B0 = B0 - A0B0"
    signs, terms = nb_expand_moment_normalisation(np.array([[1,0,0,1],
                                                            [2,0,0,0]]),
                                                np.array([2, 2]))
    assert np.allclose(signs, np.array([1, -1])), f"Signs are wrong in {case} expansion"
    assert np.allclose(terms[0], np.array([[2,0,0,0]])), f"Arrays are wrong in {case} expansion"
    assert np.allclose(terms[1], np.array([[1,0,0,0],
                                        [2,0,0,0]])), f"Arrays are wrong in {case} expansion"

    case = "A2 = 1 - A1 - A0"
    signs, terms = nb_expand_moment_normalisation(np.array([[1,0,0,2]]),
                                                np.array([3]))
    assert np.allclose(signs, np.array([1, -1, -1])), f"Signs are wrong in {case} expansion"
    assert np.allclose(terms[0], np.empty((0, 4))), f"Arrays are wrong in {case} expansion"
    assert np.allclose(terms[1], np.array([[1,0,0,0]])), f"Arrays are wrong in {case} expansion"
    assert np.allclose(terms[2], np.array([[1,0,0,1]])), f"Arrays are wrong in {case} expansion"

    case = "A2B0 = B0 - A1B0 - A0B0"
    signs, terms = nb_expand_moment_normalisation(np.array([[1,0,0,2],
                                                            [2,0,0,0]]),
                                                np.array([3, 2]))
    assert np.allclose(signs, np.array([1, -1, -1])), f"Signs are wrong in {case} expansion"
    assert np.allclose(terms[0], np.array([[2,0,0,0]])), f"Arrays are wrong in {case} expansion"
    assert np.allclose(terms[1], np.array([[1,0,0,0],
                                        [2,0,0,0]])), f"Arrays are wrong in {case} expansion"
    assert np.allclose(terms[2], np.array([[1,0,0,1],
                                        [2,0,0,0]])), f"Arrays are wrong in {case} expansion"

    case = "A2B1 = 1 - B0 - A0 - A1 + A1B0 + A0B0"
    signs, terms = nb_expand_moment_normalisation(np.array([[1,0,0,2],
                                                            [2,0,0,1]]),
                                                np.array([3, 2]))
    assert np.allclose(signs, np.array([1, -1, -1, -1, +1, +1])), f"Signs are wrong in {case} expansion"
    assert np.allclose(terms[0], np.empty((0, 4))), f"Arrays are wrong in {case} expansion"
    assert np.allclose(terms[1], np.array([[2,0,0,0]])), f"Arrays are wrong in {case} expansion"
    assert np.allclose(terms[2], np.array([[1,0,0,0]])), f"Arrays are wrong in {case} expansion"
    assert np.allclose(terms[3], np.array([[1,0,0,1]])), f"Arrays are wrong in {case} expansion"
    assert np.allclose(terms[4], np.array([[1,0,0,0],
                                        [2,0,0,0]])), f"Arrays are wrong in {case} expansion"
    assert np.allclose(terms[5], np.array([[1,0,0,1],
                                        [2,0,0,0]])), f"Arrays are wrong in {case} expansion"

    case = ("A0B2C1D0E1 = A0D0 - A0D0E0 - A0C0D0 + A0C0D0E0 - A0B0D0 - A0B1D0 + A0B0D0E0" + 
                   "+ A0B1D0E0 + A0B0C0D0 + A0B1C0D+ - A0B0C0D0E0 - A0B1C0D0E0")
    signs, terms = nb_expand_moment_normalisation(np.array([[1,0,0,0],
                                                            [2,0,0,2],
                                                            [3,0,0,1],
                                                            [4,0,0,0],
                                                            [5,0,0,1]]),
                                                np.array([2, 3, 2, 2, 2]))

    assert np.allclose(signs, np.array([ 1, -1, -1,  1, -1, -1,  1,  1,  1,  1, -1, -1])
                       ), f"Signs are wrong in {case} expansion"
    correct_terms = [np.array([[1, 0, 0, 0], [4, 0, 0, 0]]),
                    np.array([[1, 0, 0, 0], [4, 0, 0, 0], [5, 0, 0, 0]]),
                    np.array([[1, 0, 0, 0], [3, 0, 0, 0], [4, 0, 0, 0]]),
                    np.array([[1, 0, 0, 0], [3, 0, 0, 0], [4, 0, 0, 0], [5, 0, 0, 0]]),
                    np.array([[1, 0, 0, 0], [2, 0, 0, 0], [4, 0, 0, 0]]),
                    np.array([[1, 0, 0, 0], [2, 0, 0, 1], [4, 0, 0, 0]]),
                    np.array([[1, 0, 0, 0], [2, 0, 0, 0], [4, 0, 0, 0], [5, 0, 0, 0]]),
                    np.array([[1, 0, 0, 0], [2, 0, 0, 1], [4, 0, 0, 0], [5, 0, 0, 0]]),
                    np.array([[1, 0, 0, 0], [2, 0, 0, 0], [3, 0, 0, 0], [4, 0, 0, 0]]),
                    np.array([[1, 0, 0, 0], [2, 0, 0, 1], [3, 0, 0, 0], [4, 0, 0, 0]]),
                    np.array([[1, 0, 0, 0], [2, 0, 0, 0], [3, 0, 0, 0], [4, 0, 0, 0], [5, 0, 0, 0]]),
                    np.array([[1, 0, 0, 0], [2, 0, 0, 1], [3, 0, 0, 0], [4, 0, 0, 0], [5, 0, 0, 0]])]
    for term, correct_term in zip(terms, correct_terms):
        assert np.allclose(term, correct_term), f"Arrays are wrong in {case} expansion"

    signs, terms = nb_expand_moment_normalisation(np.array([[1, 0, 1, 0, 0, 0, 1, 1],
            [2, 0, 0, 0, 1, 0, 1, 1],
            [3, 1, 1, 0, 0, 0, 1, 1],
            [4, 1, 0, 0, 1, 0, 1, 1],
            [5, 1, 0, 1, 0, 0, 1, 0],
            [6, 1, 0, 0, 0, 1, 1, 0],
            [7, 0, 0, 1, 0, 0, 1, 0],
            [8, 0, 0, 0, 0, 1, 1, 0]]), np.array([2,2,2,2,2,2,2,2]))
    assert len(signs) == 16, "Number of terms is wrong"

    signs, terms = nb_expand_moment_normalisation(np.array([[1, 0, 1, 0, 0, 0, 1, 1],
            [2, 0, 0, 0, 1, 0, 1, 1],
            [3, 1, 1, 0, 0, 0, 1, 1],
            [4, 1, 0, 0, 1, 0, 1, 1],
            [5, 1, 0, 1, 0, 0, 1, 1],
            [6, 1, 0, 0, 0, 1, 1, 0],
            [7, 0, 0, 1, 0, 0, 1, 1],
            [8, 0, 0, 0, 0, 1, 1, 0]]), np.array([2,2,2,2,2,2,2,2]))
    assert len(signs) == 64, "Number of terms is wrong"