"""
This file contains helper functions to write and export the problems into
various formats.
@authors: Alejandro Pozas-Kerstjens, Emanuel-Cristian Boghiu
"""
import numpy as np
from scipy.io import savemat
from warnings import warn
from typing import Callable, Dict, List, Tuple, Union
import pickle


def convert_to_human_readable(problem):
    """Convert the SDP relaxation to a human-readable format.

    :param problem: The SDP relaxation to write.
    :type problem: :class:`inflapyon.InflationSDP`.
    :returns: tuple of the objective function in a string and a matrix of
              strings as the symbolic representation of the moment matrix
    """
    ### Process moment matrix
    # Replacer for constants
    constant_replacer = np.vectorize(lambda x: problem.known_moments.get(x, x))
    # Replacer for remaining symbols
    monomial_dict = dict(problem.monomials_list)
    def replace_known(monom):
        try:
            replacement = monomial_dict.get(float(monom), monom)
        except ValueError:
            replacement = monom
        return replacement
    known_replacer = np.vectorize(replace_known)
    # Replacer for semiknowns
    semiknown_dict = {}
    for key, val in problem.semiknown_moments.items():
        semiknown_dict[key] = f"{val[0]}*{monomial_dict[int(val[1])]}"
    semiknown_replacer = np.vectorize(lambda x: semiknown_dict.get(x, str(x)))
    matrix = constant_replacer(problem.momentmatrix)
    matrix = semiknown_replacer(matrix)
    matrix = np.triu(known_replacer(matrix).astype(object))

    ### Process objective
    is_first = True
    try:
        independent_term = float(problem._objective_as_dict[1])
        if abs(independent_term) > 1e-8:
            objective = str(independent_term)
        else:
            objective = ''
        is_first = False
    except KeyError:
        objective = ''
    for variable, coeff in problem._objective_as_dict.items():
        if variable > 1:
            if (coeff > 0) or (not is_first):
                objective += f"+{float(coeff)}*{monomial_dict[variable]}"
            else:
                objective += f"{float(coeff)}*{monomial_dict[variable]}"
                is_first = False

    return objective, matrix


def write_to_csv(problem, filename):
    """Export the problem in a human-readable form in a CSV table.

    :param problem: The SDP relaxation to write.
    :type problem: :class:`inflapyon.InflationSDP`.
    :type filename: str
    """
    objective, matrix = convert_to_human_readable(problem)
    f = open(filename, 'w')
    f.write("Objective: " + objective + "\n")
    for matrix_line in matrix:
        f.write(str(list(matrix_line)).replace('[', '').replace(']', '')
                .replace('\'', ''))
        f.write('\n')
    f.close()


def write_to_mat(problem, filename):
    """Export the problem to MATLAB .mat file.

    :param problem: The SDP relaxation to write.
    :type problem: :class:`inflapyon.InflationSDP`.
    :type filename: str
    """
    # MATLAB does not like 0s, so we shift all by 1
    final_positions_matrix = problem.momentmatrix + 1
    nr_unknown_moments     = int(np.max(final_positions_matrix))
    known_moments          = np.array(list(problem.known_moments.items()))
    known_moments[:, 0]    += 1
    physical_monomials     = np.array(problem.physical_monomials) + 1
    monomials_list         = problem.monomials_list
    monomials_list[:,0]    += 1
    semiknown_moments      = np.zeros((len(problem.semiknown_moments), 3),
                                      dtype=object)
    for idx, [var, [factor, subs]] in enumerate(problem.semiknown_moments):
        semiknown_moments[idx, 0] = var + 1
        semiknown_moments[idx, 1] = factor
        semiknown_moments[idx, 2] = subs + 1
    nr_unknown_moments = int(np.max(final_positions_matrix)
                             - len(semiknown_moments))
    objective = np.array(list(problem._objective_as_dict.items())).astype(float)
    objective[:, 0] += 1

    savemat(filename,
    mdict={'G': final_positions_matrix,
           'known_moments':      known_moments,
           'nr_unknown_moments': nr_unknown_moments,
           'propto':             semiknown_moments,
           'obj':                objective,
           'monomials_string':   monomials_list,
           "positive_vars":      physical_monomials
           }
    )

def write_to_sdpa(problem, filename):
    """Export the problem to a file in  . See specifications at
    http://euler.nmt.edu/~brian/sdplib/FORMAT.

    :param problem: The SDP relaxation to write.
    :type problem: :class:`inflapyon.InflationSDP`.
    :type filename: str
    """
    # Compute actual number of variables
    potential_nvars = problem.momentmatrix.max() - 1
    known_vars      = len(problem.known_moments) - 2 # Except 0 and 1
    semiknown_vars  = len(problem.semiknown_moments)
    nvars           = potential_nvars - known_vars - semiknown_vars

    # Replacer for semiknowns
    lines = []
    new_var_dict = {}
    new_var = 1
    for ii, row in enumerate(problem.momentmatrix):
        for jj, var in enumerate(row):
            if jj >= ii:
                if var == 0:
                    pass
                elif var == 1:
                    lines.append(f"0\t1\t{ii+1}\t{jj+1}\t-1.0\n")
                elif var <= problem._n_known + 1:
                    try:
                        coeff = problem.known_moments[var]
                        if coeff >= 0:
                            coeff_str = f"-{coeff}"
                        else:
                            coeff_str = f"+{abs(coeff)}"
                        lines.append(f"0\t1\t{ii+1}\t{jj+1}\t{coeff_str}\n")
                    except KeyError:
                        try:
                            var = new_var_dict[int(var)]
                            lines.append(f"{var}\t1\t{ii+1}\t{jj+1}\t1.0\n")
                        except KeyError:
                            new_var_dict[int(var)] = new_var
                            lines.append(f"{new_var}\t1\t{ii+1}\t{jj+1}\t1.0\n")
                            new_var += 1
                elif var <= problem._n_something_known + 1:
                    try:
                        coeff, subs = problem.semiknown_moments[var]
                    except KeyError:
                        # There is no LPI constraint associated, so we treat it
                        # as a regular unknown variable
                        try:
                            var = new_var_dict[int(var)]
                            lines.append(f"{var}\t1\t{ii+1}\t{jj+1}\t1.0\n")
                        except KeyError:
                            new_var_dict[int(var)] = new_var
                            lines.append(f"{new_var}\t1\t{ii+1}\t{jj+1}\t1.0\n")
                            new_var += 1
                    else:
                        try:
                            subs = new_var_dict[int(subs)]
                            lines.append(
                                f"{subs}\t1\t{ii+1}\t{jj+1}\t{coeff}\n"
                                         )
                        except KeyError:
                            # The substituted variable is not yet in the
                            # variable list, so we add it
                            new_var_dict[int(subs)] = new_var
                            lines.append(
                                f"{new_var}\t1\t{ii+1}\t{jj+1}\t{coeff}\n"
                                         )
                            new_var += 1
                else:
                    try:
                        var = new_var_dict[int(var)]
                        lines.append(f"{var}\t1\t{ii+1}\t{jj+1}\t1.0\n")
                    except KeyError:
                        new_var_dict[int(var)] = new_var
                        lines.append(f"{new_var}\t1\t{ii+1}\t{jj+1}\t1.0\n")
                        new_var += 1

    # Prepare objective
    objective = np.zeros(nvars)
    for variable, coeff in problem._objective_as_dict.items():
        if variable == 1:
            if abs(coeff) > 1e-8:
                warn(f"Export removed the constant {coeff} from the objective")
        else:
            objective[new_var_dict[variable] - 1] = coeff
    objective = str(objective.tolist()).replace('[', '').replace(']', '')
    objective_constant = problem._objective_as_dict[1]

    file_ = open(filename, 'w')
    file_.write('"file ' + filename + ' generated by causalinflation"\n')
    if abs(objective_constant) > 1e-8:
        file_.write('"the objective contains a constant term of value '
                    + str(objective_constant) + ' that is not included in the '
                    + 'program below"\n')
    file_.write(f'{nvars} = number of vars\n')
    file_.write('1 = number of blocks\n')
    file_.write(f'({problem.momentmatrix.shape[0]}) = BlockStructure\n')
    file_.write('{' + objective + '}\n')
    for line in sorted(lines, key=lambda x: int(x.split('\t')[0])):
        file_.write(line)
    file_.close()


def pickle_load(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)
