"""
This file contains helper functions to write and export the problems into
various formats.
@authors: Alejandro Pozas-Kerstjens, Emanuel-Cristian Boghiu
"""
import numpy as np
from copy import deepcopy
from scipy.io import savemat
from warnings import warn
from typing import Callable, Dict, List, Tuple, Union
import pickle


def convert_to_human_readable(problem):
    """Convert the SDP relaxation to a human-readable format.

    :param problem: The SDP relaxation to write.
    :type problem: :class:`causalinflation.InflationSDP`
    :returns: tuple of the objective function in a string and a matrix of
              strings as the symbolic representation of the moment matrix
    """
    matrix = deepcopy(problem.momentmatrix).astype(object)
    ### Process moment matrix
    # Replacer for constants
    constant_dict = {moment.idx: str(value)
                     for moment, value in problem.known_moments.items()}
    constant_replacer = np.vectorize(lambda x: constant_dict.get(x, x))
    # Replacer for semiknowns
    semiknown_dict = dict()
    for key, val in problem.semiknown_moments.items():
        val_str = val[1].name.replace(", ", ";")
        semiknown_dict[key.idx] = f"{val[0]}*{val_str}"
    semiknown_replacer = np.vectorize(lambda x: semiknown_dict.get(x, str(x)))
    # Replacer for remaining symbols
    monomial_dict = dict()
    for mon in problem.list_of_monomials:
        monomial_dict[mon.idx] = mon.name.replace(", ", ";")
    def replace_known(monom):
        try:
            replacement = monomial_dict.get(float(monom), monom)
        except ValueError:
            replacement = monom
        return replacement
    known_replacer = np.vectorize(replace_known)
    matrix = constant_replacer(matrix)
    matrix = semiknown_replacer(matrix)
    matrix = np.triu(known_replacer(matrix).astype(object))

    ### Process objective
    is_first = True
    try:
        independent_term = float(problem.objective[problem.One])
        if abs(independent_term) > 1e-8:
            objective = str(independent_term)
            is_first = False
        else:
            objective = ''
    except KeyError:
        objective = ''
    for variable, coeff in problem.objective.items():
        if variable != problem.One:
            if (coeff < 0) or is_first:
                objective += f"{float(coeff)}*{variable.name}"
            else:
                objective += f"+{float(coeff)}*{variable.name}"
                is_first = False

    return objective, matrix


def write_to_csv(problem, filename):
    """Export the problem in a human-readable form in a CSV table.

    :param problem: The SDP relaxation to write.
    :type problem: :class:`causalinflation.InflationSDP`
    :type filename: str
    """
    objective, matrix = convert_to_human_readable(problem)
    f = open(filename, "w")
    f.write("Objective: " + objective + "\n")
    for matrix_line in matrix:
        f.write(str(list(matrix_line))[1:-1].replace(" ", "").replace("\'", ""))
        f.write("\n")
    f.close()


def write_to_mat(problem, filename):
    """Export the problem to MATLAB .mat file.

    :param problem: The SDP relaxation to write.
    :type problem: :class:`causalinflation.InflationSDP`
    :type filename: str
    """
    # MATLAB does not like 0s, so we shift all by 1 if the variable 0 exists
    offset = 1 if problem.momentmatrix_has_a_zero else 0
    final_positions_matrix = problem.momentmatrix + offset
    known_moments = [[mon.idx + offset, val]
                     for mon, val in problem.known_moments.items()]
    semiknown_initial = []
    semiknown_factors = []
    semiknown_final   = []
    for semiknown in problem.semiknown_moments.items():
        semiknown_initial.append(semiknown[0].idx + offset)
        semiknown_factors.append(semiknown[1][0])
        semiknown_final.append(semiknown[1][1].idx + offset)
    semiknown_moments = np.vstack([semiknown_initial,
                                   semiknown_factors,
                                   semiknown_final]).T
    objective   = [[mon.idx + offset, float(coeff)]
                   for mon, coeff in problem.objective.items()
                   if abs(coeff) > 1e-8]
    lowerbounds = [[mon.idx + offset, bnd]
                   for mon, bnd in problem.moment_lowerbounds.items()]
    upperbounds = [[mon.idx + offset, bnd]
                   for mon, bnd in problem.moment_upperbounds.items()]
    names       = [[mon.idx + offset, mon.name]
                   for mon in problem.list_of_monomials]

    savemat(filename,
            mdict={"Gamma":           final_positions_matrix,
                   "known_moments":   known_moments,
                   "semiknown":       semiknown_moments,
                   "obj":             objective,
                   "monomials_names": np.asarray(names, dtype=object),
                   "lowerbounds":     lowerbounds,
                   "upperbounds":     upperbounds
                  }
            )

def write_to_sdpa(problem, filename):
    """Export the problem to a file in .dat-s format. See specifications at
    http://euler.nmt.edu/~brian/sdplib/FORMAT.

    :param problem: The SDP relaxation to write.
    :type problem: :class:`causalinflation.InflationSDP`
    :type filename: str
    """
    # Compute actual number of variables
    potential_nvars = problem.momentmatrix.max() - 1
    known_vars = 0 if len(problem.known_moments_idx_dict) == 0 else len(problem.known_moments_idx_dict) - 2
    semiknown_vars = 0 if len(problem.semiknown_moments_idx_dict) == 0 else len(problem.semiknown_moments_idx_dict)
    nvars = potential_nvars - known_vars - semiknown_vars

    # Replacer for semiknowns
    if len(problem.semiknown_moments_idx_dict) > 0:
        semiknown_list = np.zeros((problem.semiknown_moments_idx_dict.shape[0], 2),
                                  dtype=object)
        for ii, semiknown in enumerate(problem.semiknown_moments_idx_dict):
            semiknown_list[ii,0] = int(semiknown[0])
            semiknown_list[ii,1] = semiknown[1:]
        semiknown_dict = dict(semiknown_list)
    else:
        semiknown_dict = {}
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
                        coeff = problem.known_moments_idx_dict[var]
                        lines.append(f"0\t1\t{ii+1}\t{jj+1}\t-{abs(coeff)}\n")
                    except IndexError:
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
