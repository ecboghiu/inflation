"""
This file contains helper functions to write and export the problems to various
formats.
@authors: Emanuel-Cristian Boghiu, Elie Wolfe, Alejandro Pozas-Kerstjens
"""
import numpy as np

from copy import deepcopy
from scipy.io import savemat
from warnings import warn


def convert_to_human_readable(problem):
    """Convert the SDP relaxation to a human-readable format.

    Parameters
    ----------
    problem : :class:`inflation.InflationSDP`
        The SDP relaxation to write.

    Returns
    -------
    Tuple[str, numpy.ndarray, List, List]
        The first element is the objective function in a string, the second is
        a matrix of strings as the symbolic representation of the moment
        matrix, the third is a list of variable upper and lower bounds, and the
        fourth is a list of equality constraints of the form ``line = 0``.
    """
    matrix = deepcopy(problem.momentmatrix).astype(object)
    ### Process moment matrix
    # Replacer for constants
    constants = {moment.idx: str(value)
                 for moment, value in problem.known_moments.items()}
    constant_replacer = np.vectorize(lambda x: constants.get(x, x))
    # Replacer for semiknowns
    semiknowns = dict()
    for key, val in problem.semiknown_moments.items():
        val_str = val[1].name.replace(", ", ";")
        semiknowns[key.idx] = f"{val[0]}*{val_str}"
    semiknown_replacer = np.vectorize(lambda x: semiknowns.get(x, str(x)))
    # Replacer for remaining symbols
    monomials = dict()
    for mon in problem.monomials:
        monomials[mon.idx] = mon.name.replace(", ", ";")

    def replace_known(monom):
        try:
            replacement = monomials.get(float(monom), monom)
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
            objective = ""
    except KeyError:
        objective = ""
    for variable, coeff in problem.objective.items():
        if variable != problem.One:
            if (coeff < 0) or is_first:
                objective += f"{float(coeff)}*{variable.name}"
            else:
                objective += f"+{float(coeff)}*{variable.name}"
            is_first = False

    ### Process bounds
    bounded_vars = sorted(list(set(problem.moment_upperbounds.keys()).union(
                               set(problem.moment_lowerbounds.keys()))),
                          key=lambda x: x.name)
    bounds = np.zeros((len(bounded_vars), 3), dtype=object)
    for idx, var in enumerate(bounded_vars):
        bounds[idx, 0] = var
        bounds[idx, 1] = problem.moment_lowerbounds.get(var, None)
        bounds[idx, 2] = problem.moment_upperbounds.get(var, None)

    ### Process equalities
    equalities = []
    for eq in problem.moment_equalities:
        equality = ""
        for monom, coeff in eq.items():
            equality += "+" if coeff > 0 else "-"
            if monom == problem.One:
                equality += str(abs(coeff))
            else:
                if np.isclose(abs(coeff), 1):
                    equality += monom.name
                else:
                    equality += f"{abs(coeff)}*{monom.name}"
        equalities.append(equality[1:] if equality[0] == "+" else equality)
    return objective, matrix, bounds.tolist(), equalities


def write_to_csv(problem, filename):
    """Export the problem in a human-readable form in a CSV table.

    :param problem: The SDP relaxation to write.
    :type problem: :class:`inflation.InflationSDP`
    :type filename: str
    """
    objective, matrix, bounds, equalities = convert_to_human_readable(problem)
    f = open(filename, "w")
    f.write("Objective: " + objective + "\n")
    for matrix_line in matrix:
        f.write(
            str(list(matrix_line))[1:-1].replace(" ", "").replace("\'", ""))
        f.write("\n")
    f.write("Bounds:\n")
    f.write("Variable,lower,upper\n")
    for bound_line in bounds:
        f.write(str(bound_line)[1:-1].replace(" ", ""))
        f.write("\n")
    f.write("\nEqualities (format: line = 0):\n")
    for equality in equalities:
        f.write(equality)
        f.write("\n")
    f.close()


def write_to_mat(problem, filename):
    """Export the problem to MATLAB .mat file.

    :param problem: The SDP relaxation to write.
    :type problem: :class:`inflation.InflationSDP`
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
                   for mon, bnd in problem._processed_moment_lowerbounds.items()]
    upperbounds = [[mon.idx + offset, bnd]
                   for mon, bnd in problem._processed_moment_upperbounds.items()]
    names       = np.array([[mon.idx + offset, mon.name]
                             for mon in problem.monomials], dtype=object)
    equalities  = []
    for eq in problem.moment_equalities:
        equality = {'moments': np.array([mon.idx + offset for mon in eq]),
                    'coeffs':  np.array(list(eq.values()))}
        equalities.append(equality)
  
    inequalities  = []
    for ineq in problem.moment_inequalities:
        inequality = {'moments': np.array([mon.idx + offset for mon in ineq]),
                      'coeffs':  np.array(list(ineq.values()))}
        inequalities.append(inequality)

    savemat(filename,
            mdict={"moments_idx2name":    names,
                   "momentmatrix":        final_positions_matrix,
                   "objective":           objective,
                   "known_moments":       known_moments,
                   "semiknown_moments":   semiknown_moments,
                   "moment_lowerbounds":  lowerbounds,
                   "moment_upperbounds":  upperbounds,
                   "moment_equalities":   equalities,
                   "moment_inequalities": inequalities
                   }
            )


def write_to_sdpa(problem, filename):
    """Export the problem to a file in .dat-s format. See specifications at
    http://euler.nmt.edu/~brian/sdplib/FORMAT.

    :param problem: The SDP relaxation to write.
    :type problem: :class:`inflation.InflationSDP`
    :type filename: str
    """
    # Compute actual number of variables: all in the moment matrix, minus those
    # with known values, minus those that participate in LPI constraints, plus
    # those where the unknown part of the LPI constraint is not in the original
    # moment matrix
    potential_nvars = problem.momentmatrix.max() - 1
    if len(problem.known_moments) == 1:
        known_vars = 0
    else:
        known_vars = (len(problem.known_moments)
                      - problem.momentmatrix_has_a_zero
                      - problem.momentmatrix_has_a_one)
    semiknown_vars = len(problem.semiknown_moments)
    new_vars       = sum([mon[1][1].idx > problem.momentmatrix.max()
                          for mon in problem.semiknown_moments.items()])
    nvars = potential_nvars - known_vars - semiknown_vars + new_vars

    known_moments_indices = {mon.idx: coeff
                             for mon, coeff in problem.known_moments.items()}
    semiknowns  = {mon.idx: (subs[0], subs[1].idx)
                   for mon, subs in problem.semiknown_moments.items()}
    lines       = []
    var_corresp = {}
    new_var     = 1
    block       = 1
    blockstruct = [str(problem.momentmatrix.shape[0])]
    for ii, row in enumerate(problem.momentmatrix):
        for jj, var in enumerate(row):
            if jj >= ii:
                if var == 0:
                    pass
                elif var == 1:
                    lines.append(f"0\t{block}\t{ii+1}\t{jj+1}\t-1.0\n")
                elif var in known_moments_indices.keys():
                    coeff = known_moments_indices[var]
                    lines.append(
                        f"0\t{block}\t{ii+1}\t{jj+1}\t-{abs(coeff)}\n")
                elif var in semiknowns.keys():
                    coeff, subs = semiknowns[var]
                    try:
                        var = var_corresp[subs]
                        lines.append(
                            f"{var}\t{block}\t{ii+1}\t{jj+1}\t{coeff}\n")
                    except KeyError:
                        var_corresp[subs] = new_var
                        lines.append(
                            f"{new_var}\t{block}\t{ii+1}\t{jj+1}\t{coeff}\n")
                        new_var += 1
                else:
                    try:
                        var = var_corresp[var]
                        lines.append(f"{var}\t{block}\t{ii+1}\t{jj+1}\t1.0\n")
                    except KeyError:
                        var_corresp[int(var)] = new_var
                        lines.append(
                            f"{new_var}\t{block}\t{ii+1}\t{jj+1}\t1.0\n")
                        new_var += 1

    # Prepare objective
    objective = np.zeros(nvars)
    for variable, coeff in problem.objective.items():
        if variable == problem.One:
            if abs(coeff) > 1e-8:
                warn(f"Export removed the constant {coeff} from the objective")
        else:
            objective[var_corresp[variable.idx] - 1] = coeff
    objective = str(objective.tolist()).replace("[", "").replace("]", "")
    objective_constant = problem.objective[problem.One]

    # Prepare upper bounds
    if len(problem.moment_upperbounds) > 0:
        block += 1
        ii     = 1
        block_size = len([mon for mon in problem.moment_upperbounds.keys()
                          if ((mon != problem.One)
                          and (mon not in problem.known_moments.keys())
                          and (mon not in problem.semiknown_moments.keys()))])
        blockstruct.append(str(-block_size))
    for var, ub in problem.moment_upperbounds.items():
        if ((var != problem.One)
            and (var not in problem.known_moments.keys())
            and (var not in problem.semiknown_moments.keys())):
            var = var_corresp[var.idx]
            lines.append(f"{var}\t{block}\t{ii}\t{ii}\t-1.0\n")
            if abs(ub) > 1e-8:
                lines.append(f"0\t{block}\t{ii}\t{ii}\t-{ub}\n")
            ii += 1

    # Prepare lower bounds
    if len(problem.moment_lowerbounds) > 0:
        block += 1
        ii     = 1
        block_size = len([mon for mon in problem.moment_lowerbounds.keys()
                          if ((mon != problem.One)
                          and (mon not in problem.known_moments.keys())
                          and (mon not in problem.semiknown_moments.keys()))])
        blockstruct.append(str(-block_size))
    for var, lb in problem.moment_lowerbounds.items():
        if ((var != problem.One)
            and (var not in problem.known_moments.keys())
            and (var not in problem.semiknown_moments.keys())):
            var = var_corresp[var.idx]
            lines.append(f"{var}\t{block}\t{ii}\t{ii}\t1.0\n")
            if abs(lb) > 1e-8:
                lines.append(f"0\t{block}\t{ii}\t{ii}\t{lb}\n")
            ii += 1

    # Prepare equalities
    if len(problem.moment_equalities) > 0:
        block += 1
        ii     = 1
        block_size = 2*len(problem.moment_equalities)
        blockstruct.append(str(-block_size))
    for equality in problem.moment_equalities:
        for var, coeff in equality.items():
            if var != problem.One:
                var = var_corresp[var.idx]
                lines.append(f"{var}\t{block}\t{ii}\t{ii}\t{coeff}\n")
                lines.append(f"{var}\t{block}\t{ii+1}\t{ii+1}\t{-coeff}\n")
            else:
                lines.append(f"0\t{block}\t{ii}\t{ii}\t{-coeff}\n")
                lines.append(f"0\t{block}\t{ii+1}\t{ii+1}\t{coeff}\n")
        ii += 2

    file_ = open(filename, "w")
    file_.write("\"file " + filename + " generated by inflation\"\n")
    if abs(objective_constant) > 1e-8:
        file_.write("\"the objective contains a constant term of value "
                    + str(objective_constant) + " that is not included in the "
                    + "program below\"\n")
    file_.write(f"{nvars} = number of vars\n")
    file_.write(f"{block} = number of blocks\n")
    file_.write("(" + ",".join(blockstruct) + ") = BlockStructure\n")
    file_.write("{" + objective + "}\n")
    for line in sorted(lines, key=lambda x: (int(x.split("\t")[1]),
                                             int(x.split("\t")[0]))):
        file_.write(line)
    file_.close()

