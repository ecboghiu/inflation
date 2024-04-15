"""
This file contains helper functions to save a linear programming relaxation
problem to file.

@authors: Erica Han, Elie Wolfe
"""

import mosek
from copy import deepcopy
import numpy as np


def write_to_lp(args: dict, filename: str) -> None:
    """Export the problem to a file in .lp format. Specification can be found
    at https://docs.mosek.com/latest/pythonapi/lp-format.html.

    Parameters
    ----------
    args : dict
        The arguments of the problem to write to the file.
    filename : str
        The file to write to.
    """
    processed_args = process_arguments(args)
    objective = processed_args["objective"]
    known_vars = processed_args["known_vars"]
    internal_equalities = processed_args["equalities"]
    inequalities = processed_args["inequalities"]
    lower_bounds = processed_args["lower_bounds"]
    upper_bounds = processed_args["upper_bounds"]
    variables = processed_args["variables"]


    # Write problem to file
    f = open(filename, "w")
    f.write(f"\\ File: {filename}\n")
    f.write("maximize\n")
    f.write(f"obj: {format_constraint_lp(known_vars, objective, 'obj')}\n")
    f.write("subject to\n")

    i = 0
    for eq in internal_equalities:
        cons = format_constraint_lp(known_vars, eq, 'eq')
        if cons != "":
            f.write(f"c{i}: {cons}\n")
            i += 1
    for ineq in inequalities:
        cons = format_constraint_lp(known_vars, ineq, 'ineq')
        if cons != "":
            f.write(f"c{i}: {cons}\n")
            i += 1

    f.write("bounds\n")
    for x in lower_bounds:
        f.write(f"{lower_bounds[x]} <= {x}")
    for x in upper_bounds:
        f.write(f"{x} <= {upper_bounds[x]}")
    f.write("end")
    f.close()


def write_to_mps(args: dict, filename: str) -> None:
    """Export the problem to a file in .mps format. Specification can be found
    at https://docs.mosek.com/latest/pythonapi/mps-format.html.

    Parameters
    ----------
    args : dict
        The arguments of the problem to write to the file.
    filename : str
        The file to write to.
    """
    processed_args = process_arguments(args)
    objective = processed_args["objective"]
    known_vars = processed_args["known_vars"]
    internal_equalities = processed_args["equalities"]
    inequalities = processed_args["inequalities"]
    lower_bounds = processed_args["lower_bounds"]
    upper_bounds = processed_args["upper_bounds"]
    variables = processed_args["variables"]

    # Exclude constraints in which variables are all known values
    internal_equalities = [eq for eq in internal_equalities
                           if set(eq).difference(known_vars)]
    inequalities = [ineq for ineq in inequalities
                    if set(ineq).difference(known_vars)]
    constraints = internal_equalities + inequalities
    nof_equalities = len(internal_equalities)
    nof_inequalities = len(inequalities)

    f = open(filename, "w")
    f.write(f"* File: {filename}\n")
    f.write(f"NAME          {filename.split('.')[0]}\n")
    f.write(f"OBJSENSE\n    MAX\n")

    # Specify rows of the problem
    f.write(f"ROWS\n")
    f.write(f" N  obj\n")
    for i in range(nof_equalities):
        f.write(f" E  c{i}\n")
    for i in range(nof_inequalities):
        f.write(f" G  c{nof_equalities + i}\n")

    # Specify columns of the problem
    f.write(f"COLUMNS\n")
    for x in variables:
        if x in set(objective).difference(known_vars):
            arr = [x, 'obj', objective[x]]
            f.write("    {:10}{:10}{:<10.4f}\n".format(*arr))
        for i, eq in enumerate(internal_equalities):
            if x in set(eq).difference(known_vars):
                arr = [x, f"c{i}", eq[x]]
                f.write("    {:10}{:10}{:<10.4f}\n".format(*arr))
        for i, ineq in enumerate(inequalities):
            if x in set(ineq).difference(known_vars):
                arr = [x, f"c{nof_equalities + i}",
                       ineq[x]]
                f.write("    {:10}{:10}{:<10.4f}\n".format(*arr))

    f.write(f"RHS\n")
    # Specify objective offset
    rhs = 0
    for x in set(objective).intersection(known_vars):
        rhs -= objective[x] * known_vars[x]
    if rhs != 0:
        arr = ['rhs', 'obj', rhs]
        f.write("    {:10}{:10}{:<10.4f}\n".format(*arr))

    # Specify non-zero RHS values of the constraints
    for i, c in enumerate(constraints):
        rhs = 0
        for x in set(c).intersection(known_vars):
            rhs -= c[x] * known_vars[x]
        if rhs != 0:
            arr = ['rhs', f"c{i}", rhs]
            f.write("    {:10}{:10}{:<10.4f}\n".format(*arr))

    f.write(f"RANGES\n")

    # Specify variable bound values
    f.write(f"BOUNDS\n")
    for x in lower_bounds:
        arr = ['bound', x, lower_bounds[x]]
        f.write(" LO {:10}{:10}{:<10.4f}\n".format(*arr))
    for x in upper_bounds:
        arr = ['bound', x, upper_bounds[x]]
        f.write(" UP {:10}{:10}{:<10.4f}\n".format(*arr))
    f.write(f"ENDATA")
    f.close()


def lp_to_mps(args: dict, filename: str) -> None:
    """Export the problem to a file in .mps format by writing it as an LP file
    then converting it to MPS through Mosek.

    Parameters
    ----------
    args : dict
        The arguments of the problem to write to the file.
    filename : str
        The file to write to.
    """
    lp_filename = f"{filename.split('.')[0]}.lp"
    write_to_lp(args, lp_filename)
    with mosek.Task() as t:
        t.readdata(f"{lp_filename}")
        t.writedata(filename)


def process_arguments(args: dict) -> dict:
    """Helper function that processes arguments and variables in preparation
    for writing.

    Parameters
    ----------
    args : dict
        The arguments to be processed.

    Returns
    -------
    dict
        Processed objective, known_vars, internal_equalities,
        inequalities, lower_bounds, upper_bounds, variables
    """
    old_variables = args["variables"]
    old_equalities = deepcopy(args["equalities"])
    old_semiknown = args["semiknown_vars"]
    for x, (c, x2) in old_semiknown.items():
        old_equalities.append({x: 1, x2: -c})
    old_inequalities = args["inequalities"]
    old_known_vars = args["known_vars"]
    old_objective = args.get("objective", {})
    old_lower_bounds = args.get("lower_bounds", {})
    old_upper_bounds = args.get("upper_bounds", {})
    var_idx =  {x: f"x{i}" for i, x in enumerate(old_variables)}
    new_variables = list(var_idx.values())
    new_equalities = [{var_idx[x]: c for x, c in eq.items()} for eq in  old_equalities]
    new_inequalities = [{var_idx[x]: c for x, c in ineq.items()} for ineq in old_inequalities]
    new_known_vars = {var_idx[x]: c for x, c in old_known_vars.items()}
    new_lower_bounds = {var_idx[x]: c for x, c in old_lower_bounds.items()}
    new_upper_bounds = {var_idx[x]: c for x, c in old_upper_bounds.items()}
    new_objective = {var_idx[x]: c for x, c in old_objective.items()}

    return {
        "objective": new_objective,
        "variables": new_variables,
        "equalities": new_equalities,
        "inequalities": new_inequalities,
        "known_vars": new_known_vars,
        "lower_bounds": new_lower_bounds,
        "upper_bounds": new_upper_bounds}


def format_constraint_lp(known_vars: dict,
                         constraint: dict,
                         cons_type: str) -> str:
    """Helper function to format constraints (and the objective) for the LP
    file format.

    Parameters
    ----------
    known_vars : dict
        Known values of variables.
    constraint : dict
        The constraint to be formatted.
    cons_type : str
        Whether the constraint is an objective ('obj'), equality ('eq'), or
        inequality ('ineq).

    Returns
    -------
    str
        The constraint as a human-readable string
    """
    # Process known values
    c = 0
    for x in set(constraint).intersection(known_vars):
        c += constraint[x] * known_vars[x]

    # Process unknown values
    cons = ""
    for x in set(constraint).difference(known_vars):
        cons += f"{constraint[x]} {x} + "
    if cons == "":
        return cons

    if cons_type == 'obj':
        cons += str(c)
    elif cons_type == 'eq':
        cons = cons[:-3] + ' = ' + str(-c)
    elif cons_type == 'ineq':
        cons = cons[:-3] + ' >= ' + str(-c)
    return cons
