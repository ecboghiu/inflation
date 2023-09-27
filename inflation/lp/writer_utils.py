"""
This file contains helper functions to save a linear programming relaxation
problem to file.

@authors: Erica Han, Elie Wolfe
"""

import mosek
from copy import deepcopy


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
    (objective,
     known_vars,
     semiknown_vars,
     internal_equalities,
     inequalities,
     lower_bounds,
     upper_bounds,
     variables) = process_arguments(args)

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
    (objective,
     known_vars,
     semiknown_vars,
     internal_equalities,
     inequalities,
     lower_bounds,
     upper_bounds,
     variables) = process_arguments(args)

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


def process_arguments(args: dict) -> tuple:
    """Helper function that processes arguments and variables in preparation
    for writing.

    Parameters
    ----------
    args : dict
        The arguments to be processed.

    Returns
    -------
    tuple
        Processed objective, known_vars, semiknown_vars, internal_equalities,
        inequalities, lower_bounds, upper_bounds, variables
    """
    args = deepcopy(args)
    objective = args.get('objective')
    known_vars = args.get('known_vars')
    if known_vars is None:
        known_vars = {}
    semiknown_vars = args.get('semiknown_vars')
    if semiknown_vars is None:
        semiknown_vars = {}
    equalities = args.get('equalities')
    if equalities is None:
        equalities = []
    inequalities = args.get('inequalities')
    if inequalities is None:
        inequalities = []
    lower_bounds = args.get('lower_bounds')
    if lower_bounds is None:
        lower_bounds = {}
    upper_bounds = args.get('upper_bounds')
    if upper_bounds is None:
        upper_bounds = {}

    # Process semiknown variables
    internal_equalities = equalities.copy()
    for x, (c, x2) in semiknown_vars.items():
        internal_equalities.append({x: 1, x2: -c})

    # Set variables (excluding known variables)
    variables = set()
    variables.update(objective)
    for ineq in inequalities:
        variables.update(ineq)
    for eq in internal_equalities:
        variables.update(eq)
    variables.difference_update(known_vars)

    # Replace variable names (LP file cannot have <, >, etc.)
    var_index = {x: f"x{i}" for i, x in enumerate(variables)}
    for x in set(objective).difference(known_vars):
        objective[var_index[x]] = objective.pop(x)
    for ineq in inequalities:
        for x in set(ineq).difference(known_vars):
            ineq[var_index[x]] = ineq.pop(x)
    for eq in internal_equalities:
        for x in set(eq).difference(known_vars):
            eq[var_index[x]] = eq.pop(x)
    for x in set(lower_bounds).difference(known_vars):
        lower_bounds[var_index[x]] = lower_bounds.pop(x)
    for x in set(upper_bounds).difference(known_vars):
        upper_bounds[var_index[x]] = upper_bounds.pop(x)
    variables = list(var_index.values())

    return objective, known_vars, semiknown_vars, internal_equalities, \
        inequalities, lower_bounds, upper_bounds, variables


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
