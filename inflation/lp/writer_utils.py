import mosek


def write_to_lp(args: dict,
                filename: str) -> None:
    """Export the problem to a file in .lp format. Specification can be found
    at https://docs.mosek.com/latest/pythonapi/lp-format.html.

    Parameters
    ----------
    args : dict
        The arguments of the problem to write to the file.
    filename : str
        The file to write to.
    """
    objective = args['objective']
    known_vars = args['known_vars']
    semiknown_vars = args['semiknown_vars']
    equalities = args['equalities']
    inequalities = args['inequalities']
    lower_bounds = args['lower_bounds']
    upper_bounds = args['upper_bounds']

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
    variables = set(var_index.values())

    # Write problem to file
    f = open(filename, "w")
    f.write(f"\\ File: {filename}\n")
    f.write("maximize\n")
    f.write(f"obj: {format_constraint_lp(known_vars, objective, 'obj')}\n")
    f.write("subject to\n")

    i = 1
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
    for var in variables:
        if var in lower_bounds:
            lb = lower_bounds[var]
        else:
            lb = "-infinity"
        if var in upper_bounds:
            ub = upper_bounds[var]
        else:
            ub = "+infinity"
        f.write(f" {lb} <= {var} <= {ub}\n")
    f.write("end")
    f.close()


def write_to_mps(args: dict,
                 filename: str) -> None:
    """Export the problem to a file in .mps format. Specification can be found
    at https://docs.mosek.com/latest/pythonapi/mps-format.html.

    Parameters
    ----------
    args : dict
        The arguments of the problem to write to the file.
    filename : str
        The file to write to.
    """
    lps_filename = f"{filename.split('.')[0]}.lp"
    write_to_lp(args, lps_filename)
    with mosek.Task() as t:
        t.readdata(f"{lps_filename}")
        t.writedata(filename)


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
