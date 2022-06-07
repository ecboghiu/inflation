# -*- coding: utf-8 -*-
"""
This file contains helper functions to work with noncommutative polynomials
and Hamiltonians.

Created on Thu May  2 16:03:05 2013

@author: Peter Wittek
"""
from __future__ import division, print_function
from sympy import adjoint, conjugate, S, Symbol, Pow, Number, expand, I
from sympy.physics.quantum import HermitianOperator, Operator
from scipy.sparse import lil_matrix


def flatten(lol):
    """Flatten a list of lists to a list.

    :param lol: A list of lists in arbitrary depth.
    :type lol: list of list.

    :returns: flat list of elements.
    """
    new_list = []
    for element in lol:
        if element is None:
            continue
        elif not isinstance(element, list) and not isinstance(element, tuple):
            new_list.append(element)
        elif len(element) > 0:
            new_list.extend(flatten(element))
    return new_list


def simplify_polynomial(polynomial, monomial_substitutions):
    """Simplify a polynomial for uniform handling later.
    """
    if isinstance(polynomial, (int, float, complex)):
        return polynomial
    polynomial = (1.0 * polynomial).expand(mul=True,
                                           multinomial=True)
    if is_number_type(polynomial):
        return polynomial
    if polynomial.is_Mul:
        elements = [polynomial]
    else:
        elements = polynomial.as_coeff_mul()[1][0].as_coeff_add()[1]
    new_polynomial = 0
    # Identify its constituent monomials
    for element in elements:
        monomial, coeff = separate_scalar_factor(element)
        monomial = apply_substitutions(monomial, monomial_substitutions)
        new_polynomial += coeff * monomial
    return new_polynomial


def is_pure_substitution_rule(lhs, rhs):
    if is_number_type(rhs):
        return True
    if rhs.is_Mul:
        elements = [rhs]
    else:
        elements = rhs.as_coeff_mul()[1][0].as_coeff_add()[1]
    for element in elements:
        monomial, _ = separate_scalar_factor(element)
        for atom in monomial.atoms():
            if atom.is_Number:
                continue
            if not lhs.has(atom):
                return False
    return True


def remove_scalar_factor(monomial):
    """Return monomial without constant factor.
    """
    monomial, _ = separate_scalar_factor(monomial)
    return monomial


def __separate_scalar_factor(monomial):
    """Separate the constant factor from a monomial.
    """
    scalar_factor = 1
    if is_number_type(monomial):
        return S.One, monomial
    if monomial == 0:
        return S.One, 0
    comm_factors, _ = split_commutative_parts(monomial)
    if len(comm_factors) > 0:
        if isinstance(comm_factors[0], Number):
            scalar_factor = comm_factors[0]
    if scalar_factor != 1:
        return monomial / scalar_factor, scalar_factor
    else:
        return monomial, scalar_factor


def get_support(variables, polynomial):
    """Gets the support of a polynomial.
    """
    support = []
    if is_number_type(polynomial):
        support.append([0] * len(variables))
        return support
    for monomial in polynomial.expand().as_coefficients_dict():
        tmp_support = [0] * len(variables)
        mon, _ = __separate_scalar_factor(monomial)
        symbolic_support = flatten(split_commutative_parts(mon))
        for s in symbolic_support:
            if isinstance(s, Pow):
                base = s.base
                if is_adjoint(base):
                    base = base.adjoint()
                tmp_support[variables.index(base)] = s.exp
            elif is_adjoint(s):
                tmp_support[variables.index(s.adjoint())] = 1
            elif isinstance(s, (Operator, Symbol)):
                tmp_support[variables.index(s)] = 1
        support.append(tmp_support)
    return support


def get_support_variables(polynomial):
    """Gets the support of a polynomial.
    """
    support = []
    if is_number_type(polynomial):
        return support
    for monomial in polynomial.expand().as_coefficients_dict():
        mon, _ = __separate_scalar_factor(monomial)
        symbolic_support = flatten(split_commutative_parts(mon))
        for s in symbolic_support:
            if isinstance(s, Pow):
                base = s.base
                if is_adjoint(base):
                    base = base.adjoint()
                support.append(base)
            elif is_adjoint(s):
                support.append(s.adjoint())
            elif isinstance(s, Operator):
                support.append(s)
    return support


def separate_scalar_factor(element):
    """Construct a monomial with the coefficient separated
    from an element in a polynomial.
    """
    coeff = 1.0
    monomial = S.One
    if isinstance(element, (int, float, complex)):
        coeff *= element
        return monomial, coeff
    for var in element.as_coeff_mul()[1]:
        if not (var.is_Number or var.is_imaginary):
            monomial = monomial * var
        else:
            if var.is_Number:
                coeff = float(var)
            # If not, then it is imaginary
            else:
                coeff = 1j * coeff
    coeff = float(element.as_coeff_mul()[0]) * coeff
    return monomial, coeff


def count_ncmonomials(monomials, degree):
    """Given a list of monomials, it counts those that have a certain degree,
    or less. The function is useful when certain monomials were eliminated
    from the basis.

    :param variables: The noncommutative variables making up the monomials
    :param monomials: List of monomials (the monomial basis).
    :param degree:  Maximum degree to count.

    :returns: The count of appropriate monomials.
    """
    ncmoncount = 0
    for monomial in monomials:
        if ncdegree(monomial) <= degree:
            ncmoncount += 1
        else:
            break
    return ncmoncount


def apply_substitutions(monomial, monomial_substitutions, pure=False):
    """Helper function to remove monomials from the basis."""
    if is_number_type(monomial):
        return monomial
    original_monomial = monomial
    changed = True
    if not pure:
        substitutions = monomial_substitutions
    else:
        substitutions = {}
        for lhs, rhs in monomial_substitutions.items():
            irrelevant = False
            for atom in lhs.atoms():
                if atom.is_Number:
                    continue
                if not monomial.has(atom):
                    irrelevant = True
                    break
            if not irrelevant:
                substitutions[lhs] = rhs
    while changed:
        for lhs, rhs in substitutions.items():
            monomial = fast_substitute(monomial, lhs, rhs)
        if original_monomial == monomial:
            changed = False
        original_monomial = monomial
    return monomial


def fast_substitute(monomial, old_sub, new_sub):
    """Experimental fast substitution routine that considers only restricted
    cases of noncommutative algebras. In rare cases, it fails to find a
    substitution. Use it with proper testing.

    :param monomial: The monomial with parts need to be substituted.
    :param old_sub: The part to be replaced.
    :param new_sub: The replacement.
    """
    # If monomial is number -> nothing to do
    if is_number_type(monomial):
        return monomial
    # If polynomial then apply to each term in sum
    if monomial.is_Add:
        return sum([fast_substitute(element, old_sub, new_sub) for element in
                    monomial.as_ordered_terms()])

    # Split the monomial into its factors (commutative separated)
    comm_factors, ncomm_factors = split_commutative_parts(monomial)
    # Split the part to be substituted into factors
    old_comm_factors, old_ncomm_factors = split_commutative_parts(old_sub)
    # This is a temporary hack
    # If the new_part is not a number then get its commutative parts
    if not isinstance(new_sub, (int, float, complex)):
        new_comm_factors, _ = split_commutative_parts(new_sub)
    else:
        new_comm_factors = [new_sub]
    comm_monomial = 1
    is_constant_term = False
    # If we have commutative variables then deal with them here
    if comm_factors != ():
        if len(comm_factors) == 1 and is_number_type(comm_factors[0]):
            is_constant_term = True
            comm_monomial = comm_factors[0]
        else:
            for comm_factor in comm_factors:
                comm_monomial *= comm_factor
            if old_comm_factors != ():
                comm_old_sub = 1
                for comm_factor in old_comm_factors:
                    comm_old_sub *= comm_factor
                comm_new_sub = 1
                for comm_factor in new_comm_factors:
                    comm_new_sub *= comm_factor
                # Dummy heuristic to get around retarded SymPy bug
                if isinstance(comm_old_sub, Pow):
                    # In this case, we are in trouble
                    old_base = comm_old_sub.base
                    if comm_monomial.has(old_base):
                        old_degree = comm_old_sub.exp
                        new_monomial = 1
                        match = False
                        for factor in comm_monomial.as_ordered_factors():
                            if factor.has(old_base):
                                if isinstance(factor, Pow):
                                    degree = factor.exp
                                    if degree >= old_degree:
                                        match = True
                                        new_monomial *= \
                                            old_base**(degree-old_degree) * \
                                            comm_new_sub

                                else:
                                    new_monomial *= factor
                            else:
                                new_monomial *= factor
                        if match:
                            comm_monomial = new_monomial
                else:
                    comm_monomial = comm_monomial.subs(comm_old_sub,
                                                       comm_new_sub)
    if ncomm_factors == () or old_ncomm_factors == ():
        return comm_monomial

    # old_factors = old_sub.as_ordered_factors()
    # factors = monomial.as_ordered_factors()
    # Now dealing with the non_comm factors
    new_var_list = []
    new_monomial = 1
    left_remainder = 1
    right_remainder = 1
    # ncomm_factors -- factors of the monomial
    # old_ncomm_factors -- factors of the part to be substituted
    # We move through the monomial, checking that the substituted part
    # lines up with some part of the monomial (if it does then we substitute)
    for i in range(len(ncomm_factors) - len(old_ncomm_factors) + 1):
        for j, old_ncomm_factor in enumerate(old_ncomm_factors):
            ncomm_factor = ncomm_factors[i + j]
            # If monomial factor is Symbol AND
            # [sub_part is not symbol or the same symbol then move to next mono_part]
            if isinstance(ncomm_factor, Symbol) and \
                (isinstance(old_ncomm_factor, Operator) or
                 (isinstance(old_ncomm_factor, Symbol) and
                  ncomm_factor != old_ncomm_factor)):
                left_remainder, right_remainder = 1, 1
                break
            # If monomial fact is Operator AND
            # [sub_part is not operator or not same operator then move to next mono_part]
            if isinstance(ncomm_factor, Operator) and \
                    ((isinstance(old_ncomm_factor, Operator) and
                      ncomm_factor != old_ncomm_factor) or
                     isinstance(old_ncomm_factor, Pow)):
                left_remainder, right_remainder = 1, 1
                break
            # If mono_part is daggered:
            #   if sub is not daggered or doesnt equal mono_part then move to next mono_part
            if is_adjoint(ncomm_factor):
                if not is_adjoint(old_ncomm_factor) or \
                         ncomm_factor != old_ncomm_factor:
                    left_remainder, right_remainder = 1, 1
                    break
            # If mono_part is not adjoint
            else:
                # if mono_part is not a pow  but sub_point is adjoint then move to next mono_part
                if not isinstance(ncomm_factor, Pow):
                    if is_adjoint(old_ncomm_factor):
                        left_remainder, right_remainder = 1, 1
                        break
                # if mono_part is a pow
                else:
                    # if sub_part is also a pow  then split the base and exponent
                    if isinstance(old_ncomm_factor, Pow):
                        old_base = old_ncomm_factor.base
                        old_degree = old_ncomm_factor.exp
                    else:
                        old_base = old_ncomm_factor
                        old_degree = 1
                    # If the bases don't match then move to next mono part
                    if old_base != ncomm_factor.base:
                        left_remainder, right_remainder = 1, 1
                        break
                    # If the exponent of sub part is too large then move to next mono_part
                    if old_degree > ncomm_factor.exp:
                        left_remainder, right_remainder = 1, 1
                        break
                    # If the exponent of sub_part is smaller than the monomial part
                    if old_degree < ncomm_factor.exp:
                        # if we are not at the last factor of sub_part
                        if j != len(old_ncomm_factors) - 1:
                            # neither last or first factor then sub doesn't fit and break
                            if j != 0:
                                left_remainder, right_remainder = 1, 1
                                break
                            # We must now be the left-most factor of sub
                            # So we can potentially sub in with some remainer to the left
                            else:
                                left_remainder = old_base ** (
                                    ncomm_factor.exp - old_degree)
                        # if we are the last factor then we can sub in with remainer to the right
                        else:
                            right_remainder = old_base ** (
                                ncomm_factor.exp - old_degree)
        # This else is executed after the inner-loop has been exhausted (no break)
        # This should only happen if we find a match for the substitution
        else:
            new_monomial = 1
            # All parts of mono up to the substituted bit
            for var in new_var_list:
                new_monomial *= var
            new_monomial *= left_remainder * new_sub * right_remainder
            for j in range(i + len(old_ncomm_factors), len(ncomm_factors)):
                new_monomial *= ncomm_factors[j]
            new_monomial *= comm_monomial
            break
        # Each time we broke out of the inner loop we found another part
        # of the monomial that will not be substituted. Add it to the new_var_list
        new_var_list.append(ncomm_factors[i])
    # We broke out of the outer loop
    else:
        if not is_constant_term and comm_factors != ():
            new_monomial = comm_monomial
            for factor in ncomm_factors:
                new_monomial *= factor
        else:
            return monomial
    if not isinstance(new_sub, (float, int, complex)) and new_sub.is_Add:
        return expand(new_monomial)
    else:
        return new_monomial


def generate_variables(name, n_vars=1, hermitian=None, commutative=True):
    """Generates a number of commutative or noncommutative variables

    :param name: The prefix in the symbolic representation of the noncommuting
                 variables. This will be suffixed by a number from 0 to
                 n_vars-1 if n_vars > 1.
    :type name: str.
    :param n_vars: The number of variables.
    :type n_vars: int.
    :param hermitian: Optional parameter to request Hermitian variables .
    :type hermitian: bool.
    :param commutative: Optional parameter to request commutative variables.
                        Commutative variables are Hermitian by default.
    :type commutative: bool.

    :returns: list of :class:`sympy.physics.quantum.operator.Operator` or
              :class:`sympy.physics.quantum.operator.HermitianOperator`
              variables or `sympy.Symbol`

    :Example:

    >>> generate_variables('y', 2, commutative=True)
    ￼[y0, y1]
    """

    variables = []
    for i in range(n_vars):
        if n_vars > 1:
            var_name = '%s%s' % (name, i)
        else:
            var_name = '%s' % name
        if commutative:
            if hermitian is None or hermitian:
                variables.append(Symbol(var_name, real=True))
            else:
                variables.append(Symbol(var_name, complex=True))
        elif hermitian is not None and hermitian:
            variables.append(HermitianOperator(var_name))
        else:
            variables.append(Operator(var_name))
    return variables


def generate_operators(name, n_vars=1, hermitian=None, commutative=False):
    """Generates a number of commutative or noncommutative operators

    :param name: The prefix in the symbolic representation of the noncommuting
                 variables. This will be suffixed by a number from 0 to
                 n_vars-1 if n_vars > 1.
    :type name: str.
    :param n_vars: The number of variables.
    :type n_vars: int.
    :param hermitian: Optional parameter to request Hermitian variables .
    :type hermitian: bool.
    :param commutative: Optional parameter to request commutative variables.
                        Commutative variables are Hermitian by default.
    :type commutative: bool.

    :returns: list of :class:`sympy.physics.quantum.operator.Operator` or
              :class:`sympy.physics.quantum.operator.HermitianOperator`
              variables

    :Example:

    >>> generate_variables('y', 2, commutative=True)
    ￼[y0, y1]
    """

    variables = []
    for i in range(n_vars):
        if n_vars > 1:
            var_name = '%s%s' % (name, i)
        else:
            var_name = '%s' % name
        if hermitian is not None and hermitian:
            variables.append(HermitianOperator(var_name))
        else:
            variables.append(Operator(var_name))
        variables[-1].is_commutative = commutative
    return variables


def get_monomials(variables, degree):
    """Generates all noncommutative monomials up to a degree

    :param variables: The noncommutative variables to generate monomials from
    :type variables: list of :class:`sympy.physics.quantum.operator.Operator`
                     or
                     :class:`sympy.physics.quantum.operator.HermitianOperator`.
    :param degree: The maximum degree.
    :type degree: int.

    :returns: list of monomials.
    """
    if degree == -1:
        return []
    if not variables:
        return [S.One]
    else:
        _variables = variables[:]
        _variables.insert(0, 1)
        ncmonomials = [S.One]
        ncmonomials.extend(var for var in variables)
        for var in variables:
            if not is_hermitian(var):
                ncmonomials.append(var.adjoint())
        for _ in range(1, degree):
            temp = []
            for var in _variables:
                for new_var in ncmonomials:
                    temp.append(var * new_var)
                    if var != 1 and not is_hermitian(var):
                        temp.append(var.adjoint() * new_var)
            ncmonomials = unique(temp[:])
        return ncmonomials


def ncdegree(polynomial):
    """Returns the degree of a noncommutative polynomial.

    :param polynomial: Polynomial of noncommutive variables.
    :type polynomial: :class:`sympy.core.expr.Expr`.

    :returns: int -- the degree of the polynomial.
    """
    degree = 0
    if is_number_type(polynomial):
        return degree
    polynomial = polynomial.expand()
    for monomial in polynomial.as_coefficients_dict():
        subdegree = 0
        for variable in monomial.as_coeff_mul()[1]:
            if isinstance(variable, Pow):
                subdegree += variable.exp
            elif not isinstance(variable, Number) and variable != I:
                subdegree += 1
        if subdegree > degree:
            degree = subdegree
    return degree


def iscomplex(polynomial):
    """Returns whether the polynomial has complex coefficients

    :param polynomial: Polynomial of noncommutive variables.
    :type polynomial: :class:`sympy.core.expr.Expr`.

    :returns: bool -- whether there is a complex coefficient.
    """
    if isinstance(polynomial, (int, float)):
        return False
    if isinstance(polynomial, complex):
        return True
    polynomial = polynomial.expand()
    for monomial in polynomial.as_coefficients_dict():
        for variable in monomial.as_coeff_mul()[1]:
            if isinstance(variable, complex) or variable == I:
                return True
    return False


def get_all_monomials(variables, extramonomials, substitutions, degree,
                      removesubstitutions=True):
    """Return the monomials of a certain degree.
    """
    monomials = get_monomials(variables, degree)
    if extramonomials is not None:
        monomials.extend(extramonomials)
    if removesubstitutions and substitutions is not None:
        monomials = [monomial for monomial in monomials if monomial not
                     in substitutions]
        monomials = [remove_scalar_factor(apply_substitutions(monomial,
                                                              substitutions))
                     for monomial in monomials]
    monomials = unique(monomials)
    return monomials


def pick_monomials_up_to_degree(monomials, degree):
    """Collect monomials up to a given degree.
    """
    ordered_monomials = []
    if degree >= 0:
        ordered_monomials.append(S.One)
    for deg in range(1, degree + 1):
        ordered_monomials.extend(pick_monomials_of_degree(monomials, deg))
    return ordered_monomials


def pick_monomials_of_degree(monomials, degree):
    """Collect all monomials up of a given degree.
    """
    selected_monomials = []
    for monomial in monomials:
        if ncdegree(monomial) == degree:
            selected_monomials.append(monomial)
    return selected_monomials


def convert_monomial_to_string(monomial):
    monomial_str = ('%s' % monomial)
    monomial_str = monomial_str.replace('Dagger(', '')
    monomial_str = monomial_str.replace(')', 'T')
    monomial_str = monomial_str.replace('**', '^')
    return monomial_str


def save_monomial_index(filename, monomial_index):
    """Save a monomial dictionary for debugging purposes.

    :param filename: The name of the file to save to.
    :type filename: str.
    :param monomial_index: The monomial index of the SDP relaxation.
    :type monomial_index: dict of :class:`sympy.core.expr.Expr`.

    """
    monomial_translation = [''] * (len(monomial_index) + 1)
    for key, k in monomial_index.items():
        monomial_translation[k] = convert_monomial_to_string(key)
    file_ = open(filename, 'w')
    for k in range(len(monomial_translation)):
        file_.write('%s %s\n' % (k, monomial_translation[k]))
    file_.close()


def unique(seq):
    """Helper function to include only unique monomials in a basis."""
    seen = {}
    result = []
    for item in seq:
        marker = item
        if marker in seen:
            continue
        seen[marker] = 1
        result.append(item)
    return result


def build_permutation_matrix(permutation):
    """Build a permutation matrix for a permutation.
    """
    matrix = lil_matrix((len(permutation), len(permutation)))
    column = 0
    for row in permutation:
        matrix[row, column] = 1
        column += 1
    return matrix


def convert_relational(relational):
    """Convert all inequalities to >=0 form.
    """
    rel = relational.rel_op
    if rel in ['==', '>=', '>']:
        return relational.lhs-relational.rhs
    elif rel in ['<=', '<']:
        return relational.rhs-relational.lhs
    else:
        raise Exception("The relational operation ' + rel + ' is not "
                        "implemented!")


def find_variable_set(variable_sets, polynomial):
    if not isinstance(variable_sets[0], list):
        return 0
    support = set(get_support_variables(polynomial))
    for i, variable_set in enumerate(variable_sets):
        if len(support-set(variable_set)) == 0:
            return i
    return -1


def moment_of_entry(pos, monomials, ineq, substitutions):
    row = pos[0]
    column = pos[1]
    if isinstance(ineq, str):
        return row, column, ineq
    else:
        return row, column, simplify_polynomial(monomials[row].adjoint() *
                                                ineq * monomials[column],
                                                substitutions)


def assemble_monomial_and_do_substitutions(arg, monomialsA, monomialsB, ppt,
                                           substitutions,
                                           pure_substitution_rules):
    rowA = arg[0]
    columnA = arg[1]
    rowB = arg[2]
    columnB = arg[3]
    if (not ppt) or (columnB >= rowB):
        monomial = monomialsA[rowA].adjoint() * monomialsA[columnA] * \
            monomialsB[rowB].adjoint() * monomialsB[columnB]
    else:
        monomial = monomialsA[rowA].adjoint() * monomialsA[columnA] * \
            monomialsB[columnB].adjoint() * monomialsB[rowB]
        # Apply the substitutions if any
    monomial = apply_substitutions(monomial, substitutions,
                                   pure_substitution_rules)

    return columnA, rowB, columnB, monomial


def split_commutative_parts(e):
    if e.is_Mul:
        args = e.args
    else:
        args = (e,)
    for i, m in enumerate(args):
        if not m.is_commutative:
            break
    else:
        i += 1
    return args[:i], args[i:]


def is_number_type(exp):
    return isinstance(exp, (int, float, complex)) or exp.is_number


def is_adjoint(exp):
    return isinstance(exp, (adjoint, conjugate))


def is_hermitian(exp):
    return exp.is_hermitian or (exp.is_hermitian is None and exp.is_complex)


def flip_sign(exp):
    if isinstance(exp, str):
        if not exp.startswith("+") and not exp.startswith("+"):
            tmp = "+" + exp
        else:
            tmp = exp
        tmp = tmp.replace("+", "p")
        tmp = tmp.replace("-", "+")
        tmp = tmp.replace("p", "-")
        return tmp
    else:
        return -exp
