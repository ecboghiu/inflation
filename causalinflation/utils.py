"""
This file contains auxiliary functions of general purpose
@authors: Emanuel-Cristian Boghiu, Elie Wolfe and Alejandro Pozas-Kerstjens
"""
def blank_tqdm(*args, **kwargs):
    """Placeholder in case the tqdm library is not installed (see
    https://tqdm.github.io). If code is set to print a tqdm progress bar and
    tqdm is not installed, this just prints the ``desc`` argument.
    """
    try:
        if not kwargs["disable"]:
            print(kwargs["desc"])
    except KeyError:
        pass
    return args[0]
