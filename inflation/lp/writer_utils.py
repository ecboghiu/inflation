from inflation import InflationLP


def write_to_lp(problem: InflationLP,
                filename: str) -> None:
    """Export the problem to a file in .lp format.

    Parameters
    ----------
    problem : inflation.InflationLP
        The problem to write to the file.
    filename : str
        The file to write to.
    """
