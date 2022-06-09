import numpy as np
from itertools import product
import qutip as qt

def probfunc2array(func, noise, outcomes_per_party, settings_per_party):
    p = np.zeros([*outcomes_per_party, *settings_per_party])
    for a,b,c,x,y,z in product(*[range(i) for i in [*outcomes_per_party, *settings_per_party]]):
        p[a,b,c,x,y,z] = func(a,b,c,x,y,z,noise)
    return p

def target_distribution_gen(name, parameter1, parameter2):
    """ 
    This function has been copied from: 
    https://github.com/tkrivachy/neural-network-for-nonlocality-in-networks/blob/master/sample_code/targets.py
    parameter1 is usually a parameter of distribution (not always relevant). parameter2 is usually noise."""

    if name == "Fritz-visibility":
        """ parameter2 is the visibility"""
        ids = np.zeros((4, 4, 4)).astype(str)
        p = np.zeros((4, 4, 4))
        for i, j, k, l, m, n in product('01', repeat=6):
            a = int(i + j, 2)
            b = int(k + l, 2)
            c = int(m + n, 2)
            temp0 = [str(a), str(b), str(c)]
            temp = [a, b, c]
            ids[temp[0], temp[1], temp[2]] = ''.join(temp0)
            cspi8 = 1 / (2 * np.sqrt(2))
            cos2pi8 = (2 + np.sqrt(2)) / 4
            sin2pi8 = (2 - np.sqrt(2)) / 4
            if m == j and n == l:
                if n == '0':
                    if i == k:
                        p[temp[0], temp[1], temp[2]] = (1 - parameter2 * (cos2pi8 - sin2pi8)) / 16
                    if i != k:
                        p[temp[0], temp[1], temp[2]] = (1 + parameter2 * (cos2pi8 - sin2pi8)) / 16
                if n == '1':
                    if m == '0':
                        if i == k:
                            p[temp[0], temp[1], temp[2]] = 1 / 16 - cspi8 * parameter2 / 8
                        if i != k:
                            p[temp[0], temp[1], temp[2]] = 1 / 16 + cspi8 * parameter2 / 8
                    if m == '1':
                        if i == k:
                            p[temp[0], temp[1], temp[2]] = 1 / 16 + cspi8 * parameter2 / 8
                        if i != k:
                            p[temp[0], temp[1], temp[2]] = 1 / 16 - cspi8 * parameter2 / 8
        p = p.flatten()
        ids = ids.flatten()

    if name == "Renou-visibility":
        """ Info: If param_c >~ 0.886 or <~0.464, there is no classical 3-local model."""
        """ In terms of c**2: above 0.785 or below 0.215 no classical 3-local model."""
        c = parameter1
        v = parameter2
        p = np.array([
            -(-1 + v) ** 3 / 64., -((-1 + v) * (1 + v) ** 2) / 64., ((-1 + v) ** 2 * (1 + v)) / 64.,
            ((-1 + v) ** 2 * (1 + v)) / 64., -((-1 + v) * (1 + v) ** 2) / 64., -((-1 + v) * (1 + v) ** 2) / 64.,
            ((1 + v) * (1 + (-2 + 4 * c ** 2) * v + v ** 2)) / 64.,
            ((1 + v) * (1 + (2 - 4 * c ** 2) * v + v ** 2)) / 64., ((-1 + v) ** 2 * (1 + v)) / 64.,
            ((1 + v) * (1 + (2 - 4 * c ** 2) * v + v ** 2)) / 64.,
            (1 + (1 - 2 * c ** 2) ** 2 * v - (1 - 2 * c ** 2) ** 2 * v ** 2 - v ** 3) / 64.,
            (1 + v - 4 * c ** 4 * v + (-1 + 4 * c ** 4) * v ** 2 - v ** 3) / 64., ((-1 + v) ** 2 * (1 + v)) / 64.,
            ((1 + v) * (1 + (-2 + 4 * c ** 2) * v + v ** 2)) / 64.,
            (1 + (-3 + 8 * c ** 2 - 4 * c ** 4) * v + (3 - 8 * c ** 2 + 4 * c ** 4) * v ** 2 - v ** 3) / 64.,
            (1 + (1 - 2 * c ** 2) ** 2 * v - (1 - 2 * c ** 2) ** 2 * v ** 2 - v ** 3) / 64.,
            -((-1 + v) * (1 + v) ** 2) / 64., -((-1 + v) * (1 + v) ** 2) / 64.,
            ((1 + v) * (1 + (2 - 4 * c ** 2) * v + v ** 2)) / 64.,
            ((1 + v) * (1 + (-2 + 4 * c ** 2) * v + v ** 2)) / 64.,
            -((-1 + v) * (1 + v) ** 2) / 64., -(-1 + v) ** 3 / 64., ((-1 + v) ** 2 * (1 + v)) / 64.,
            ((-1 + v) ** 2 * (1 + v)) / 64., ((1 + v) * (1 + (-2 + 4 * c ** 2) * v + v ** 2)) / 64.,
            ((-1 + v) ** 2 * (1 + v)) / 64.,
            (1 + (1 - 2 * c ** 2) ** 2 * v - (1 - 2 * c ** 2) ** 2 * v ** 2 - v ** 3) / 64.,
            (1 + (-3 + 8 * c ** 2 - 4 * c ** 4) * v + (3 - 8 * c ** 2 + 4 * c ** 4) * v ** 2 - v ** 3) / 64.,
            ((1 + v) * (1 + (2 - 4 * c ** 2) * v + v ** 2)) / 64., ((-1 + v) ** 2 * (1 + v)) / 64.,
            (1 + v - 4 * c ** 4 * v + (-1 + 4 * c ** 4) * v ** 2 - v ** 3) / 64.,
            (1 + (1 - 2 * c ** 2) ** 2 * v - (1 - 2 * c ** 2) ** 2 * v ** 2 - v ** 3) / 64.,
            ((-1 + v) ** 2 * (1 + v)) / 64., ((1 + v) * (1 + (-2 + 4 * c ** 2) * v + v ** 2)) / 64.,
            (1 + (1 - 2 * c ** 2) ** 2 * v - (1 - 2 * c ** 2) ** 2 * v ** 2 - v ** 3) / 64.,
            (1 + (-3 + 8 * c ** 2 - 4 * c ** 4) * v + (3 - 8 * c ** 2 + 4 * c ** 4) * v ** 2 - v ** 3) / 64.,
            ((1 + v) * (1 + (2 - 4 * c ** 2) * v + v ** 2)) / 64., ((-1 + v) ** 2 * (1 + v)) / 64.,
            (1 + (1 - 2 * c ** 2) ** 2 * v - (1 - 2 * c ** 2) ** 2 * v ** 2 - v ** 3) / 64.,
            (1 + v - 4 * c ** 4 * v + (-1 + 4 * c ** 4) * v ** 2 - v ** 3) / 64.,
            (1 + (1 - 2 * c ** 2) ** 2 * v - (1 - 2 * c ** 2) ** 2 * v ** 2 - v ** 3) / 64.,
            (1 + (1 - 2 * c ** 2) ** 2 * v - (1 - 2 * c ** 2) ** 2 * v ** 2 - v ** 3) / 64., (
                        1 + 3 * (1 - 2 * c ** 2) ** 2 * v + 3 * (1 - 2 * c ** 2) ** 2 * v ** 2 + (
                            1 + 16 * c ** 3 * np.sqrt(1 - c ** 2) - 16 * c ** 5 * np.sqrt(1 - c ** 2)) * v ** 3) / 64.,
            (1 - (1 - 2 * c ** 2) ** 2 * v - (1 - 2 * c ** 2) ** 2 * v ** 2 + (
                        1 - 16 * c ** 3 * np.sqrt(1 - c ** 2) + 16 * c ** 5 * np.sqrt(1 - c ** 2)) * v ** 3) / 64.,
            (1 + v - 4 * c ** 4 * v + (-1 + 4 * c ** 4) * v ** 2 - v ** 3) / 64.,
            (1 + (-3 + 8 * c ** 2 - 4 * c ** 4) * v + (3 - 8 * c ** 2 + 4 * c ** 4) * v ** 2 - v ** 3) / 64., (
                        1 - (1 - 2 * c ** 2) ** 2 * v - (1 - 2 * c ** 2) ** 2 * v ** 2 + (
                            1 - 16 * c ** 3 * np.sqrt(1 - c ** 2) + 16 * c ** 5 * np.sqrt(1 - c ** 2)) * v ** 3) / 64.,
            (1 - (1 - 2 * c ** 2) ** 2 * v - (1 - 2 * c ** 2) ** 2 * v ** 2 + (
                        1 + 16 * c ** 3 * np.sqrt(1 - c ** 2) - 16 * c ** 5 * np.sqrt(1 - c ** 2)) * v ** 3) / 64.,
            ((-1 + v) ** 2 * (1 + v)) / 64., ((1 + v) * (1 + (2 - 4 * c ** 2) * v + v ** 2)) / 64.,
            (1 + v - 4 * c ** 4 * v + (-1 + 4 * c ** 4) * v ** 2 - v ** 3) / 64.,
            (1 + (1 - 2 * c ** 2) ** 2 * v - (1 - 2 * c ** 2) ** 2 * v ** 2 - v ** 3) / 64.,
            ((1 + v) * (1 + (-2 + 4 * c ** 2) * v + v ** 2)) / 64., ((-1 + v) ** 2 * (1 + v)) / 64.,
            (1 + (-3 + 8 * c ** 2 - 4 * c ** 4) * v + (3 - 8 * c ** 2 + 4 * c ** 4) * v ** 2 - v ** 3) / 64.,
            (1 + (1 - 2 * c ** 2) ** 2 * v - (1 - 2 * c ** 2) ** 2 * v ** 2 - v ** 3) / 64.,
            (1 + (-3 + 8 * c ** 2 - 4 * c ** 4) * v + (3 - 8 * c ** 2 + 4 * c ** 4) * v ** 2 - v ** 3) / 64.,
            (1 + v - 4 * c ** 4 * v + (-1 + 4 * c ** 4) * v ** 2 - v ** 3) / 64., (
                        1 - (1 - 2 * c ** 2) ** 2 * v - (1 - 2 * c ** 2) ** 2 * v ** 2 + (
                            1 - 16 * c ** 3 * np.sqrt(1 - c ** 2) + 16 * c ** 5 * np.sqrt(1 - c ** 2)) * v ** 3) / 64.,
            (1 - (1 - 2 * c ** 2) ** 2 * v - (1 - 2 * c ** 2) ** 2 * v ** 2 + (
                        1 + 16 * c ** 3 * np.sqrt(1 - c ** 2) - 16 * c ** 5 * np.sqrt(1 - c ** 2)) * v ** 3) / 64.,
            (1 + (1 - 2 * c ** 2) ** 2 * v - (1 - 2 * c ** 2) ** 2 * v ** 2 - v ** 3) / 64.,
            (1 + (1 - 2 * c ** 2) ** 2 * v - (1 - 2 * c ** 2) ** 2 * v ** 2 - v ** 3) / 64., (
                        1 - (1 - 2 * c ** 2) ** 2 * v - (1 - 2 * c ** 2) ** 2 * v ** 2 + (
                            1 + 16 * c ** 3 * np.sqrt(1 - c ** 2) - 16 * c ** 5 * np.sqrt(1 - c ** 2)) * v ** 3) / 64.,
            (1 + 3 * (1 - 2 * c ** 2) ** 2 * v + 3 * (1 - 2 * c ** 2) ** 2 * v ** 2 + (
                        1 - 16 * c ** 3 * np.sqrt(1 - c ** 2) + 16 * c ** 5 * np.sqrt(1 - c ** 2)) * v ** 3) / 64.
        ])
        ids = np.array([
            "000", "001", "002", "003", "010", "011", "012", "013", "020", "021", \
            "022", "023", "030", "031", "032", "033", "100", "101", "102", "103", \
            "110", "111", "112", "113", "120", "121", "122", "123", "130", "131", \
            "132", "133", "200", "201", "202", "203", "210", "211", "212", "213", \
            "220", "221", "222", "223", "230", "231", "232", "233", "300", "301", \
            "302", "303", "310", "311", "312", "313", "320", "321", "322", "323", \
            "330", "331", "332", "333"
        ])

    if name == "Renou-localnoise":
        """ Info: If param_c >~ 0.886 or <~0.464, there is no classical 3-local model."""
        """ In terms of c**2: above 0.785 or below 0.215 no classical 3-local model."""
        param_c = parameter1
        param_s = np.np.sqrt(1 - param_c ** 2)

        # the si and ci functions
        param2_c = {'2': param_c, '3': param_s}
        param2_s = {'2': param_s, '3': -1 * param_c}

        # First create noiseless Salman distribution.
        ids = np.zeros((4, 4, 4)).astype(str)
        p = np.zeros((4, 4, 4))
        for a, b, c in product('0123', repeat=3):
            temp0 = [a, b, c]
            temp = [int(item) for item in temp0]
            ids[temp[0], temp[1], temp[2]] = ''.join(temp0)

            # p(12vi) et al.
            if (a == '0' and b == '1' and c == '2') or (a == '1' and b == '0' and c == '3'):
                p[temp[0], temp[1], temp[2]] = 1 / 8 * param_c ** 2
            elif (c == '0' and a == '1' and b == '2') or (c == '1' and a == '0' and b == '3'):
                p[temp[0], temp[1], temp[2]] = 1 / 8 * param_c ** 2
            elif (b == '0' and c == '1' and a == '2') or (b == '1' and c == '0' and a == '3'):
                p[temp[0], temp[1], temp[2]] = 1 / 8 * param_c ** 2

            elif (a == '0' and b == '1' and c == '3') or (a == '1' and b == '0' and c == '2'):
                p[temp[0], temp[1], temp[2]] = 1 / 8 * param_s ** 2
            elif (c == '0' and a == '1' and b == '3') or (c == '1' and a == '0' and b == '2'):
                p[temp[0], temp[1], temp[2]] = 1 / 8 * param_s ** 2
            elif (b == '0' and c == '1' and a == '3') or (b == '1' and c == '0' and a == '2'):
                p[temp[0], temp[1], temp[2]] = 1 / 8 * param_s ** 2

            # p(vi vj vk) et al.
            elif a in '23' and b in '23' and c in '23':
                p[temp[0], temp[1], temp[2]] = 1 / 8 * (
                            param2_c[a] * param2_c[b] * param2_c[c] + param2_s[a] * param2_s[b] * param2_s[c]) ** 2
            else:
                p[temp[0], temp[1], temp[2]] = 0

        # Let's add local noise.
        new_values = np.zeros_like(p)
        for a, b, c in product('0123', repeat=3):
            temp0 = [a, b, c]
            temp = [int(item) for item in temp0]
            new_values[temp[0], temp[1], temp[2]] = (
                    parameter2 ** 3 * p[temp[0], temp[1], temp[2]] +
                    parameter2 ** 2 * (1 - parameter2) * 1 / 4 * (
                                np.sum(p, axis=2)[temp[0], temp[1]] + np.sum(p, axis=0)[temp[1], temp[2]] +
                                np.sum(p, axis=1)[temp[0], temp[2]]) +
                    parameter2 * (1 - parameter2) ** 2 * 1 / 16 * (
                                np.sum(p, axis=(1, 2))[temp[0]] + np.sum(p, axis=(0, 2))[temp[1]] +
                                np.sum(p, axis=(0, 1))[temp[2]]) +
                    (1 - parameter2) ** 3 * 1 / 64
            )
        p = new_values.flatten()
        ids = ids.flatten()

    if name == "elegant-visibility":
        """ Recreating the elegant distribution with visibility v (parameter2) in each singlet. """
        ids = np.zeros((4, 4, 4)).astype(str)
        for a, b, c in product('0123', repeat=3):
            temp0 = [a, b, c]
            temp = [int(item) for item in temp0]
            ids[temp[0], temp[1], temp[2]] = ''.join(temp0)
        ids = ids.flatten()
        p = np.array([1 / 256 * (4 + 9 * parameter2 + 9 * parameter2 ** 2 + 3 * parameter2 ** 3),
                      1 / 256 * (4 + parameter2 - 3 * parameter2 ** 2 - parameter2 ** 3),
                      1 / 256 * (4 + parameter2 - 3 * parameter2 ** 2 - parameter2 ** 3),
                      1 / 256 * (4 + parameter2 - 3 * parameter2 ** 2 - parameter2 ** 3),
                      1 / 256 * (4 + parameter2 - 3 * parameter2 ** 2 - parameter2 ** 3),
                      1 / 256 * (4 + parameter2 - 3 * parameter2 ** 2 - parameter2 ** 3),
                      1 / 256 * (4 - 3 * parameter2 + 3 * parameter2 ** 2 + parameter2 ** 3),
                      1 / 256 * (4 - 3 * parameter2 + 3 * parameter2 ** 2 + parameter2 ** 3),
                      1 / 256 * (4 + parameter2 - 3 * parameter2 ** 2 - parameter2 ** 3),
                      1 / 256 * (4 - 3 * parameter2 + 3 * parameter2 ** 2 + parameter2 ** 3),
                      1 / 256 * (4 + parameter2 - 3 * parameter2 ** 2 - parameter2 ** 3),
                      1 / 256 * (4 - 3 * parameter2 + 3 * parameter2 ** 2 + parameter2 ** 3),
                      1 / 256 * (4 + parameter2 - 3 * parameter2 ** 2 - parameter2 ** 3),
                      1 / 256 * (4 - 3 * parameter2 + 3 * parameter2 ** 2 + parameter2 ** 3),
                      1 / 256 * (4 - 3 * parameter2 + 3 * parameter2 ** 2 + parameter2 ** 3),
                      1 / 256 * (4 + parameter2 - 3 * parameter2 ** 2 - parameter2 ** 3),
                      1 / 256 * (4 + parameter2 - 3 * parameter2 ** 2 - parameter2 ** 3),
                      1 / 256 * (4 + parameter2 - 3 * parameter2 ** 2 - parameter2 ** 3),
                      1 / 256 * (4 - 3 * parameter2 + 3 * parameter2 ** 2 + parameter2 ** 3),
                      1 / 256 * (4 - 3 * parameter2 + 3 * parameter2 ** 2 + parameter2 ** 3),
                      1 / 256 * (4 + parameter2 - 3 * parameter2 ** 2 - parameter2 ** 3),
                      1 / 256 * (4 + 9 * parameter2 + 9 * parameter2 ** 2 + 3 * parameter2 ** 3),
                      1 / 256 * (4 + parameter2 - 3 * parameter2 ** 2 - parameter2 ** 3),
                      1 / 256 * (4 + parameter2 - 3 * parameter2 ** 2 - parameter2 ** 3),
                      1 / 256 * (4 - 3 * parameter2 + 3 * parameter2 ** 2 + parameter2 ** 3),
                      1 / 256 * (4 + parameter2 - 3 * parameter2 ** 2 - parameter2 ** 3),
                      1 / 256 * (4 + parameter2 - 3 * parameter2 ** 2 - parameter2 ** 3),
                      1 / 256 * (4 - 3 * parameter2 + 3 * parameter2 ** 2 + parameter2 ** 3),
                      1 / 256 * (4 - 3 * parameter2 + 3 * parameter2 ** 2 + parameter2 ** 3),
                      1 / 256 * (4 + parameter2 - 3 * parameter2 ** 2 - parameter2 ** 3),
                      1 / 256 * (4 - 3 * parameter2 + 3 * parameter2 ** 2 + parameter2 ** 3),
                      1 / 256 * (4 + parameter2 - 3 * parameter2 ** 2 - parameter2 ** 3),
                      1 / 256 * (4 + parameter2 - 3 * parameter2 ** 2 - parameter2 ** 3),
                      1 / 256 * (4 - 3 * parameter2 + 3 * parameter2 ** 2 + parameter2 ** 3),
                      1 / 256 * (4 + parameter2 - 3 * parameter2 ** 2 - parameter2 ** 3),
                      1 / 256 * (4 - 3 * parameter2 + 3 * parameter2 ** 2 + parameter2 ** 3),
                      1 / 256 * (4 - 3 * parameter2 + 3 * parameter2 ** 2 + parameter2 ** 3),
                      1 / 256 * (4 + parameter2 - 3 * parameter2 ** 2 - parameter2 ** 3),
                      1 / 256 * (4 + parameter2 - 3 * parameter2 ** 2 - parameter2 ** 3),
                      1 / 256 * (4 - 3 * parameter2 + 3 * parameter2 ** 2 + parameter2 ** 3),
                      1 / 256 * (4 + parameter2 - 3 * parameter2 ** 2 - parameter2 ** 3),
                      1 / 256 * (4 + parameter2 - 3 * parameter2 ** 2 - parameter2 ** 3),
                      1 / 256 * (4 + 9 * parameter2 + 9 * parameter2 ** 2 + 3 * parameter2 ** 3),
                      1 / 256 * (4 + parameter2 - 3 * parameter2 ** 2 - parameter2 ** 3),
                      1 / 256 * (4 - 3 * parameter2 + 3 * parameter2 ** 2 + parameter2 ** 3),
                      1 / 256 * (4 - 3 * parameter2 + 3 * parameter2 ** 2 + parameter2 ** 3),
                      1 / 256 * (4 + parameter2 - 3 * parameter2 ** 2 - parameter2 ** 3),
                      1 / 256 * (4 + parameter2 - 3 * parameter2 ** 2 - parameter2 ** 3),
                      1 / 256 * (4 + parameter2 - 3 * parameter2 ** 2 - parameter2 ** 3),
                      1 / 256 * (4 - 3 * parameter2 + 3 * parameter2 ** 2 + parameter2 ** 3),
                      1 / 256 * (4 - 3 * parameter2 + 3 * parameter2 ** 2 + parameter2 ** 3),
                      1 / 256 * (4 + parameter2 - 3 * parameter2 ** 2 - parameter2 ** 3),
                      1 / 256 * (4 - 3 * parameter2 + 3 * parameter2 ** 2 + parameter2 ** 3),
                      1 / 256 * (4 + parameter2 - 3 * parameter2 ** 2 - parameter2 ** 3),
                      1 / 256 * (4 - 3 * parameter2 + 3 * parameter2 ** 2 + parameter2 ** 3),
                      1 / 256 * (4 + parameter2 - 3 * parameter2 ** 2 - parameter2 ** 3),
                      1 / 256 * (4 - 3 * parameter2 + 3 * parameter2 ** 2 + parameter2 ** 3),
                      1 / 256 * (4 - 3 * parameter2 + 3 * parameter2 ** 2 + parameter2 ** 3),
                      1 / 256 * (4 + parameter2 - 3 * parameter2 ** 2 - parameter2 ** 3),
                      1 / 256 * (4 + parameter2 - 3 * parameter2 ** 2 - parameter2 ** 3),
                      1 / 256 * (4 + parameter2 - 3 * parameter2 ** 2 - parameter2 ** 3),
                      1 / 256 * (4 + parameter2 - 3 * parameter2 ** 2 - parameter2 ** 3),
                      1 / 256 * (4 + parameter2 - 3 * parameter2 ** 2 - parameter2 ** 3),
                      1 / 256 * (4 + 9 * parameter2 + 9 * parameter2 ** 2 + 3 * parameter2 ** 3)])

    if name == "elegant-localnoise":
        """ Recreating the elegant distribution, with each detector having 1-v (1-parameter2) chance of outputting a uniformly random output, and v chance of working properly. """
        ids = np.zeros((4, 4, 4)).astype(str)
        p = np.zeros((4, 4, 4))
        for a, b, c in product('0123', repeat=3):
            temp0 = [a, b, c]
            temp = [int(item) for item in temp0]
            ids[temp[0], temp[1], temp[2]] = ''.join(temp0)
            if (a == b) and (b == c):
                p[temp[0], temp[1], temp[2]] = 25 / 256
            elif (a == b and b != c) or (b == c and c != a) or (c == a and a != b):
                p[temp[0], temp[1], temp[2]] = 1 / 256
            else:
                p[temp[0], temp[1], temp[2]] = 5 / 256

        # Let's add local noise.
        new_values = np.zeros_like(p)
        for a, b, c in product('0123', repeat=3):
            temp0 = [a, b, c]
            temp = [int(item) for item in temp0]
            new_values[temp[0], temp[1], temp[2]] = (
                    parameter2 ** 3 * p[temp[0], temp[1], temp[2]] +
                    parameter2 ** 2 * (1 - parameter2) * 1 / 4 * (
                                np.sum(p, axis=2)[temp[0], temp[1]] + np.sum(p, axis=0)[temp[1], temp[2]] +
                                np.sum(p, axis=1)[temp[0], temp[2]]) +
                    parameter2 * (1 - parameter2) ** 2 * 1 / 16 * (
                                np.sum(p, axis=(1, 2))[temp[0]] + np.sum(p, axis=(0, 2))[temp[1]] +
                                np.sum(p, axis=(0, 1))[temp[2]]) +
                    (1 - parameter2) ** 3 * 1 / 64
            )

        p = new_values.flatten()
        ids = ids.flatten()
    assert (np.abs(np.sum(p) - 1.0) < (1E-6)), "Improperly normalized p!"
    return np.reshape(p, [4, 4, 4])


def P_GHZ(a, b, c, x, y, z, v):
    def p_ideal(a, b, c, x, y, z):
        if a == b == c:
            return 1 / 2
        else:
            return 0.0

    # prob_array = np.ones([2, 2, 2]) * (1-v)/8
    # prob_array[0][0][0] += 0.5*v
    # prob_array[1][1][1] += 0.5*v
    return v * p_ideal(a, b, c, x, y, z) + (1 - v) / 8

def P_GHZ_array(visibility=1):
    return probfunc2array(P_GHZ, visibility, [2, 2, 2], [1, 1, 1])

def P_W(a, b, c, x, y, z, v):
    def p_ideal(a, b, c, x, y, z):
        if a + b + c == 1:
            return 1 / 3
        else:
            return 0.0

    return v * p_ideal(a, b, c, x, y, z) + (1 - v) / 8

def P_W_array(visibility=1):
    return probfunc2array(P_W, visibility, [2, 2, 2], [1, 1, 1])

def P_Mermin(a, b, c, x, y, z, v):
    def p_ideal(a, b, c, x, y, z):
        if (x + y + z) % 2 == 0:
            return 1 / 8
        elif x + y + z == 1:
            return (1 + (-1) ** (a + b + c)) / 8
        elif x + y + z == 3:
            return (1 - (-1) ** (a + b + c)) / 8
        else:
            raise Exception('x, y or z are not 0/1')

    return v * p_ideal(a, b, c, x, y, z) + (1 - v) / 8

def P_Mermin_array(visibility=1):
    return probfunc2array(P_Mermin, visibility, [2, 2, 2], [2, 2, 2])

def P_2PR(a, b, c, x, y, z, v):
    def p_ideal(a, b, c, x, y, z):
        return ( 1 + (-1) ** (a + b + c + x*y + y*z) ) / 8

    return v * p_ideal(a, b, c, x, y, z) + (1 - v) / 8

def P_2PR_array(visibility=1):
    return probfunc2array(P_2PR, visibility, [2, 2, 2], [2, 2, 2])


def P_Salman(a, b, c, x, y, z, vis):
    u2 = 0.8
    prob_array = target_distribution_gen("Renou-visibility", np.sqrt(u2), vis).astype(float)
    return prob_array[a, b, c]

def P_Salman_array(visibility=1):
    return probfunc2array(P_Salman, visibility, [4, 4, 4], [1, 1, 1])

def P_CHSH_array(visibility=1):
    b0 = qt.ket("0")
    b1 = qt.ket("1")

    b00 = qt.tensor(b0,b0)
    b01 = qt.tensor(b0,b1)
    b10 = qt.tensor(b1,b0)
    b11 = qt.tensor(b1,b1)

    Phi_plus  = 1/np.sqrt(2) * (b01 - b10) # 1/2**0.5 * (01-10)

    #
    visibility = visibility
    #

    iden = qt.Qobj(np.array(qt.identity(4)),
                   dims=[[2,2],[2,2]], type='oper', isherm=True)
    rho = visibility*qt.ket2dm(Phi_plus) + (1-visibility)*iden/4

    A0 = qt.sigmax()
    A1 = qt.sigmaz()
    A = [[A0.eigenstates()[1][0].proj(),
          A0.eigenstates()[1][1].proj()],
         [A1.eigenstates()[1][0].proj(),
          A1.eigenstates()[1][1].proj()]]

    B0 = (-qt.sigmax()-qt.sigmaz())/np.sqrt(2)
    B1 = (-qt.sigmax()+qt.sigmaz())/np.sqrt(2)
    B = [[B0.eigenstates()[1][0].proj(),
          B0.eigenstates()[1][1].proj()],
         [B1.eigenstates()[1][0].proj(),
          B1.eigenstates()[1][1].proj()]]

    p_array = np.zeros((2,2,2,2),dtype=np.float)
    for a, b, x, y in product(range(2), repeat=4):
        p_array[a, b, x, y] = (rho*qt.tensor(A[x][a],B[y][b])).tr()

    return p_array

def P_PRbox_array(visibility=1):
    P_PRbox_array = np.zeros((2,2,2,2))
    for x, y, a, b in product(range(2), repeat=4):
        if (x, y) == (1, 1):
            if a != b:
                P_PRbox_array[a, b, x, y] = 1/2
        else:
            if a == b:
                P_PRbox_array[a, b, x, y] = 1/2
    return visibility * P_PRbox_array + (1 - visibility) / 4
        

def P_Fritz(out_a, out_b, out_c, in_x, in_y, in_z, visibility):
    b0 = qt.ket("0")
    b1 = qt.ket("1")

    b00 = qt.tensor(b0,b0)
    b01 = qt.tensor(b0,b1)
    b10 = qt.tensor(b1,b0)
    b11 = qt.tensor(b1,b1)

    Phi_plus  = 1/np.sqrt(2) * (b01 - b10) # 1/2**0.5 * (01-10)

    #
    #visibility = 1.0
    #

    iden = qt.Qobj(np.array(qt.identity(4)),
                   dims=[[2,2],[2,2]], type='oper', isherm=True)
    rho = visibility*qt.ket2dm(Phi_plus) + (1-visibility)*iden/4

    A0 = qt.sigmax()
    A1 = qt.sigmaz()
    A = [[A0.eigenstates()[1][0].proj(),
          A0.eigenstates()[1][1].proj()],
         [A1.eigenstates()[1][0].proj(),
          A1.eigenstates()[1][1].proj()]]

    B0 = (-qt.sigmax()-qt.sigmaz())/np.sqrt(2)
    B1 = (-qt.sigmax()+qt.sigmaz())/np.sqrt(2)
    B = [[B0.eigenstates()[1][0].proj(),
          B0.eigenstates()[1][1].proj()],
         [B1.eigenstates()[1][0].proj(),
          B1.eigenstates()[1][1].proj()]]

    def probBELL(rho,a,b,x,y):
        return (rho*qt.tensor(A[x][a],B[y][b])).tr()

    def to_binary(a):
        i = int(a/2)
        j = int(a%2)
        return i,j

    assert to_binary(0) == (0, 0) 
    assert to_binary(1) == (0, 1) 
    assert to_binary(2) == (1, 0)
    assert to_binary(3) == (1, 1)

    def probFF(rho, a, b, c):
        # this converts 0-->00, 1-->01, 2-->10, 3-->11
        # Comment how 4-output is related to 2-output 2-input 
        # a=0 -- (x=0,a=0)
        # a=1 -- (x=0,a=1)
        # a=2 -- (x=1,a=0)
        # a=3 -- (x=1,a=1)
        
        # b=0 -- (y=0,b=0)
        # b=1 -- (y=0,b=1)
        # b=2 -- (y=1,b=0)
        # b=3 -- (y=1,b=1)
        
        # c=0 -- (x=0,y=0)
        # c=1 -- (x=0,y=1)
        # c=2 -- (x=1,y=0)
        # c=3 -- (x=1,y=1)
        ai, ao = to_binary(a)
        bi, bo = to_binary(b)
        cx, cy = to_binary(c)
        
        if ai == cx and bi == cy:
            return 0.25*probBELL(rho, ao, bo, ai, bi)
        else:
            return 0.0

    def qreshape(inp,dims):
        return qt.Qobj(inp.data.toarray(), dims=dims)

    def state_noise(visibility):
        Id4    = qreshape(0.25*qt.identity(4),[[2,2],[2,2]])
        # What I will use as the shared state between A and B.
        Phi_plus  = 1/np.sqrt(2) * (b01 - b10) # 1/2**0.5 * (01-10)
        ini_state =  qt.ket2dm(Phi_plus)
        return visibility * ini_state + (1-visibility) * (Id4)


    ################### Set noise here #######################################
    rho_AB = state_noise(visibility)

    # the following two are 0,5*|00><00|+0,5*|11><11|
    rho_BC = 0.5 * qt.ket2dm(b00) + 0.5 * qt.ket2dm(b11)
    rho_CA = 0.5 * qt.ket2dm(b00) + 0.5 * qt.ket2dm(b11)

    # The following line of code gives the state in tensor product of the form
    # A1 B1 B2 C1 C2 A2
    # 1  2  3  4  5  6
    # where
    #  A --(A1)-- rho_AB --(B1)-- B
    #    \                       /
    #     (A2)                (B2)
    #        \                /
    #       rho_CA        rho_BC
    #           \          / 
    #           (C2)    (C1)
    #              \    /
    #                C
    # where between parenthesis we have Hilbert spaces of dimension 2
    # ex B2  === \mathcal{H}_{B2}=\mathbb{C}^2

    full_state_ini = qt.tensor(rho_AB,rho_BC,rho_CA)

    # Now for the measurements it is more convenient to use the ordering
    # A2 A1 B1 B2 C1 C2, so we need to change to this.
    # 6  1  2  3  4  5
    #
    # End(A1 B1 B2 C1 C2 A2) = Sum (..) |i>_1 |j>_2 |k>_3 |l>_4 |m>_5 |n>_6 
    #                                   <o|_1 <p|_2 <q|_3 <r|_4 <s|_5 <t|_6
    #                             |
    #                             |
    #                             V
    # End(A2 A1 B1 B2 C1 C2) = Sum (..) |n>_6 |i>_1 |j>_2 |k>_3 |l>_4 |m>_5 
    #                                   <t|_6 <o|_1 <p|_2 <q|_3 <r|_4 <s|_5
    # and this correspondes to shifting all the corresponding indices.
    #full_state = np.reshape(full_state, [2,2,  2,2,  2,2,  2,2,  2,2,  2,2])
    #                             idx nr  0 1   2 3   4 5   6 7   8 9  10 11
    full_state = qt.tensor_swap(full_state_ini,[4,5])
    full_state = qt.tensor_swap(full_state,[3,4])
    full_state = qt.tensor_swap(full_state,[2,3])
    full_state = qt.tensor_swap(full_state,[1,2])
    full_state = qt.tensor_swap(full_state,[0,1])

    full_state = qt.tensor_swap(full_state,[10,11])
    full_state = qt.tensor_swap(full_state,[9,10])
    full_state = qt.tensor_swap(full_state,[8,9])
    full_state = qt.tensor_swap(full_state,[7,8])
    full_state = qt.tensor_swap(full_state,[6,7])

    b0proj = b0.proj()
    b1proj = b1.proj()

    A00 = qt.tensor(b0proj, A[0][0])
    A01 = qt.tensor(b0proj, A[0][1])
    A10 = qt.tensor(b1proj, A[1][0])
    A11 = qt.tensor(b1proj, A[1][1])

    Ameas = [A00,A01,A10,A11]

    B00 = qt.tensor(B[0][0], b0proj)
    B01 = qt.tensor(B[0][1], b0proj)
    B10 = qt.tensor(B[1][0], b1proj)
    B11 = qt.tensor(B[1][1], b1proj)

    Bmeas = [B00,B01,B10,B11]

    #Cxy
    C00 = qt.tensor(b0proj, b0proj)
    C10 = qt.tensor(b0proj, b1proj)
    C01 = qt.tensor(b1proj, b0proj)
    C11 = qt.tensor(b1proj, b1proj)

    Cmeas = [C00,C01,C10,C11]

    def probability(state,a,b,c):
        return (state*qt.tensor(Ameas[a],Bmeas[b],Cmeas[c])).tr()

    # I know this one is correct, but it has no noise
    # Taken from Alex's code which is taken from pg 6 from 
    # Causal Compatibility Inequalities Admitting of Quantum 
    # Violations in the Triangle Structure
    # https://arxiv.org/abs/1709.06242
    def probF(a, b, c):
        if (((a == 0) and (b == 0) and (c == 0))
            or ((a == 1) and (b == 1) and (c == 0))
            or ((a == 0) and (b == 2) and (c == 1))
            or ((a == 1) and (b == 3) and (c == 1))
            or ((a == 2) and (b == 0) and (c == 2))
            or ((a == 3) and (b == 1) and (c == 2))
            or ((a == 2) and (b == 3) and (c == 3))
            or ((a == 3) and (b == 2) and (c == 3))):
            return (2 + np.sqrt(2)) / 32
        elif (((a == 0) and (b == 1) and (c == 0))
            or ((a == 1) and (b == 0) and (c == 0))
            or ((a == 0) and (b == 3) and (c == 1))
            or ((a == 1) and (b == 2) and (c == 1))
            or ((a == 2) and (b == 1) and (c == 2))
            or ((a == 3) and (b == 0) and (c == 2))
            or ((a == 2) and (b == 2) and (c == 3))
            or ((a == 3) and (b == 3) and (c == 3))):
            return (2 - np.sqrt(2)) / 32
        else:
            return 0.0

    '''
    # check they give the same
    for a,b,c in itertools.product(range(4), repeat=3):
        #diff = np.array(probability(full_state,a,b,c)-probFF(rho,a,b,c))
        #diff = np.array(probability(full_state,a,b,c)-probF(a,b,c))
        diff = np.array(probFF(rho,a,b,c)-probF(a,b,c))
       
        #diff[abs(diff)<1e-10]=0
        print("(a,b,c)=(",a,",",b,",",c,")", diff, "=", probability(full_state,a,b,c), "-", probFF(rho,a,b,c) )
        #print("(a,b,c)=(",a,",",b,",",c,")", diff, "=", probFF(rho,a,b,c), "-", probF(a,b,c) )
    print("The comparison with the analytical formula for the paper only makes sense for 0 visibility!")

    prob_array = [[[0 for i in range(4)] for j in range(4)] for k in range(4)]
    for a,b,c in itertools.product(range(4),repeat=3):
        prob_array[a][b][c] = probFF(rho,a,b,c)#probability(full_state,a,b,c)
    print("visibility =",visibility,"\nprobs[a][b][c] =",prob_array)
    '''
    return float(probFF(rho,out_a,out_b,out_c))
