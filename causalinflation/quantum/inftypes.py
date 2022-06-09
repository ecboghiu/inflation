from typing import Dict, List, Tuple, Union, Any
import sympy
import numpy

# Note: one can also use constrained TypeVar to encode a type which is 
# an "or" of different types, but for our usage it doesn't really matter
# See https://stackoverflow.com/questions/58903906/
# whats-the-difference-between-a-constrained-typevar-and-a-union
Symbolic = Union[sympy.core.symbol.Symbol, 
                 sympy.core.numbers.One,
                 sympy.physics.quantum.operator.HermitianOperator, # For some
                            # reason, the HermitianOperator class is 
                            # being used, maybe remove this?
                 sympy.core.power.Pow,
                 sympy.core.mul.Mul]

ArrayMonomial = numpy.ndarray
StringMonomial = str
IntMonomial = int
Monom = Union[ArrayMonomial, StringMonomial, IntMonomial]


    