from sympy import sqrt, simplify, binomial, radsimp
from functools import cache

u=sqrt(5+4*sqrt(2))
term_1 = simplify(2/(u-1))
term_2 = simplify(-2/(u+1))

@cache
def expec_line(n: int) -> float:
    return simplify(radsimp((term_1**(n-1)-term_2**(n-1)) / u))
@cache
def expec_loop(n: int) -> float:
    if n==1:
        return 0
    else:
        return simplify(term_1**(n)+term_2**(n))

def prob_line_s(n: int) -> float:
    return sum(binomial(n, k)*expec_line(k) for k in range(0,n+1))/(2**n)
def prob_loop_s(n: int) -> float:
    return sum((binomial(n, k) * (expec_line(k) if k<n else expec_loop(n))) for k in range(0, n + 1))/(2**n)
def prob_line(n: int) -> float:
    return float(prob_line_s(n))
def prob_loop(n: int) -> float:
    return float(prob_loop_s(n))



if __name__ == '__main__':
    print([prob_line(n) for n in range(1, 10)])
    print([prob_loop(n) for n in range(1, 10)])