"""Independent check of the n=4 closed forms: evaluate the explicit protocol by FULL enumeration of the 12 directed
links (no symmetry reduction, no LP), exact Fractions, at rational p; compare with the closed-form E[D] and t*.
Also: do the two families' closed forms agree exactly at p = 1/2?"""
import itertools
from fractions import Fraction as F

def tstar_half(p):
    return -(2*p**2 - 8*p + 3) / (2*p*(12*p**9 - 38*p**8 + 46*p**7 - 3*p**6 - 66*p**5 + 101*p**4 - 65*p**3 + 17*p**2 + 8*p + 3))
def opt_half(p):
    return (p - 1)*(4*p**12 - 54*p**11 + 224*p**10 - 501*p**9 + 691*p**8 - 616*p**7 + 333*p**6 - 81*p**5 - 14*p**4
                    + 2*p**3 + 8*p**2 - 11*p - 3) / (12*p**9 - 38*p**8 + 46*p**7 - 3*p**6 - 66*p**5 + 101*p**4 - 65*p**3 + 17*p**2 + 8*p + 3)
def tstar_one(p):
    return -(2*p**5 - 3*p**4 + 5*p**3 - 11*p**2 + 11*p - 3) / (2*p*(12*p**10 - 50*p**9 + 84*p**8 - 49*p**7 - 63*p**6 + 167*p**5 - 168*p**4 + 85*p**3 - 12*p**2 - 4*p - 3))
def opt_one(p):
    return (p - 1)*(16*p**15 - 88*p**14 + 240*p**13 - 444*p**12 + 682*p**11 - 967*p**10 + 1222*p**9 - 1234*p**8 + 909*p**7
                    - 435*p**6 + 106*p**5 - 3*p**4 + 9*p**3 - 17*p**2 + 7*p + 3) / (12*p**10 - 50*p**9 + 84*p**8 - 49*p**7 - 63*p**6 + 167*p**5 - 168*p**4 + 85*p**3 - 12*p**2 - 4*p - 3)

def decide(x, k0, k1, t, c3):
    """explicit protocol; a 1-process mirrors a 0-process (flip symmetry)"""
    if x == 1:
        return 1 - decide(0, k1, k0, t, c3)
    if k1 == 0: return F(0)
    table = {(0, 1): F(1, 2), (1, 2): F(1, 2), (0, 2): F(1, 2) + t, (1, 1): F(1, 2) - t, (2, 1): F(1, 2) - t, (0, 3): c3}
    return table[(k0, k1)]

def worst(p, t, c3):
    n = 4; links = [(i, j) for i in range(n) for j in range(n) if i != j]
    res = {}
    for w in (1, 2, 3):
        x = [1]*w + [0]*(n - w)
        tot = F(0)
        for bits in itertools.product((0, 1), repeat=len(links)):
            pr = F(1); recv = [[0, 0] for _ in range(n)]
            for (i, j), b in zip(links, bits):
                if b: pr *= p; recv[j][x[i]] += 1
                else: pr *= 1 - p
            ys = [decide(x[j], recv[j][0], recv[j][1], t, c3) for j in range(n)]
            tot += pr * (max(ys) - min(ys))
        res[w] = tot
    return res

for p in (F(4, 5), F(3, 5), F(9, 10)):
    t = tstar_half(p); r = worst(p, t, F(1, 2) + t)
    print('half p=%s t*=%.6f  E_w=%s  closed=%.10f  match=%s' % (p, float(t), [float(v) for v in r.values()],
          float(opt_half(p)), all(v == opt_half(p) for v in r.values())))
p = F(23, 50); t = tstar_one(p); r = worst(p, t, F(1))
print('one  p=%s t*=%.6f  E_w=%s  closed=%.10f  match=%s' % (p, float(t), [float(v) for v in r.values()],
      float(opt_one(p)), all(v == opt_one(p) for v in r.values())))
h = F(1, 2)
print('p=1/2: opt_half=%s opt_one=%s equal=%s; t*_half=%s t*_one=%s' % (opt_half(h), opt_one(h), opt_half(h) == opt_one(h), tstar_half(h), tstar_one(h)))
