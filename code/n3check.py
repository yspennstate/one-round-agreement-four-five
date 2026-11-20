"""Reproduce DISC 2025 Theorem 13 (n=3, one round) by the vertex method. Symmetric y(0,k0,k1), k0 in {0,1},
k1 in {1,2}; inputs (1,0,0) and (1,1,0) are mirror images, so OPT = min_u E(u) over u in [0,1]^4, and E is linear on
cells of {u_i=u_j, u_i+u_j=1, u_i in {0,1/2,1}}, so OPT = lower envelope of the 81 vertex polynomials."""
import itertools
from collections import defaultdict
from fractions import Fraction as F
import sympy as sp
p = sp.Symbol('p'); q = 1 - p
n = 3; LINKS = [(i, j) for i in range(n) for j in range(n) if i != j]; x = [1, 0, 0]
KEYS = [(0, 1), (0, 2), (1, 1), (1, 2)]
def E_of(u):
    tab = dict(zip(KEYS, u))
    def out(xi, k0, k1):
        if xi == 0: return F(0) if k1 == 0 else tab[(k0, k1)]
        return F(1) if k0 == 0 else 1 - tab[(k1, k0)]
    cnt = defaultdict(F)
    for bits in itertools.product((0, 1), repeat=len(LINKS)):
        recv = [[0, 0] for _ in range(n)]; k = 0
        for (i, j), b in zip(LINKS, bits):
            if b: recv[j][x[i]] += 1; k += 1
        ys = [out(x[j], *recv[j]) for j in range(n)]
        cnt[k] += max(ys) - min(ys)
    return sp.expand(sum(sp.Rational(v.numerator, v.denominator) * p**k * q**(len(LINKS) - k) for k, v in cnt.items()))
polys = {u: E_of(u) for u in itertools.product((F(0), F(1, 2), F(1)), repeat=4)}
# exact lower envelope on (0,1): breakpoints = roots of pairwise differences
distinct = {}
for u, f in polys.items(): distinct.setdefault(f, u)
fl = list(distinct)
bps = set()
for a, b in itertools.combinations(fl, 2):
    d = sp.Poly(a - b, p)
    if d.degree() > 0:
        for fac, _ in d.factor_list()[1]:
            for r in fac.real_roots():
                if 0 < r < 1: bps.add(r)
pts = [sp.Integer(0)] + sorted(bps, key=float) + [sp.Integer(1)]
env = []
for lo, hi in zip(pts, pts[1:]):
    m = sp.Rational(int((float(lo) + float(hi)) / 2 * 10**9), 10**9)
    best = min(fl, key=lambda f: f.subs(p, m))
    if env and env[-1][2] == best: env[-1] = (env[-1][0], hi, best)
    else: env.append((lo, hi, best))
for lo, hi, f in env:
    print('[%.6f, %.6f]  OPT = %s   at u = %s' % (float(lo), float(hi), sp.factor(f), tuple(str(a) for a in distinct[f])))
print('p>=1/2 piece == q:', sp.expand(env[-1][2] - q) == 0)
t13b = sp.Rational(1, 2) * (2*p*q**2 + p**3 + p*q*(1 - q**2) + p**2*q**2) + q**2*(1 - p**2)
t13c = p*(1 - p**2) + q**2*(1 - p*q)
for lo, hi, f in env:
    print('  piece [%.4f,%.4f]: equals T13 middle form %s, low form %s' % (float(lo), float(hi), sp.expand(f - t13b) == 0, sp.expand(f - t13c) == 0))
