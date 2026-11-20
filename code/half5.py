"""The n = 5 lower bound at p = 1/2 exactly, pointwise: for each regime whose closed interval contains 1/2, check
lambda(1/2) in [0,1] and lambda E_1(w) + (1-lambda) E_2(w) >= OPT_5(1/2) at every w in {0,1/2,1}^10, in exact rationals.
With the upper bound of ub5.py this gives OPT_5(1/2) = 394955/851328 independently of the Sturm runs of regcheck.py.
usage: half5.py [NEG]   NEG = opt raises OPT by 1/1000 and must fail."""
import sys
from fractions import Fraction as F
import numpy as np, sympy as sp
import n5regimes as N

p = sp.Symbol('p'); HALF = sp.Rational(1, 2)
neg = sys.argv[1] if len(sys.argv) > 1 else None
Z = np.load('hull_M_n5.npz'); M1, M2 = Z['M1'], Z['M2']; L = M1.shape[1] - 1
den = F(2 ** (L + 1))                       # 2 E_w = sum_k M[w,k] p^k q^(L-k); at p = 1/2 every term is M[w,k] / 2^L
E1 = [F(int(sum(int(c) for c in row))) / den for row in M1]
E2 = [F(int(sum(int(c) for c in row))) / den for row in M2]
target = F(394955, 851328)
allok = True; seen = 0
for name, a, b, mu, opt in N.build():
    if not (float(a) <= 0.5 <= float(b)):
        continue
    seen += 1
    lam_s = sp.cancel(sp.sympify(mu).subs(p, HALF)); o_s = sp.cancel(sp.sympify(opt).subs(p, HALF))
    assert isinstance(lam_s, sp.Rational) and isinstance(o_s, sp.Rational), 'lambda or OPT not rational at 1/2'
    lam = F(int(lam_s.p), int(lam_s.q)); o = F(int(o_s.p), int(o_s.q))
    if neg == 'opt':
        o += F(1, 1000)
    vals = [lam * e1 + (1 - lam) * e2 - o for e1, e2 in zip(E1, E2)]
    m = min(vals); bad = [j for j, v in enumerate(vals) if v < 0]; tight = sum(1 for v in vals if v == 0)
    ok = 0 <= lam <= 1 and not bad and (neg == 'opt' or (o == target and m == 0))
    allok &= ok
    print('%-8s lambda(1/2)=%s  OPT(1/2)=%s  equals 394955/851328: %s  vectors %d  failing %d  tight %d  min slack %s'
          % (name, lam, o, o == target, len(vals), len(bad), tight, m))
    for j in bad[:3]:
        print('   FAIL vertex', ''.join('0h1'[c] for c in Z['verts'][j]))
if seen == 0:
    allok = False; print('no regime contains 1/2')
print(('NEG=%s: ' % neg if neg else '') + ('ALL OK' if allok else 'SOME FAILED'))
