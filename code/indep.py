"""Independent exact checks (they use only the full-enumeration evaluator of verify_cert.py):
(A) closed forms opt_half/opt_one/t* of verify_exact.py equal verify_cert's E_1(t*) and t* IDENTICALLY in p;
(B) min over the box of E_2 by vertex enumeration: E_2 (input split 2-2) depends on u=(y(0,1),y(0,2),y(1,1),y(1,2)) only
    and is linear on the cells of {u_i=u_j, u_i+u_j=1, u_i in {0,1/2,1}}; every vertex lies in {0,1/2,1}^4, so
    min_box E_2 = min of 81 polynomials. Each is computed by its own full 12-link enumeration (own code below), and
    f_v - OPT_lower(p) is checked >= 0 on each regime interval by exact roots;
(C) the low formula equals 1 - 2 q^4 (1-q^2)^2."""
import itertools
from collections import defaultdict
from fractions import Fraction as F
import sympy as sp
import verify_exact as X
import verify_cert as V
p = sp.Symbol('p'); q = 1 - p
# (A)
for fam, opt_c, t_c in (('half', X.opt_half, X.tstar_half), ('one', X.opt_one, X.tstar_one)):
    Y = V.family(fam)
    E1 = V.enumerate_input(Y, 1)[0]; E2 = V.enumerate_input(Y, 2)[0]
    a1, b1 = sp.Poly(E1, V.T).all_coeffs(); a2, b2 = sp.Poly(E2, V.T).all_coeffs()
    ts = sp.cancel((b2 - b1) / (a1 - a2)).subs(V.P, p); op = sp.cancel(E1.subs(V.T, (b2 - b1) / (a1 - a2))).subs(V.P, p)
    print('(A) %s: t* identity %s, opt identity %s' % (fam, sp.cancel(ts - t_c(p)) == 0, sp.cancel(op - opt_c(p)) == 0))
# (B) own enumeration of E_2 for a 0/1-mirrored protocol given u
LINKS = [(i, j) for i in range(4) for j in range(4) if i != j]
x = [1, 1, 0, 0]
def E2_of(u):
    tab = {(0, 1): u[0], (0, 2): u[1], (1, 1): u[2], (1, 2): u[3]}
    def out(xi, k0, k1):
        if xi == 0: return F(0) if k1 == 0 else tab[(k0, k1)]
        return F(1) if k0 == 0 else 1 - tab[(k1, k0)]
    cnt = defaultdict(F)
    for bits in itertools.product((0, 1), repeat=12):
        recv = [[0, 0] for _ in range(4)]; k = 0
        for (i, j), b in zip(LINKS, bits):
            if b: recv[j][x[i]] += 1; k += 1
        ys = [out(x[j], *recv[j]) for j in range(4)]
        cnt[k] += max(ys) - min(ys)
    return sp.expand(sum(sp.Rational(v.numerator, v.denominator) * p**k * q**(12 - k) for k, v in cnt.items()))
low = 1 - 8*p**2 + 40*p**3 - 82*p**4 + 88*p**5 - 52*p**6 + 16*p**7 - 2*p**8
mid = 1 - (1 - q**2)**2
print('(C) low == 1-2q^4(1-q^2)^2:', sp.expand(low - (1 - 2*q**4*(1 - q**2)**2)) == 0)
r0 = [x for x in sp.Poly(2*q**4 - 1, p).real_roots() if 0 < x < 1][0]
pc = sp.Poly(2*p**5 - 3*p**4 + 5*p**3 - 11*p**2 + 11*p - 3, p).real_roots()
r1 = [x for x in pc if 0 < x < 1][0]
print('thresholds %.12f %.12f' % (float(r0), float(r1)))
def nonneg_on(f, lo, hi):
    f = sp.Poly(sp.expand(f), p)
    if f.is_zero: return True
    for fac, _ in f.factor_list()[1]:
        for r in fac.real_roots():
            if lo < r < hi: return False
    mid_pt = sp.Rational(int((float(lo) + float(hi)) / 2 * 10**9), 10**9)
    return f.eval(mid_pt) >= 0
vals = (F(0), F(1, 2), F(1))
bad = []; attain_low = []; attain_mid = []
for u in itertools.product(vals, repeat=4):
    f = E2_of(u)
    okL = nonneg_on(f - low, 0, r0); okM = nonneg_on(f - mid, r0, r1)
    if not (okL and okM): bad.append((u, okL, okM))
    if sp.expand(f - low) == 0: attain_low.append(u)
    if sp.expand(f - mid) == 0: attain_mid.append(u)
print('(B) 81 vertices; violations:', bad)
print('    vertices equal to low formula:', [tuple(str(a) for a in u) for u in attain_low])
print('    vertices equal to middle formula:', [tuple(str(a) for a in u) for u in attain_mid])
