"""n=5 one round: exact UPPER bounds for every regime of n5regimes.REG.
Mixture regime (u, v), lambda in (0,1): put y = t u + (1-t) v with t (a_u - b_u) = (1-t)(b_v - a_v)  (a = E_1, b = E_2).
By convexity of E_w, E_w(y) <= t E_w(u) + (1-t) E_w(v), and both right-hand sides equal OPT (see NOTES).  So
max(E_1, E_2)(y) <= OPT as soon as t is in [0,1], i.e. (a_u - b_u) >= 0 >= (a_v - b_v) (or the reverse) on the interval.
lambda=0 regime: OPT = E_2(u) for a class of vertices u; a protocol with E_1 <= E_2 = OPT is needed.  For each sub-
interval we name one class vertex with E_2 identical to OPT and check E_1 - OPT <= 0 there exactly.
usage: ub5.py"""
import numpy as np, sympy as sp
import regcheck as RC, n5regimes as N
p = N.p; R = sp.Rational
def P(x): return sp.Poly(sp.numer(sp.cancel(x)), p, domain='QQ'), sp.Poly(sp.denom(sp.cancel(x)), p, domain='QQ')

def nonpos(expr, a, b):
    num, den = P(expr)
    assert RC.nonvanishing(den, a, b)
    s = 1 if den.eval(RC.rat_between(a, b)) > 0 else -1
    return RC.nonneg_on(-num * s, a, b)[0]

S = N.build(); allok = True
codes = [''.join('0h1'[c] for c in v) for v in N.Z['verts']]
L = N.L; kk = np.arange(L + 1)
for (name, a, b, lam, opt), (_, r, _) in zip(S, N.REG):
    if not isinstance(r, str):
        u, v = r
        du = N.E(u, 1) - N.E(u, 2); dv = N.E(v, 1) - N.E(v, 2)
        ok = (nonpos(-du, a, b) and nonpos(dv, a, b)) or (nonpos(du, a, b) and nonpos(-dv, a, b))
        print('%-7s [%.6f, %.6f] mixture %s/%s  t in [0,1]: %s' % (name, float(a), float(b), u, v, ok), flush=True)
        allok &= ok; continue
    # lambda=0: class = vertices whose M2 row equals that of r
    row = N.M2[N.idx(r)]; cls = np.where((N.M2 == row).all(1))[0]
    # pick, on a grid, the class vertex of smallest E_1; then cover [a, b] exactly, splitting where needed
    grid = np.linspace(float(a), float(b), 41)[1:-1]
    law = np.array([g**kk * (1 - g)**(L - kk) for g in grid])
    e1 = (N.M1[cls].astype(np.float64) @ law.T) / 2
    picks = []
    for j in np.unique(e1.argmin(0)):
        c = codes[cls[j]]
        if c not in picks: picks.append(c)
    # exact: for each pick find the sub-intervals of [a,b] where E_1 <= OPT, and check their union covers [a,b]
    pieces = []
    for c in picks:
        assert sp.expand(N.E(c, 2) - opt) == 0
        num, den = P(N.E(c, 1) - opt)
        rs = RC.inside_roots(num, a, b); pts = [a] + rs + [b]
        for x, y in zip(pts, pts[1:]):
            if num.eval(RC.rat_between(x, y)) * (1 if den.eval(RC.rat_between(x, y)) > 0 else -1) <= 0:
                pieces.append((x, y, c))
    pieces.sort(key=lambda t: float(t[0]))
    cur = a; covered = True; used = []
    while not (RC.same(cur, b) if not RC.israt(cur) or not RC.israt(b) else cur == b):
        nxt = [t for t in pieces if (RC.same(t[0], cur) if not (RC.israt(t[0]) and RC.israt(cur)) else t[0] == cur)
               or (RC.lt(t[0], cur) and RC.lt(cur, t[1]))]
        if not nxt: covered = False; break
        best = max(nxt, key=lambda t: float(t[1])); used.append((float(cur), float(best[1]), best[2])); cur = best[1]
    print('%-7s [%.6f, %.6f] lambda=0  class %d vertices, picks %d, covered: %s' % (name, float(a), float(b), len(cls), len(picks), covered))
    for x, y, c in used: print('     [%.6f, %.6f] %s' % (x, y, c))
    allok &= covered
print('UPPER ALL OK' if allok else 'UPPER SOME FAILED')
