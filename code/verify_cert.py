"""LP-free exact check of one dumped certificate (a JSON file) for the n=4 one-round problem.
Rebuilds everything from a FULL enumeration of the 12 directed links (not the per-receiver
product law of hull.py), then checks, exactly in Q(p):
  1. E_3 == E_1 for the family, and t*(p) equalises E_1 and E_2;
  2. the dumped groups are exactly the subgradient groups of E_1, E_2 at y*(p) (argmax/argmin gradient sets);
  3. stationarity: sum_g P_g(p) (sum_A a v - sum_B b v) + nu == 0 in all six coordinates, identically in p;
  4. group sums: every group of input w has sum a = sum b = c_w (c_1 = mu, c_2 = 1 - mu), identically in p;
  5. signs on the claimed interval: no sign change strictly inside (exact real roots), sign at an exact rational
     interior point, denominators nonzero on the closed interval, 0 < t* < 1/2 on the closed interval.
Prints the interval endpoints as exact algebraic numbers (irreducible polynomial + root index).
usage: verify_cert.py certs/<file>.json [more ...]"""
import sys, json, itertools
from fractions import Fraction as F
from collections import defaultdict
import sympy as sp

P = sp.Symbol('p'); Q = 1 - P; T = sp.Symbol('t')
FREE = [(0, 1), (0, 2), (0, 3), (1, 1), (1, 2), (2, 1)]
n = 4
LINKS = [(i, j) for i in range(n) for j in range(n) if i != j]

def family(fam):
    h = F(1, 2)   # each value as (a, b) meaning a + b t
    c3 = (h, F(1)) if fam == 'half' else (F(1), F(0))
    return {(0, 1): (h, F(0)), (1, 2): (h, F(0)), (0, 2): (h, F(1)), (1, 1): (h, F(-1)), (2, 1): (h, F(-1)), (0, 3): c3}

def view_val_grad(Y, x, k0, k1):
    g = [0] * 6
    if x == 0:
        if k1 == 0: return (F(0), F(0)), tuple(g)
        g[FREE.index((k0, k1))] = 1; return Y[(k0, k1)], tuple(g)
    if k0 == 0: return (F(1), F(0)), tuple(g)
    a, b = Y[(k1, k0)]; g[FREE.index((k1, k0))] = -1
    return (1 - a, -b), tuple(g)

def enumerate_input(Y, w):
    """returns E_w as a sympy expr in (p,t), and group probability polynomials {(w, gmax, gmin): poly in p}"""
    x = [1] * w + [0] * (n - w)
    Dcount = defaultdict(int); Gcount = defaultdict(int)
    for bits in itertools.product((0, 1), repeat=len(LINKS)):
        recv = [[0, 0] for _ in range(n)]
        k = 0
        for (i, j), b in zip(LINKS, bits):
            if b: recv[j][x[i]] += 1; k += 1
        vg = [view_val_grad(Y, x[j], recv[j][0], recv[j][1]) for j in range(n)]
        key = lambda ab: ab[0] + ab[1] * F(1, 4)        # ordering is constant for 0 < t < 1/2
        vals = [v for v, _ in vg]
        mx = max(vals, key=key); mn = min(vals, key=key)
        # exact ties only (identical linear forms); a strict order at t=1/4 that flips inside (0,1/2) is checked below
        gmax = frozenset(g for v, g in vg if v == mx); gmin = frozenset(g for v, g in vg if v == mn)
        Dcount[(k, mx[0] - mn[0], mx[1] - mn[1])] += 1
        Gcount[((w, gmax, gmin), k)] += 1
    E = sum(c * P**k * Q**(12 - k) * (sp.Rational(a.numerator, a.denominator) + sp.Rational(b.numerator, b.denominator) * T)
            for (k, a, b), c in Dcount.items())
    G = defaultdict(lambda: 0)
    for (key, k), c in Gcount.items(): G[key] += c * P**k * Q**(12 - k)
    return sp.expand(E), {k: sp.expand(v) for k, v in G.items()}

def order_is_constant(Y):
    """every pair of view values keeps its order (or equality) on 0 < t < 1/2"""
    vals = set([(F(0), F(0)), (F(1), F(0))] + list(Y.values()) + [(1 - a, -b) for a, b in Y.values()])
    for (a1, b1), (a2, b2) in itertools.combinations(vals, 2):
        if b1 != b2:
            tc = (a2 - a1) / (b1 - b2)
            if 0 < tc < F(1, 2): return False
    return True

def irreducible_roots(expr, lo=0, hi=1):
    out = []
    num, den = sp.fraction(sp.cancel(sp.together(expr)))
    for poly in (num, den):
        poly = sp.Poly(poly, P)
        if poly.degree() <= 0: continue
        for fac, _ in poly.factor_list()[1]:
            for r in fac.real_roots():
                if lo < r < hi: out.append(r)
    return out

def check(path):
    d = json.load(open(path))
    fam = d['family']; Y = family(fam)
    assert order_is_constant(Y), 'value order not constant on (0,1/2)'
    E, G = {}, {}
    for w in (1, 2, 3):
        E[w], g = enumerate_input(Y, w)
        if w < 3: G.update(g)
    assert sp.expand(E[3] - E[1]) == 0, 'E3 != E1'
    a1, b1 = sp.Poly(E[1], T).all_coeffs(); a2, b2 = sp.Poly(E[2], T).all_coeffs()
    tstar = sp.cancel((b2 - b1) / (a1 - a2))
    opt = sp.cancel(E[1].subs(T, tstar))
    # coordinates of y* fixed at 1 or 0 for all t
    at1 = [i for i, k in enumerate(FREE) if Y[k] == (F(1), F(0))]
    at0 = [i for i, k in enumerate(FREE) if Y[k] == (F(0), F(0))]
    assert sorted(at1) == sorted(d['at1']) and sorted(at0) == sorted(d['at0']), 'box faces differ'
    mu = sp.Integer(0); a = defaultdict(lambda: 0); b = defaultdict(lambda: 0); nu = {}
    for v in d['vars']:
        e = sp.sympify(v['expr'], locals={'p': P})
        assert e.free_symbols <= {P}, 'free symbols in multiplier'
        if v['kind'] == 'mu': mu = e
        elif v['kind'] == 'nu':
            assert v['i'] in at1 + at0, 'nu on an interior coordinate'; nu[v['i']] = e
        else:
            key = (v['w'], frozenset(tuple(x) for x in v['A']), frozenset(tuple(x) for x in v['B']))
            assert key in G, 'dumped group not a real subgradient group'
            gv = tuple(v['v'])
            assert gv in (key[1] if v['kind'] == 'a' else key[2]), 'gradient not in its active set'
            (a if v['kind'] == 'a' else b)[(key, gv)] = e
    # 3. stationarity
    for i in range(6):
        s = nu.get(i, 0)
        for key, pg in G.items():
            s += pg * (sum(a[(key, g)] * g[i] for g in key[1]) - sum(b[(key, g)] * g[i] for g in key[2]))
        assert sp.cancel(sp.together(s)) == 0, 'stationarity fails in coordinate %d' % i
    # 4. group sums
    for key in G:
        c = mu if key[0] == 1 else 1 - mu
        for S, m in ((key[1], a), (key[2], b)):
            assert sp.cancel(sum(m[(key, g)] for g in S) - c) == 0, 'group sum fails'
    # 5. signs on the claimed interval
    conds = [mu, 1 - mu, tstar, sp.Rational(1, 2) - tstar] + [e for e in a.values()] + [e for e in b.values()] \
            + [(nu[i] if i in at1 else -nu[i]) for i in nu]
    conds = [sp.cancel(sp.together(c)) for c in conds if c != 0]
    lo_f, hi_f = d['interval']
    roots = set()
    for c in conds: roots.update(irreducible_roots(c))
    roots = sorted(roots, key=lambda r: float(r))
    def snap(x):
        if abs(x) < 1e-12: return sp.Integer(0)
        if abs(x - 1) < 1e-12: return sp.Integer(1)
        near = [r for r in roots if abs(float(r) - x) < 1e-8]
        assert len(near) == 1, 'endpoint %.10f is not a unique exact root' % x
        return near[0]
    L, H = snap(lo_f), snap(hi_f)
    inside = [r for r in roots if L < r < H]
    assert not inside, 'a condition changes sign inside the interval at %s' % [float(r) for r in inside]
    mid = sp.Rational(int((float(L) + float(H)) / 2 * 10**9), 10**9)
    assert L < mid < H
    for c in conds: assert c.subs(P, mid) >= 0, 'negative condition at the interior point'
    for c in conds[2:4]: assert c.subs(P, mid) > 0
    # denominators nonzero on the closed interval (endpoint 1 excluded: p=1 is the trivial channel)
    for c in conds:
        den = sp.fraction(c)[1]
        for r in irreducible_roots(den, -1, 2):
            assert not (L <= r <= H) or (r == 1 and H == 1), 'denominator vanishes at %s' % float(r)
    # strict conditions at the endpoints
    # strict at endpoints, except t* = 0 exactly (the bottom of the F(1) range; that point is covered by rays.py)
    for X in (L, H):
        if X in (0, 1): continue
        tv = sp.N(tstar.subs(P, X), 30)
        tnum = sp.Poly(sp.fraction(sp.cancel(tstar))[0], P)
        tz = (tnum.eval(X) == 0) if X.is_Rational else tnum.rem(sp.Poly(sp.minimal_polynomial(X, P), P)).is_zero
        assert (0 < tv or tz) and tv < 0.5, 't* leaves (0,1/2) at an endpoint'
        if tz: print('   note: t* = 0 at endpoint %.12f' % float(X))
    def desc(r):
        if r.is_Rational: return str(r)
        return '%s root#%d (~%.12f)' % (sp.Poly(r.poly.as_expr(), P).as_expr(), r.index, float(r))
    print('OK %s  fam=%s  [%s, %s]' % (path, fam, desc(L), desc(H)))
    return fam, L, H, opt

if __name__ == '__main__':
    cells = []
    for f in sys.argv[1:]:
        try:
            cells.append(check(f))
        except Exception as ex:
            print('FAIL %s  %s: %s' % (f, type(ex).__name__, ex), flush=True)
    print('checked %d, passed %d' % (len(sys.argv) - 1, len(cells)))
    for fam in sorted(set(c[0] for c in cells)):
        iv = sorted([(c[1], c[2]) for c in cells if c[0] == fam], key=lambda x: float(x[0]))
        cur_lo, cur_hi = iv[0]
        segs = []
        for lo, hi in iv[1:]:
            if lo <= cur_hi:
                if hi > cur_hi: cur_hi = hi
            else:
                segs.append((cur_lo, cur_hi)); cur_lo, cur_hi = lo, hi
        segs.append((cur_lo, cur_hi))
        print('COVER fam=%s: %d exact segment(s): %s' % (fam, len(segs), [(float(x), float(y)) for x, y in segs]))
