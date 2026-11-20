"""Exact regime check.  On [a, b] (rational or algebraic endpoints, exact), with a weight mu(p) and a claimed value
OPT(p), verify   0 <= mu <= 1   and   mu E_1(v) + (1-mu) E_2(v) - OPT >= 0   for every vertex v of {0,1/2,1}^m
(2 E_w(v) = sum_k M[v,k] p^k q^(L-k), exact integers from hull_M_n<n>.npz; n=3 has one objective, mu=1).
Exact throughout: a Sturm count on a rational bracket first; when it is positive, the real roots are isolated as
algebraic numbers, compared with the endpoints by minimal polynomial plus isolating interval, the interval is split
at the interior roots, and the sign is read at a rational point of every piece.
usage: regcheck.py n [NEG]     NEG in {opt, mu, ext}: negative controls that must FAIL."""
import sys, time
import numpy as np, sympy as sp
p = sp.Symbol('p'); q = 1 - p; R = sp.Rational

def alg(poly, approx):
    rs = sp.Poly(poly, p).real_roots(radicals=False); return min(rs, key=lambda r: abs(float(r) - approx))
def israt(x): return isinstance(x, sp.Rational)
def bracket(x, k):
    if israt(x): return x, x
    c = x.eval_rational(dx=R(1, 10**k), n=k + 5); d = R(1, 10**k); return c - d, c + d
def minpoly(x): return sp.Poly(sp.minimal_polynomial(x, p), p, domain='QQ').monic()
def same(x, y):
    if israt(x) or israt(y): return israt(x) and israt(y) and x == y
    m = minpoly(x)
    if m != minpoly(y): return False
    for k in range(8, 200, 8):
        lo, hi = bracket(y, k)
        if m.count_roots(lo, hi) == 1:
            xl, xh = bracket(x, k + 4); return lo < xl and xh < hi
    raise RuntimeError('cannot isolate')
def lt(x, y):
    """x < y exactly, for x != y"""
    if israt(x) and israt(y): return x < y
    for k in range(8, 400, 8):
        xl, xh = bracket(x, k); yl, yh = bracket(y, k)
        if xh < yl: return True
        if yh < xl: return False
    raise RuntimeError('cannot separate')
def outer(a, b): return bracket(a, 12)[0], bracket(b, 12)[1]

def inside_roots(P, a, b):
    lo, hi = outer(a, b)
    if P.count_roots(lo, hi) == 0: return []
    out = []
    for fac, _ in P.factor_list()[1]:
        if fac.degree() == 0 or fac.count_roots(lo, hi) == 0: continue
        for r in fac.real_roots(radicals=False):
            if same(r, a) or same(r, b): continue
            if lt(a, r) and lt(r, b): out.append(r)
    return sorted(out, key=float)
def rat_between(u, v):
    for k in range(10, 400, 6):
        uh = bracket(u, k)[1]; vl = bracket(v, k)[0]
        if uh < vl: return (uh + vl) / 2
    raise RuntimeError('no rational between')
def nonneg_on(P, a, b):
    if P.is_zero: return True, 0
    rs = inside_roots(P, a, b); pts = [a] + rs + [b]
    return all(P.eval(rat_between(u, v)) >= 0 for u, v in zip(pts, pts[1:])), len(rs)
def nonvanishing(P, a, b):
    """P has no root on the CLOSED interval [a, b]"""
    for e in (a, b):
        if israt(e):
            if P.eval(e) == 0: return False
        elif sp.gcd(P, minpoly(e)).degree() > 0: return False
    return not inside_roots(P, a, b)

def spec(n, neg):
    if n == 3:
        A3 = alg(2*p**3 - 10*p**2 + 9*p - 2, 0.3461)
        return [('T13 low', R(0), A3, 1, (p - 1)*(p**3 - 3*p**2 + p - 1)),
                ('T13 middle', A3, R(1, 2), 1, (2*p**3 - p**2 - 2*p + 2) / 2),
                ('T13 high', R(1, 2), R(1), 1, q)]
    if n == 4:
        import verify_exact as X
        B = alg(2*q**4 - 1, 0.159); A4 = alg(2*p**5 - 3*p**4 + 5*p**3 - 11*p**2 + 11*p - 3, 0.4245)
        c = 8*p**8 - 32*p**7 + 58*p**6 - 54*p**5 + 18*p**4 + 16*p**3 - 15*p**2 - p + 8
        mu_one = 2*p*(p - 1)*c / (12*p**10 - 50*p**9 + 84*p**8 - 49*p**7 - 63*p**6 + 167*p**5 - 168*p**4 + 85*p**3 - 12*p**2 - 4*p - 3)
        mu_half = 2*p*c / (12*p**9 - 38*p**8 + 46*p**7 - 3*p**6 - 66*p**5 + 101*p**4 - 65*p**3 + 17*p**2 + 8*p + 3)
        top1 = R(11, 20) if neg == 'ext' else R(1, 2)
        return [('FV', R(0), B, 0, 1 - 2*q**4*(1 - q**2)**2),
                ('AMP(1/2)', B, A4, 0, 1 - (1 - q**2)**2),
                ('F(1)', A4, top1, mu_one, X.opt_one(p)),
                ('F(1/2+t*)', R(1, 2), R(1), mu_half, X.opt_half(p))]
    if n == 5:
        import n5regimes as N
        return N.build()
    raise SystemExit('no SPEC for n=%d yet' % n)

def run(n, neg, only=None):
    Z = np.load('hull_M_n%d.npz' % n); two = 'M2' in Z
    M1 = Z['M1']; M2 = Z['M2'] if two else None; L = M1.shape[1] - 1
    basis = [sp.Poly(p**k * q**(L - k), p, domain='QQ') for k in range(L + 1)]
    def poly(vec): return sum((int(c) * basis[k] for k, c in enumerate(vec) if c), sp.Poly(0, p, domain='QQ')) * R(1, 2)
    S = spec(n, neg); allok = True
    for i in range(len(S) - 1):
        assert S[i][2] is S[i + 1][1] or (israt(S[i][2]) and S[i][2] == S[i + 1][1]) or neg == 'ext', 'gap between regimes'
    for name, a, b, mu, opt in S:
        if only and name != only: continue
        t0 = time.time()
        if neg == 'opt': opt = opt + R(1, 1000)
        if neg == 'mu' and mu not in (0, 1): mu = mu + R(1, 100)
        mus = sp.sympify(mu)
        # dedupe on what this regime actually reads
        if not two or mus == 1: keys = {tuple(M1[j]): j for j in range(len(M1))}; mode = 1
        elif mus == 0: keys = {tuple(M2[j]): j for j in range(len(M2))}; mode = 2
        else: keys = {(tuple(M1[j]), tuple(M2[j])): j for j in range(len(M1))}; mode = 3
        mun, mud = [sp.Poly(x, p, domain='QQ') for x in sp.fraction(sp.cancel(mus))]
        on, od = [sp.Poly(x, p, domain='QQ') for x in sp.fraction(sp.cancel(opt))]
        den = mud * od
        assert nonvanishing(den, a, b), 'denominator vanishes on the closed interval'
        s = 1 if den.eval(rat_between(a, b)) > 0 else -1; smu = 1 if mud.eval(rat_between(a, b)) > 0 else -1
        ok_mu = nonneg_on(mun * smu, a, b)[0] and nonneg_on((mud - mun) * smu, a, b)[0]
        bad = []; tight = 0; nroots = 0
        for key, j in keys.items():
            if mode == 1: f = mud * poly(M1[j])
            elif mode == 2: f = mud * poly(M2[j])
            else: f = mun * poly(M1[j]) + (mud - mun) * poly(M2[j])
            P = (f * od - on * mud) * s
            if P.is_zero: tight += 1; continue
            ok, k = nonneg_on(P, a, b); nroots += k
            if not ok: bad.append(j)
        allok &= ok_mu and not bad
        print('%-10s [%.6f, %.6f] mu in [0,1]: %s  distinct polys %d  failing %d  identically tight %d  interior roots %d  (%.0fs)'
              % (name, float(a), float(b), ok_mu, len(keys), len(bad), tight, nroots, time.time() - t0), flush=True)
        for j in bad[:3]: print('   FAIL vertex', ''.join('0h1'[c] for c in Z['verts'][j]))
    print(('NEG=%s: ' % neg if neg else '') + ('ALL OK' if allok else 'SOME FAILED'))

if __name__ == '__main__':
    a = sys.argv[1:] + [None, None]
    run(int(a[0]), None if a[1] in (None, '-') else a[1], a[2])
