"""Exact lower-bound certificate for the n=4 one-round optimum on a p-interval.
Symmetric protocols y in [0,1]^6 suffice (perm+flip averaging). For symmetric y, E_1 = E_3, so
OPT = min_y max(E_1,E_2) >= min_y f(y),  f = mu E_1 + (1-mu) E_2, any mu in [0,1].
If 0 is in the subdifferential of f at y* (plus the box normal cone), f(y*) = min f, and when E_1(y*)=E_2(y*)
this lower bound equals the upper bound max(E_1,E_2)(y*): y* is optimal.
Subdifferential of E_w = sum over executions pr * (conv grads of argmax views - conv grads of argmin views).
We find a vertex of the certificate polytope numerically at p0, keep its support, re-solve it exactly in Q(p),
and print the sign conditions that make it valid; the caller checks them on the interval.
usage: cert_fixed.py FAMILY p0      FAMILY in {half, one}"""
import sys, itertools
from collections import defaultdict
from math import comb
import numpy as np
import sympy as sp
from scipy.optimize import linprog

n = 4
P, T = sp.symbols('p t'); Q = 1 - P
half = sp.Rational(1, 2)
FREE = [(0, 1), (0, 2), (0, 3), (1, 1), (1, 2), (2, 1)]
fam, p0 = sys.argv[1], float(sys.argv[2])
# FIXED protocol y* = all 1/2 (no t); certificate with mu = 0 proves min_y E_2 = E_2(y*); need E_2 >= E_1 at y*.
assert fam == 'fixedhalf'
Y = {k: half for k in FREE}

# t*(p) from the equaliser (recomputed here so the script stands alone)
def per_receiver(w):
    per = []
    for j in range(n):
        xj = 1 if j < w else 0
        a = (n - w) - (1 - xj); b = w - xj
        per.append([((xj, k0, k1), comb(a, k0) * comb(b, k1) * P**(k0 + k1) * Q**(a + b - k0 - k1))
                    for k0 in range(a + 1) for k1 in range(b + 1)])
    return per

def val_grad(v):
    x, k0, k1 = v
    g = [0] * 6
    if x == 0:
        if k1 == 0: return sp.Integer(0), tuple(g)
        g[FREE.index((k0, k1))] = 1; return Y[(k0, k1)], tuple(g)
    if k0 == 0: return sp.Integer(1), tuple(g)
    g[FREE.index((k1, k0))] = -1; return 1 - Y[(k1, k0)], tuple(g)

# exact E_w(p,t) and t*
Ew = {}
execs = {}
for w in (1, 2):
    acc = 0; lst = []
    for combo in itertools.product(*per_receiver(w)):
        views = sorted(set(v for v, _ in combo))
        pr = sp.Integer(1)
        for _, pj in combo: pr *= pj
        lst.append((views, pr))
    execs[w] = lst
for w in (1, 2):
    tot = 0
    # value ordering is valid for 0<t<1/2; use t=0.1 to order
    for views, pr in execs[w]:
        vals = [val_grad(v)[0] for v in views]
        num = [float(e.subs(T, 0.1)) for e in vals]
        tot += (vals[num.index(max(num))] - vals[num.index(min(num))]) * pr
    Ew[w] = sp.expand(tot)
gap21 = sp.factor(Ew[2] - Ew[1])
print('E_2 =', sp.factor(Ew[2]), '  E_2 - E_1 =', gap21)
t0 = 0.0
print('family', fam, 'p0', p0, 't*(p0)', t0)

# group executions by (w, argmax grad set, argmin grad set) at (p0, t*(p0))
groups = defaultdict(lambda: 0)
for w in (1, 2):
    for views, pr in execs[w]:
        vg = [val_grad(v) for v in views]
        num = [float(e.subs(T, t0)) for e, _ in vg]
        mx, mn = max(num), min(num)
        gmax = frozenset(g for (e, g), x in zip(vg, num) if abs(x - mx) < 1e-9)
        gmin = frozenset(g for (e, g), x in zip(vg, num) if abs(x - mn) < 1e-9)
        groups[(w, gmax, gmin)] += pr
groups = {k: sp.factor(v) for k, v in groups.items()}
print('groups', len(groups), ' with ties:', sum(1 for (w, A, B) in groups if len(A) > 1 or len(B) > 1))

# box normal cone: coordinates at 1 or 0
ystar = {k: float(Y[k].subs(T, t0)) for k in FREE}
at1 = [i for i, k in enumerate(FREE) if abs(ystar[k] - 1) < 1e-12]
at0 = [i for i, k in enumerate(FREE) if abs(ystar[k]) < 1e-12]

# variables: mu, a[g,v] (argmax side), b[g,v] (argmin side), nu[i] for boundary coords
var = ['mu']; vidx = {}
for gi, (w, A, B) in enumerate(groups):
    for v in sorted(A): vidx[('a', gi, v)] = len(var); var.append(('a', gi, v))
    for v in sorted(B): vidx[('b', gi, v)] = len(var); var.append(('b', gi, v))
for i in at1 + at0: vidx[('nu', i)] = len(var); var.append(('nu', i))
NV = len(var)
keys = list(groups)
def build(prfun):
    Aeq = []; beq = []
    for i in range(6):   # sum_g pr_g (sum_v a v_i - sum_v b v_i) + nu_i = 0
        r = [0] * NV
        for gi, k in enumerate(keys):
            w, A, B = k; pg = prfun(groups[k])
            for v in A: r[vidx[('a', gi, v)]] += pg * v[i]
            for v in B: r[vidx[('b', gi, v)]] -= pg * v[i]
        if ('nu', i) in vidx: r[vidx[('nu', i)]] = 1
        Aeq.append(r); beq.append(0)
    for gi, k in enumerate(keys):   # sum_v a = c_w, sum_v b = c_w, c_1 = mu, c_2 = 1-mu
        w, A, B = k
        for side, S in (('a', A), ('b', B)):
            r = [0] * NV
            for v in S: r[vidx[(side, gi, v)]] = 1
            if w == 1: r[0] = -1; Aeq.append(r); beq.append(0)
            else: r[0] = 1; Aeq.append(r); beq.append(1)
    return Aeq, beq
Aeq, beq = build(lambda e: float(e.subs(P, p0)))
bounds = [(0, 0)] + [(0, None) if (isinstance(v, tuple) and v[0] != 'nu') else
                     ((0, None) if v[1] in at1 else (None, 0)) for v in var[1:]]
res = linprog(np.zeros(NV), A_eq=np.array(Aeq, float), b_eq=np.array(beq, float), bounds=bounds, method='highs-ds')
print('numeric certificate feasible:', res.status == 0, res.message[:60])
if res.status != 0: sys.exit(1)
x = res.x
support = [j for j in range(NV) if abs(x[j]) > 1e-10]
print('support size', len(support), 'mu(p0)=%.6f' % x[0])
# exact re-solve on the support, symbolic in p
Aeq_s, beq_s = build(lambda e: e)
syms = sp.symbols('z0:%d' % NV)
eqs = []
for r, b in zip(Aeq_s, beq_s):
    e = sum(r[j] * syms[j] for j in support if r[j] != 0) - b
    if e != 0: eqs.append(e)
sol = sp.solve(eqs, [syms[j] for j in support], dict=True)
print('exact solutions:', len(sol))
if not sol: sys.exit(2)
sol = sol[0]
free_left = set().union(*[sp.sympify(v).free_symbols for v in sol.values()]) - {P}
print('undetermined symbols left:', free_left)
# validity interval: every support multiplier >= 0 (nu sign per its cone), mu in [0,1], 0 < t* < 1/2
conds = []
for j in support:
    e = sp.cancel(sol[syms[j]])
    if var[j] == 'mu':
        conds += [e, 1 - e]
    elif isinstance(var[j], tuple) and var[j][0] == 'nu':
        conds.append(e if var[j][1] in at1 else -e)
    else:
        conds.append(e)
conds += [gap21]
crit = set()
for e in conds:
    num, den = sp.fraction(sp.cancel(sp.together(e)))
    for poly in (num, den):
        if poly.free_symbols:
            for r in sp.Poly(poly, P).real_roots():
                rf = float(r)
                if 0 < rf < 1: crit.add(rf)
pts = sorted(crit | {0.0, 1.0})
ok = []
for lo, hi in zip(pts, pts[1:]):
    mid = (lo + hi) / 2
    good = all(float(sp.cancel(e).subs(P, mid)) >= 0 for e in conds)
    ok.append((lo, hi, good))
# merge to the maximal valid interval containing p0
cur = [c for c in ok if c[0] <= p0 <= c[1]][0]
i = ok.index(cur); lo_i = hi_i = i
while lo_i > 0 and ok[lo_i - 1][2]: lo_i -= 1
while hi_i < len(ok) - 1 and ok[hi_i + 1][2]: hi_i += 1
print('certificate valid on p in [%.10f, %.10f]  (p0 cell valid: %s)' % (ok[lo_i][0], ok[hi_i][1], cur[2]))
print('critical points in (0,1):', ['%.6f' % c for c in sorted(crit)])
if '--quiet' in sys.argv: sys.exit(0)
# report every support variable as a rational function of p and its value at p0
for j in support:
    e = sp.factor(sol[syms[j]])
    vv = float(e.subs(P, p0)) if not free_left else None
    if j == 0: print('mu(p) =', e)
    print('  %s  = %s   (at p0: %s)' % (var[j], str(e)[:160], vv))
