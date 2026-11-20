"""Exact-structure numeric solver for the one-round problem at general n (symmetric protocols).
A symmetric protocol is y(0,k0,k1) for k1 >= 1 (the 0-input player heard k0 zeros and k1 ones), with
y(1,k0,k1) = 1 - y(0,k1,k0). E_w(y) (w ones among n) is convex and piecewise linear, linear on the cells of
{y_i = y_j, y_i + y_j = 1, y_i = 1/2}, whose vertices all lie in {0,1/2,1}^m. E_w = E_{n-w}.
By Sion, OPT = min_y max_w E_w = max_{lambda in simplex} min_y sum_w lambda_w E_w, and the inner min is over the
3^m vertices. For n <= 5 there are two distinct objectives (w=1, w=2), so OPT(p) is a 1-D concave maximisation.
usage: hull.py n p1 p2 ...      prints OPT(p), lambda*, and the active vertices (those within 1e-12 of the min)."""
import sys, itertools
from collections import defaultdict
import numpy as np

n = int(sys.argv[1]); ps = [float(a) for a in sys.argv[2:]]
LINKS = [(i, j) for i in range(n) for j in range(n) if i != j]; L = len(LINKS)
# free coordinates: views of a 0-input player that heard at least one 1
KEYS = sorted({(k0, k1) for w in range(1, n) for k1 in range(1, w + 1) for k0 in range(0, n - w)})
m = len(KEYS); KI = {k: i for i, k in enumerate(KEYS)}
WS = list(range(1, n // 2 + 1))          # distinct objectives

def groups(w):
    """{sorted view tuple: counts by number of delivered links}"""
    x = [1] * w + [0] * (n - w)
    cnt = defaultdict(lambda: np.zeros(L + 1, dtype=np.int64))
    # receivers are independent: enumerate per receiver, then combine (product of per-receiver laws)
    per = []
    for j in range(n):
        a = sum(1 for i in range(n) if i != j and x[i] == 0); b = sum(1 for i in range(n) if i != j and x[i] == 1)
        from math import comb
        per.append([((x[j], k0, k1), comb(a, k0) * comb(b, k1), k0 + k1) for k0 in range(a + 1) for k1 in range(b + 1)])
    for combo in itertools.product(*per):
        views = tuple(sorted(v for v, _, _ in combo)); c = 1; k = 0
        for _, cc, kk in combo: c *= cc; k += kk
        cnt[views][k] += c
    return cnt

def view_code(view):
    """value of a view as (const, coord index, sign): value = const + sign*y[idx]"""
    xi, k0, k1 = view
    if xi == 0: return (0.0, -1, 0) if k1 == 0 else (0.0, KI[(k0, k1)], 1)
    return (1.0, -1, 0) if k0 == 0 else (1.0, KI[(k1, k0)], -1)

G = {w: groups(w) for w in WS}
# merge groups whose views give the same SET of value codes (max-min depends only on that set)
GC = {}
for w in WS:
    acc = defaultdict(lambda: np.zeros(L + 1, dtype=np.int64))
    for views, c in G[w].items(): acc[tuple(sorted(set(view_code(v) for v in views)))] += c
    GC[w] = acc
VERTS = np.array(list(itertools.product((0, 1, 2), repeat=m)), dtype=np.int8)     # y = code/2
print('n=%d m=%d vertices=%d groups=%s merged=%s' % (n, m, len(VERTS), {w: len(G[w]) for w in WS},
      {w: len(GC[w]) for w in WS}), flush=True)
# M_w[v, k] = 2 * coefficient of p^k q^(L-k) in E_w(v): exact integers
M = {}
for w in WS:
    keys = list(GC[w]); C = np.array([GC[w][k] for k in keys], dtype=np.int64)        # G x (L+1)
    Mw = np.zeros((len(VERTS), L + 1), dtype=np.int64)
    for s0 in range(0, len(VERTS), 4096):
        V = VERTS[s0:s0 + 4096].astype(np.int64)
        Dc = np.zeros((len(V), len(keys)), dtype=np.int64)                              # 2*(max-min)
        for gi, codes in enumerate(keys):
            vals = np.array([np.full(len(V), 2 * int(c)) if idx < 0 else 2 * int(c) + sgn * V[:, idx] for c, idx, sgn in codes])
            Dc[:, gi] = vals.max(0) - vals.min(0)
        Mw[s0:s0 + 4096] = Dc @ C
    M[w] = Mw
np.savez_compressed('hull_M_n%d.npz' % n, verts=VERTS, **{'M%d' % w: M[w] for w in WS})
for p in ps:
    kk = np.arange(L + 1); law = p**kk * (1 - p)**(L - kk)
    E = np.array([(M[w].astype(np.float64) @ law) / 2 for w in WS])
    if len(WS) == 1:
        v = E[0].argmin(); print('p=%.6f OPT=%.12f' % (p, E[0, v])); continue
    assert len(WS) == 2
    a, b = E[0], E[1]
    g = lambda lam: (lam * a + (1 - lam) * b).min()
    lo, hi = 0.0, 1.0
    for _ in range(200):
        m1, m2 = lo + (hi - lo) / 3, hi - (hi - lo) / 3
        if g(m1) < g(m2): lo = m1
        else: hi = m2
    lam = (lo + hi) / 2; val = g(lam)
    f = lam * a + (1 - lam) * b; act = np.where(f < val + 1e-10)[0]
    print('p=%.6f OPT=%.12f lambda1=%.6f active=%d' % (p, val, lam, len(act)), flush=True)
    for v in act[:6]:
        print('   v=%s E1=%.10f E2=%.10f' % (''.join('0h1'[c] for c in VERTS[v]), a[v], b[v]))
