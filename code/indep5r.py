"""Independent control for hull_M_n5.npz on a seeded random sample of vertices (plus named ones): enumerate all 2^20
delivery patterns of the 20 directed links directly (no view grouping), compute every process's doubled output, and
compare 2*E_w's coefficient vector (by number of delivered links) with the stored rows M1, M2.
The received counts (k0, k1) per (w, receiver) are computed once and reused for every vertex.
usage: indep5r.py NSAMPLE SEED [code ...]"""
import sys
import numpy as np
n = 5
LINKS = [(i, j) for i in range(n) for j in range(n) if i != j]; L = len(LINKS)
KEYS = sorted({(k0, k1) for w in range(1, n) for k1 in range(1, w + 1) for k0 in range(0, n - w)})
KI = {k: i for i, k in enumerate(KEYS)}
Z = np.load('hull_M_n5.npz'); verts = Z['verts']; M = {1: Z['M1'], 2: Z['M2']}
def idx(code):
    i = 0
    for ch in code: i = 3 * i + '0h1'.index(ch)
    return i
ns, seed = int(sys.argv[1]), int(sys.argv[2])
rng = np.random.default_rng(seed)
sample = sorted(set(rng.choice(len(verts), size=ns, replace=False).tolist()) | {idx(c) for c in sys.argv[3:]})
masks = np.arange(1 << L, dtype=np.int64)
bits = ((masks[:, None] >> np.arange(L)) & 1).astype(np.int8)
pop = bits.sum(1)
CNT = {}
for w in (1, 2):
    x = [1] * w + [0] * (n - w)
    for j in range(n):
        k0 = np.zeros(len(masks), np.int64); k1 = np.zeros(len(masks), np.int64)
        for li, (i, jj) in enumerate(LINKS):
            if jj != j: continue
            if x[i] == 0: k0 += bits[:, li]
            else: k1 += bits[:, li]
        CNT[(w, j)] = (x[j], k0 * n + k1)
del bits
bad = 0
for vi in sample:
    y2 = verts[vi]
    for w in (1, 2):
        outs = []
        for j in range(n):
            xj, code_kk = CNT[(w, j)]
            lut = np.full(n * n, -9, np.int64)
            for a in range(n):
                for b in range(n):
                    if xj == 0: v = 0 if b == 0 else (int(y2[KI[(a, b)]]) if (a, b) in KI else -9)
                    else: v = 2 if a == 0 else (2 - int(y2[KI[(b, a)]]) if (b, a) in KI else -9)
                    lut[a * n + b] = v
            outs.append(lut[code_kk])
        O = np.stack(outs, 1); assert (O >= 0).all(), "unreachable view used"
        D = O.max(1) - O.min(1)
        got = np.bincount(pop, weights=D, minlength=L + 1).astype(np.int64)
        if not (got == M[w][vi]).all():
            bad += 1; print('MISMATCH', ''.join('0h1'[c] for c in y2), 'w=%d' % w, flush=True)
print('checked %d vertices x 2 objectives, seed %d, mismatches %d' % (len(sample), seed, bad))
print('INDEP5R ALL MATCH' if bad == 0 else 'INDEP5R MISMATCH')
