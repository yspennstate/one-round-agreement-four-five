"""Independent control for hull_M_n5.npz: for named vertices, enumerate all 2^20 delivery patterns of the 20 directed
links directly (no view grouping, no merging), compute each player's output from the symmetric protocol, and compare
2*E_w's coefficient vector (by number of delivered links) with the stored rows M1, M2.  usage: indep5.py code..."""
import sys, itertools
import numpy as np
n = 5
LINKS = [(i, j) for i in range(n) for j in range(n) if i != j]; L = len(LINKS)
KEYS = sorted({(k0, k1) for w in range(1, n) for k1 in range(1, w + 1) for k0 in range(0, n - w)})
KI = {k: i for i, k in enumerate(KEYS)}
Z = np.load('hull_M_n5.npz'); verts = Z['verts']
def idx(code):
    i = 0
    for ch in code: i = 3 * i + '0h1'.index(ch)
    return i
def table(y2):
    """doubled output for view (x, k0, k1), as a dict"""
    T = {}
    for x in (0, 1):
        for k0 in range(n):
            for k1 in range(n):
                if x == 0: v = 0 if k1 == 0 else (int(y2[KI[(k0, k1)]]) if (k0, k1) in KI else -9)
                else: v = 2 if k0 == 0 else (2 - int(y2[KI[(k1, k0)]]) if (k1, k0) in KI else -9)
                T[(x, k0, k1)] = v
    return T
masks = np.arange(1 << L, dtype=np.int64)
bits = ((masks[:, None] >> np.arange(L)) & 1).astype(np.int8)          # 2^20 x 20, about 20 MB
pop = bits.sum(1)
ok_all = True
for code in sys.argv[1:]:
    y2 = verts[idx(code)]; assert ''.join('0h1'[c] for c in y2) == code
    T = table(y2)
    for w, Mw in ((1, Z['M1']), (2, Z['M2'])):
        x = [1] * w + [0] * (n - w)
        outs = []
        for j in range(n):
            k0 = np.zeros(len(masks), np.int64); k1 = np.zeros(len(masks), np.int64)
            for li, (i, jj) in enumerate(LINKS):
                if jj != j: continue
                if x[i] == 0: k0 += bits[:, li]
                else: k1 += bits[:, li]
            lut = np.array([[T[(x[j], a, b)] for b in range(n)] for a in range(n)])
            outs.append(lut[k0, k1])
        O = np.stack(outs, 1); assert (O >= 0).all(), "unreachable view used"; D = O.max(1) - O.min(1)
        got = np.bincount(pop, weights=D, minlength=L + 1).astype(np.int64)
        match = bool((got == Mw[idx(code)]).all()); ok_all &= match
        print('%s w=%d match=%s' % (code, w, match), flush=True)
print('INDEP5 ALL MATCH' if ok_all else 'INDEP5 MISMATCH')
