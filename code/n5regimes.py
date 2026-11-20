"""n=5 one round: exact regime formulas from hull_M_n5.npz.
A lambda=0 regime is named by one vertex u: OPT = E_2(u).
A mixture regime is named by a vertex pair (u, v): lambda solves lambda a_u + (1-lambda) b_u = lambda a_v + (1-lambda) b_v,
OPT = lambda a_u + (1-lambda) b_u   (a = E_1, b = E_2).
Prints each regime's OPT at a test point and the exact switch points between consecutive regimes (roots of the
difference of the two OPT formulas in a bracket).  usage: n5regimes.py"""
import numpy as np, sympy as sp
p = sp.Symbol('p'); q = 1 - p
Z = np.load('hull_M_n5.npz'); M1, M2 = Z['M1'], Z['M2']; L = M1.shape[1] - 1; m = Z['verts'].shape[1]

def idx(code):
    i = 0
    for ch in code: i = 3 * i + '0h1'.index(ch)
    assert ''.join('0h1'[c] for c in Z['verts'][i]) == code
    return i
def E(code, w):
    row = (M1 if w == 1 else M2)[idx(code)]
    return sp.expand(sum(int(c) * p**k * q**(L - k) for k, c in enumerate(row)) / 2)

def regime(r):
    if isinstance(r, str): return sp.Integer(0), E(r, 2)
    u, v = r; au, bu, av, bv = E(u, 1), E(u, 2), E(v, 1), E(v, 2)
    lam = sp.cancel((bv - bu) / ((au - av) + (bv - bu)))
    return lam, sp.cancel(lam * au + (1 - lam) * bu)

REG = [('low', '1110111000', 0.05), ('AMP', 'hhhhhhhhhh', 0.3),
       ('sliver', ('hhh1hhhhhh', 'hh11hhhhhh'), 0.4125),
       ('P1', ('hhh1hhhhhh', 'h1110h10h0'), 0.46),
       ('P2', ('hhhhhhhhhh', 'h1110h10h0'), 0.505),
       ('P3', ('hhhhhhhhhh', 'h1110h1000'), 0.6),
       ('P4', ('hhhhhhhhhh', 'h1110hh0h0'), 0.85)]
def build():
    """[(name, a, b, lambda, OPT)] with exact shared endpoints (CRootOf / Rational)"""
    F = {name: regime(r) for name, r, t in REG}
    ends = [sp.Integer(0)]
    for (n1, _, t1), (n2, _, t2) in zip(REG, REG[1:]):
        num = sp.Poly(sp.numer(sp.cancel(F[n1][1] - F[n2][1])), p)
        rs = [r for fac, _ in num.factor_list()[1] for r in sp.Poly(fac, p).real_roots(radicals=False) if t1 < float(r) < t2]
        assert len(rs) == 1, (n1, n2, rs)
        ends.append(rs[0])
    ends.append(sp.Integer(1))
    return [(name, ends[i], ends[i + 1], F[name][0], F[name][1]) for i, (name, _, _) in enumerate(REG)]

if __name__ != '__main__':
    pass
else:
  F = {}
  for name, r, t in REG:
    lam, opt = regime(r); F[name] = (lam, opt)
    print('%-7s lambda(%.4f)=%.6f OPT=%.12f' % (name, t, float(lam.subs(p, t)), float(opt.subs(p, t))))
  print()
  for (n1, _, t1), (n2, _, t2) in zip(REG, REG[1:]):
    num = sp.Poly(sp.numer(sp.cancel(F[n1][1] - F[n2][1])), p)
    rs = [r for fac, _ in num.factor_list()[1] for r in sp.Poly(fac, p).real_roots(radicals=False) if t1 < float(r) < t2]
    print('%s -> %s: roots in (%.4f, %.4f):' % (n1, n2, t1, t2), [round(float(r), 9) for r in rs],
          'factors:', [str(sp.Poly(f, p).as_expr())[:70] for f, _ in num.factor_list()[1]
                        if any(t1 < float(x) < t2 for x in sp.Poly(f, p).real_roots(radicals=False))])
if __name__ == '__main__':
  # lambda=0 -> mixture: where hhh1hhhhhh has E1 = E2
  d = sp.Poly(sp.numer(sp.cancel(E('hhh1hhhhhh', 1) - E('hhh1hhhhhh', 2))), p)
  print('E1=E2 on hhh1hhhhhh in (0.40, 0.42):', [round(float(r), 9) for r in d.real_roots(radicals=False) if 0.40 < float(r) < 0.42])
  print('E2(AMP) - E2(hhh1hhhhhh) identically zero:', sp.expand(E('hhhhhhhhhh', 2) - E('hhh1hhhhhh', 2)) == 0)
  print('AMP E_w formula 1-1/2[(1-q^w)^(5-w)+(1-q^(5-w))^w]:',
      [sp.expand(E('hhhhhhhhhh', w) - (1 - ((1 - q**w)**(5 - w) + (1 - q**(5 - w))**w) / 2)) == 0 for w in (1, 2)])
