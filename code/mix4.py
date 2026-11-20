"""Is each n=4 upper regime a two-vertex mixture (Lemma mix)?  u = AMP' (hhh h h h with y(0,3)=1) resp. AMP(1/2),
v = (1/2,1,1,0,1/2,0) in KEYS order (0,1),(0,2),(0,3),(1,1),(1,2),(2,1).  Checks:
 G(1,t)       = (1-2t) AMP' + 2t v      and  G(1/2+t,t) = (1-2t) AMP(1/2) + 2t v   (as vectors)
 Lemma-mix lambda == regcheck mu_one / mu_half, Lemma-mix Phi == opt_one / opt_half, t == t*/... ."""
import numpy as np, sympy as sp
import verify_exact as X
p = sp.Symbol('p'); q = 1 - p; t = sp.Symbol('t')
Z = np.load('hull_M_n4.npz'); M1, M2 = Z['M1'], Z['M2']; L = M1.shape[1] - 1
def idx(code):
    i = 0
    for ch in code: i = 3 * i + '0h1'.index(ch)
    assert ''.join('0h1'[c] for c in Z['verts'][i]) == code; return i
def E(code, w):
    row = (M1 if w == 1 else M2)[idx(code)]
    return sp.expand(sum(int(c) * p**k * q**(L - k) for k, c in enumerate(row)) / 2)
H = sp.Rational(1, 2)
vec = lambda c: [{'0': 0, 'h': H, '1': 1}[ch] for ch in c]
AMPp, AMP, V = 'hh1hhh', 'hhhhhh', 'h110h0'
G = lambda c3, tt: [H, H + tt, c3, H - tt, H, H - tt]
for name, u, Gv in (('one', AMPp, G(1, t)), ('half', AMP, G(H + t, t))):
    mixv = [sp.expand((1 - 2 * t) * a + 2 * t * b) for a, b in zip(vec(u), vec(V))]
    print(name, 'G == (1-2t)u + 2t v:', all(sp.expand(x - y) == 0 for x, y in zip(Gv, mixv)))
    au, bu, av, bv = E(u, 1), E(u, 2), E(V, 1), E(V, 2)
    lam = sp.cancel((bv - bu) / ((au - av) + (bv - bu)))
    Phi = sp.cancel((au * bv - av * bu) / ((au - bu) - (av - bv)))
    tt = sp.cancel((bv - av) / ((au - bu) + (bv - av)))   # weight on u
    opt = X.opt_one(p) if name == 'one' else X.opt_half(p)
    ts = X.tstar_one(p) if name == 'one' else X.tstar_half(p)
    c = 8*p**8 - 32*p**7 + 58*p**6 - 54*p**5 + 18*p**4 + 16*p**3 - 15*p**2 - p + 8
    mu = (2*p*(p - 1)*c / (12*p**10 - 50*p**9 + 84*p**8 - 49*p**7 - 63*p**6 + 167*p**5 - 168*p**4 + 85*p**3 - 12*p**2 - 4*p - 3)
          if name == 'one' else 2*p*c / (12*p**9 - 38*p**8 + 46*p**7 - 3*p**6 - 66*p**5 + 101*p**4 - 65*p**3 + 17*p**2 + 8*p + 3))
    print(name, 'Phi == opt:', sp.cancel(Phi - opt) == 0, ' lambda == mu:', sp.cancel(lam - mu) == 0,
          ' weight on v == 2 t*:', sp.cancel((1 - tt) - 2 * ts) == 0)
    print('   at p=1/2: lambda', lam.subs(p, H), ' mu', sp.nsimplify(mu.subs(p, H)))
